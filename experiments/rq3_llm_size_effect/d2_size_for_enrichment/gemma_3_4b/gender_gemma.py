import os, json, time, re, requests, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from run_openrouter_experiments import load_dataset, budget, OUT

import random as _random, os as _os
_SEED = 42
_random.seed(_SEED); np.random.seed(_SEED)
_os.environ["PYTHONHASHSEED"] = str(_SEED)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "google/gemma-3-4b-it"

def call_gemma(messages, api_key, pos_label, neg_label, seed=42):
    schema = {"type": "object",
              "properties": {"label": {"type": "string", "enum": [pos_label, neg_label]},
                             "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
              "required": ["label", "confidence"], "additionalProperties": False}
    payload = {"model": MODEL_ID, "messages": messages, "max_tokens": 500,
               "temperature": 0, "seed": seed,
               "response_format": {"type": "json_schema",
                   "json_schema": {"name": "cls", "strict": True, "schema": schema}}}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            if r.status_code == 429: time.sleep(5*(attempt+1)); continue
            if r.status_code == 402:
                print(f"    [402 CREDITS EXHAUSTED]"); os._exit(2)
            if r.status_code != 200:
                if attempt == 2: return 0.5
                time.sleep(2); continue
            d = r.json()
            content = d["choices"][0].get("message", {}).get("content", "")
            u = d.get("usage", {})
            budget.add(u.get("prompt_tokens", 300), u.get("completion_tokens", 30), MODEL_ID)
            if not budget.check(): return 0.5
            try:
                p = json.loads(content)
                conf = max(0.05, min(0.95, float(p.get("confidence", 0.85))))
                return conf if p["label"].lower() == pos_label.lower() else 1 - conf
            except:
                m = re.search(r'"label"\s*:\s*"(\w+)"', content)
                if m: return 0.85 if m.group(1).lower() == pos_label.lower() else 0.15
                return 0.5
        except:
            if attempt == 2: return 0.5
            time.sleep(2)
    return 0.5

def run(dataset_name, strategy, api_key):
    data = load_dataset(dataset_name)
    cids_test, y_test = data["cids_test"], data["y_test"]
    pos_label, neg_label = data["pos_label"], data["neg_label"]

    cache = OUT / f"{dataset_name}_gemma2b_{strategy}.json"
    if cache.exists():
        c = json.load(open(cache))
        print(f"  Cached: AUC={c['auc']:.4f}"); return

    def process_one(i):
        cid = int(cids_test[i])
        profile = data["serialize"](cid)
        enrich = ""
        if strategy == "zero_shot_knn":
            k = data["knn_ctx"][cid]
            enrich = f"\nSimilar clients: {k['pos']} {pos_label}, {k['neg']} {neg_label}."
        user = (f"{profile}{enrich}\n\nClassify. Output JSON: "
                f'{{"label": "{pos_label}" or "{neg_label}", "confidence": 0-1}}')
        messages = [{"role": "system", "content": data["system_expert"]},
                    {"role": "user", "content": user}]
        return i, call_gemma(messages, api_key, pos_label, neg_label, seed=42)

    ckpt = OUT / f"{dataset_name}_gemma2b_{strategy}_ckpt.npz"
    if ckpt.exists():
        preds = list(np.load(ckpt)["preds"])
        start = len(preds); print(f"  Resuming from {start}")
    else:
        preds = []; start = 0

    if start == 0:
        print("  Sanity check...")
        for i in range(2):
            _, p = process_one(i)
            t = pos_label if y_test[i] == 1 else neg_label
            print(f"    {i}: true={t}, prob={p:.3f}")

    BATCH = 5; t0 = time.time()
    for bs in range(start, len(cids_test), BATCH):
        be = min(bs + BATCH, len(cids_test))
        with ThreadPoolExecutor(max_workers=BATCH) as ex:
            futs = [ex.submit(process_one, i) for i in range(bs, be)]
            br = {}
            for f in as_completed(futs):
                i, p = f.result(); br[i] = p
        for i in range(bs, be): preds.append(br[i])
        if len(preds) % 50 < BATCH:
            np.savez(ckpt, preds=np.array(preds))
            auc = roc_auc_score(y_test[:len(preds)], preds)
            rate = (len(preds)-start)/max(time.time()-t0, 0.1)
            print(f"    {len(preds)}/{len(cids_test)} ({rate:.1f}/s, AUC={auc:.4f})")
        if not budget.check():
            np.savez(ckpt, preds=np.array(preds)); return

    auc = roc_auc_score(y_test, preds)
    print(f"  Gemma 2B {strategy} {dataset_name}: AUC={auc:.4f}")
    with open(cache, "w") as f:
        json.dump({"auc": auc, "method": strategy, "model": "gemma-3n-e2b",
                   "dataset": dataset_name, "n_test": len(preds), "seed": 42}, f, indent=2)
    np.savez(OUT / f"{dataset_name}_gemma2b_{strategy}_preds.npz",
             preds=np.array(preds), cids=cids_test, y_test=y_test)
    if ckpt.exists(): ckpt.unlink()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["gender"])
    ap.add_argument("--strategies", nargs="*", default=["zero_shot_none", "zero_shot_knn"])
    ap.add_argument("--budget", type=float, default=0.5)
    args = ap.parse_args()
    budget.max_budget = args.budget
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key: exit("Set OPENROUTER_API_KEY")
    for ds in args.datasets:
        for s in args.strategies:
            if not budget.check(): break
            run(ds, s, api_key)
