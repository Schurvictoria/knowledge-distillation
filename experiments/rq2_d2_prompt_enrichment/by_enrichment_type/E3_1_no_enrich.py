#!/usr/bin/env python3
"""
No-enrichment zero-shot baseline — fills RQ3 D2 Δ column.
Runs zero-shot WITHOUT kNN enrichment for GLM, DeepSeek, Qwen3.6 on Gender/Rosbank.
Proper max_tokens + JSON schema.
"""
import os, json, time, re, random, requests, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from run_openrouter_experiments import load_dataset, budget, OUT

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "glm47": {"id": "z-ai/glm-4.7", "max_tokens": 400, "reasoning": {"enabled": False}},
    "qwen36": {"id": "qwen/qwen3.6-plus", "max_tokens": 4096, "reasoning": {"max_tokens": 3000}},
    "deepseek_v3": {"id": "deepseek/deepseek-v3.2-speciale", "max_tokens": 8192, "reasoning": {"max_tokens": 6000}},
    "gemma_2b": {"id": "google/gemma-3n-e2b-it", "max_tokens": 500, "reasoning": {"enabled": False}},
}


def call_model(mcfg, messages, api_key, pos_label, neg_label, seed=42):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": [pos_label, neg_label]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["label", "confidence"],
        "additionalProperties": False,
    }
    payload = {
        "model": mcfg["id"], "messages": messages,
        "max_tokens": mcfg["max_tokens"],
        "temperature": 0.6 if "deepseek" in mcfg["id"] or "qwen3.6" in mcfg["id"] else 0,
        "seed": seed,
        "reasoning": mcfg["reasoning"],
        "response_format": {"type": "json_schema",
            "json_schema": {"name": "cls", "strict": True, "schema": schema}},
    }

    for attempt in range(3):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=180)
            if resp.status_code == 429: time.sleep(5*(attempt+1)); continue
            if resp.status_code == 402:
                print(f"    [402 CREDITS EXHAUSTED] Halting.", flush=True)
                import os; os._exit(2)
            if resp.status_code != 200:
                if attempt == 2: return 0.5
                time.sleep(2); continue
            data = resp.json()
            msg = data["choices"][0].get("message", {})
            content = msg.get("content") or ""
            usage = data.get("usage", {})
            budget.add(usage.get("prompt_tokens", 500),
                      usage.get("completion_tokens", 100), mcfg["id"])
            if not budget.check(): return 0.5

            try:
                parsed = json.loads(content)
                label = parsed["label"].lower()
                conf = max(0.05, min(0.95, float(parsed.get("confidence", 0.85))))
                return conf if label == pos_label.lower() else 1 - conf
            except:
                combined = content + " " + (msg.get("reasoning") or "")
                m = re.search(r'\{[^{}]*"label"[^{}]*\}', combined)
                if m:
                    try:
                        parsed = json.loads(m.group())
                        label = parsed["label"].lower()
                        conf = max(0.05, min(0.95, float(parsed.get("confidence", 0.85))))
                        return conf if label == pos_label.lower() else 1 - conf
                    except: pass
                c = combined.lower()
                lp, ln = c.rfind(pos_label.lower()), c.rfind(neg_label.lower())
                if lp > ln: return 0.85
                elif ln > lp: return 0.15
                return 0.5
        except:
            if attempt == 2: return 0.5
            time.sleep(2)
    return 0.5


def run_baseline(model_key, dataset_name, api_key):
    mcfg = MODELS[model_key]
    print(f"\n=== {model_key} no-enrich zero-shot on {dataset_name} ===", flush=True)
    data = load_dataset(dataset_name)

    cache = OUT / f"{dataset_name}_{model_key}_zero_shot_noenrich.json"
    if cache.exists():
        c = json.load(open(cache))
        print(f"  Cached: AUC={c['auc']:.4f}", flush=True)
        return c["auc"]

    def process_one(i):
        cid = int(data["cids_test"][i])
        profile = data["serialize"](cid)
        messages = [
            {"role": "system", "content": data["system_expert"]},
            {"role": "user", "content": (
                f"{profile}\n\nClassify this client. Output JSON: "
                f'{{"label": "{data["pos_label"]}" or "{data["neg_label"]}", "confidence": 0-1}}')}
        ]
        return i, call_model(mcfg, messages, api_key, data["pos_label"], data["neg_label"], seed=42)

    ckpt = OUT / f"{dataset_name}_{model_key}_zero_shot_noenrich_ckpt.npz"
    if ckpt.exists():
        preds = list(np.load(ckpt)["preds"])
        start = len(preds)
        print(f"  Resuming from {start}", flush=True)
    else:
        preds = []; start = 0

    # sanity
    if start == 0:
        print("  Sanity check...", flush=True)
        for i in range(2):
            _, p = process_one(i)
            t = data["pos_label"] if data["y_test"][i] == 1 else data["neg_label"]
            print(f"    {i}: true={t}, prob={p:.3f}", flush=True)

    BATCH = 5 if "glm" in model_key or "qwen" in model_key else 3
    t0 = time.time()
    for bs in range(start, len(data["cids_test"]), BATCH):
        be = min(bs + BATCH, len(data["cids_test"]))
        with ThreadPoolExecutor(max_workers=BATCH) as ex:
            futs = [ex.submit(process_one, i) for i in range(bs, be)]
            br = {}
            for f in as_completed(futs):
                i, p = f.result()
                br[i] = p
        for i in range(bs, be):
            preds.append(br[i])

        if len(preds) % 50 < BATCH:
            np.savez(ckpt, preds=np.array(preds))
            auc = roc_auc_score(data["y_test"][:len(preds)], preds)
            rate = (len(preds)-start)/max(time.time()-t0, 0.1)
            print(f"    {len(preds)}/{len(data['cids_test'])} ({rate:.1f}/s, AUC={auc:.4f})", flush=True)

        if not budget.check():
            np.savez(ckpt, preds=np.array(preds))
            return 0.5

    auc = roc_auc_score(data["y_test"], preds)
    print(f"\n  {model_key} no-enrich on {dataset_name}: AUC={auc:.4f}", flush=True)
    with open(cache, "w") as f:
        json.dump({"auc": auc, "method": "zero_shot_noenrich", "model": model_key,
                   "dataset": dataset_name, "n_test": len(preds)}, f, indent=2)
    np.savez(OUT / f"{dataset_name}_{model_key}_zero_shot_noenrich_preds.npz",
             preds=np.array(preds), cids=data["cids_test"], y_test=data["y_test"])
    if ckpt.exists(): ckpt.unlink()
    return auc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["glm47", "qwen36"])
    ap.add_argument("--datasets", nargs="*", default=["gender", "rosbank"])
    ap.add_argument("--budget", type=float, default=3.0)
    args = ap.parse_args()

    budget.max_budget = args.budget
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key: exit("Set OPENROUTER_API_KEY")

    for ds in args.datasets:
        for mk in args.models:
            if not budget.check(): break
            run_baseline(mk, ds, api_key)

    print("\n" + "="*60)
    print("FINAL SUMMARY (no-enrich zero-shot)")
    print("="*60)
