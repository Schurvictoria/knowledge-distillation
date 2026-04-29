import os, json, time, requests, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, ".")
from run_openrouter_experiments import load_dataset, MODELS, budget, OUT

import random as _random, os as _os
_SEED = 42
_random.seed(_SEED); np.random.seed(_SEED)
_os.environ["PYTHONHASHSEED"] = str(_SEED)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def call_with_thinking(model_id, messages, api_key, pos_label, neg_label, thinking_mode):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 500 if thinking_mode == "on" else (300 if "deepseek" in model_id else 10),
        "temperature": 0,
    }

    if thinking_mode == "off":
        payload["reasoning"] = {"enabled": False}
    elif thinking_mode == "on":
        payload["reasoning"] = {"enabled": True}

    for attempt in range(3):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if resp.status_code != 200:
                return 0.5
            data = resp.json()
            choice = data["choices"][0]
            msg = choice.get("message", {})
            content = msg.get("content") or ""
            usage = data.get("usage", {})
            budget.add(usage.get("prompt_tokens", 500),
                      usage.get("completion_tokens", 5), model_id)

            if not content.strip() and msg.get("reasoning"):
                reasoning = msg["reasoning"]
                r_lower = reasoning.lower()
                pos_l = pos_label.lower()
                neg_l = neg_label.lower()
                if "answer:" in r_lower:
                    after = r_lower.split("answer:")[-1].strip()[:20]
                    if pos_l in after: content = pos_label
                    elif neg_l in after: content = neg_label
                if not content:
                    tail = r_lower[-50:]
                    lp, ln = tail.rfind(pos_l), tail.rfind(neg_l)
                    if lp > ln: content = pos_label
                    elif ln > lp: content = neg_label
                    else: content = ""

            c = content.lower().strip()
            pos_l, neg_l = pos_label.lower(), neg_label.lower()
            if pos_l in c and neg_l not in c: return 0.85
            elif neg_l in c and pos_l not in c: return 0.15
            elif pos_l in c and neg_l in c:
                return 0.7 if c.rfind(pos_l) > c.rfind(neg_l) else 0.3
            return 0.5
        except Exception:
            if attempt == 2:
                return 0.5
            time.sleep(2)
    return 0.5

def run_experiment(model_key, thinking_mode, dataset_name, api_key):
    m = MODELS[model_key]
    data = load_dataset(dataset_name)

    tag = f"{model_key}_thinking_{thinking_mode}_knn"
    cache_file = OUT / f"{dataset_name}_{tag}.json"
    if cache_file.exists():
        cached = json.load(open(cache_file))
        print(f"  {m['name']} thinking={thinking_mode}: AUC={cached['auc']:.4f} (cached)")
        return cached["auc"]


    system = f"{data['system_expert']} You also have analysis from an ML model."

    def build_msg(i):
        cid = data["cids_test"][i]
        profile = data["serialize"](int(cid))
        k = data["knn_ctx"][cid]
        enrich = (f"\nSimilar clients: {k['pos']} {data['pos_label']}, "
                 f"{k['neg']} {data['neg_label']} (majority: {k['majority']}).\n")
        answer_inst = f"\n\nRespond with ONLY one word: {data['answer_fmt']}. Write your answer after ANSWER:"
        user = f"{profile}{enrich}{answer_inst}"
        return [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    checkpoint = OUT / f"{dataset_name}_{tag}_checkpoint.npz"
    if checkpoint.exists():
        ckpt = np.load(checkpoint)
        preds = list(ckpt["preds"])
        start = len(preds)
        print(f"  Resuming from {start}/{len(data['cids_test'])}")
    else:
        preds = []
        start = 0

    BATCH = 5
    t0 = time.time()
    for bstart in range(start, len(data["cids_test"]), BATCH):
        bend = min(bstart + BATCH, len(data["cids_test"]))
        batch_idx = list(range(bstart, bend))
        with ThreadPoolExecutor(max_workers=BATCH) as ex:
            futures = {ex.submit(call_with_thinking, m["id"], build_msg(i),
                                 api_key, data["pos_label"], data["neg_label"],
                                 thinking_mode): i for i in batch_idx}
            br = {futures[f]: f.result() for f in as_completed(futures)}
        for i in batch_idx:
            preds.append(br[i])

        if len(preds) % 100 < BATCH:
            np.savez(checkpoint, preds=np.array(preds))
            auc = roc_auc_score(data["y_test"][:len(preds)], preds)
            rate = (len(preds) - start) / max(time.time() - t0, 0.1)
            print(f"    {len(preds)}/{len(data['cids_test'])} ({rate:.1f}/s, AUC={auc:.4f})")

        if not budget.check():
            np.savez(checkpoint, preds=np.array(preds))
            return 0.5

    auc = roc_auc_score(data["y_test"], preds)
    print(f"  {m['name']} thinking={thinking_mode}: AUC={auc:.4f}")

    with open(cache_file, "w") as f:
        json.dump({"model": m["name"], "thinking": thinking_mode,
                  "auc": auc, "dataset": dataset_name}, f, indent=2)
    np.savez(OUT / f"{dataset_name}_{tag}_preds.npz",
             preds=np.array(preds), cids=data["cids_test"], y_test=data["y_test"])
    if checkpoint.exists():
        checkpoint.unlink()
    return auc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gender")
    ap.add_argument("--budget", type=float, default=3.0)
    args = ap.parse_args()

    budget.max_budget = args.budget
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY")
        exit(1)

    print(f"CoT Reasoning Effect: {args.dataset}, budget ${args.budget}")

    results = {}

    results["qwen36_off"] = run_experiment("qwen36_35b", "off", args.dataset, api_key)
    results["qwen36_on"] = run_experiment("qwen36_35b", "on", args.dataset, api_key)

    results["glm_off"] = run_experiment("glm47", "off", args.dataset, api_key)
    results["glm_on"] = run_experiment("glm47", "on", args.dataset, api_key)

    print("COT REASONING EFFECT SUMMARY")
    for key, auc in results.items():
        print(f"  {key}: AUC={auc:.4f}")
