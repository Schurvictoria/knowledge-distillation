#!/usr/bin/env python3
"""
Proper few-shot strategies for GLM-4.7:
1. Few-shot direct (kNN retrieval, balanced, shuffled)
2. Few-shot CoT (same + structured reasoning based on actual features)

Both use:
- k=4 demonstrations (2 male + 2 female)
- kNN selection (ближайшие в CoLES embedding space)
- Shuffle to avoid position bias
- JSON schema output
- reasoning=False (direct output)
"""
import os, json, time, re, random, requests, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import NearestNeighbors

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from run_openrouter_experiments import load_dataset, MODELS, budget, OUT

# ---- Reproducibility (seed=42) ----
import random as _random, os as _os
_SEED = 42
_random.seed(_SEED); np.random.seed(_SEED)
_os.environ["PYTHONHASHSEED"] = str(_SEED)



OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_glm(messages, api_key, pos_label, neg_label):
    """Single GLM call with JSON schema, reasoning=off."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "label": {"type": "string", "enum": [pos_label, neg_label]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["label", "confidence"],
        "additionalProperties": False,
    }
    payload = {
        "model": "z-ai/glm-4.7",
        "messages": messages,
        "max_tokens": 400,
        "temperature": 0,
        "reasoning": {"enabled": False},
        "response_format": {"type": "json_schema",
            "json_schema": {"name": "cls", "strict": True, "schema": schema}},
    }

    for attempt in range(3):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429: time.sleep(5 * (attempt+1)); continue
            if resp.status_code != 200:
                if attempt == 2: return 0.5
                time.sleep(2); continue
            data = resp.json()
            msg = data["choices"][0].get("message", {})
            content = msg.get("content") or ""
            usage = data.get("usage", {})
            budget.add(usage.get("prompt_tokens", 500),
                      usage.get("completion_tokens", 30), "z-ai/glm-4.7")
            if not budget.check(): return 0.5

            try:
                parsed = json.loads(content)
                label = parsed["label"].lower()
                conf = max(0.05, min(0.95, float(parsed.get("confidence", 0.85))))
                return conf if label == pos_label.lower() else 1 - conf
            except:
                # Fallback
                m = re.search(r'"label"[:\s]*"(\w+)"', content)
                if m:
                    return 0.85 if m.group(1).lower() == pos_label.lower() else 0.15
                return 0.5
        except:
            if attempt == 2: return 0.5
            time.sleep(2)
    return 0.5


def build_knn_few_shot(query_idx, query_emb, train_embeddings, train_cids, y_train,
                       pos_label, neg_label, serialize, k_per_class=2, rng=None):
    """
    Select k_per_class nearest neighbors from each class for few-shot demos.
    Returns list of dicts: [{profile, label, cid}, ...], shuffled.
    """
    sc = MaxAbsScaler()
    # Fit on train, transform everything
    tr_scaled = sc.fit_transform(train_embeddings)
    q_scaled = sc.transform(query_emb.reshape(1, -1))

    # Separate pos and neg indices
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    # kNN for each class
    def top_k(indices, k):
        nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn.fit(tr_scaled[indices])
        _, idxs = nn.kneighbors(q_scaled)
        return indices[idxs[0]]

    pos_nn = top_k(pos_idx, k_per_class)
    neg_nn = top_k(neg_idx, k_per_class)

    demos = []
    for idx in pos_nn:
        cid = int(train_cids[idx])
        demos.append({"profile": serialize(cid), "label": pos_label, "cid": cid})
    for idx in neg_nn:
        cid = int(train_cids[idx])
        demos.append({"profile": serialize(cid), "label": neg_label, "cid": cid})

    # Shuffle to avoid position bias
    if rng: rng.shuffle(demos)
    else: random.shuffle(demos)
    return demos


def build_cot_reasoning(profile, label):
    """Generate structured reasoning based on actual profile features."""
    # Parse key stats from profile
    txns_match = re.search(r'Transactions:\s*(\d+)', profile)
    avg_match = re.search(r'avg\s*(\d+)', profile)
    cats_match = re.search(r'Top categories:\s*([^\n]+)', profile)

    n_txns = int(txns_match.group(1)) if txns_match else 0
    avg_amt = int(avg_match.group(1)) if avg_match else 0
    top_cats = cats_match.group(1) if cats_match else ""

    # Generate reasoning based on patterns
    reasoning_parts = [
        f"Client has {n_txns} transactions with average amount {avg_amt} RUB.",
        f"Top spending categories: {top_cats[:100]}.",
    ]

    # Simple pattern analysis (domain knowledge)
    if "Retail" in top_cats or "Clothing" in top_cats:
        reasoning_parts.append("High retail/clothing spending is a characteristic pattern.")
    if "Transportation" in top_cats:
        reasoning_parts.append("Significant transportation spending observed.")
    if avg_amt > 3000:
        reasoning_parts.append("Higher-than-average ticket size suggests premium behavior.")

    reasoning_parts.append(f"Based on these patterns, classification: {label}.")
    return " ".join(reasoning_parts)


def run_strategy(strategy, dataset_name, api_key):
    """
    strategy: 'few_shot_knn' or 'few_shot_cot_knn'
    """
    print(f"\n=== {strategy} (GLM-4.7, kNN k=4 balanced) ===", flush=True)

    data = load_dataset(dataset_name)
    # Load train embeddings for kNN
    train_emb = np.load(f"embeddings/{dataset_name}/emb_train_seed42.npy")
    test_emb = np.load(f"embeddings/{dataset_name}/emb_test_seed42.npy")
    train_cids = np.load(f"embeddings/{dataset_name}/cids_train_seed42.npy")
    y_train = np.load(f"embeddings/{dataset_name}/y_train_seed42.npy")

    cache = OUT / f"{dataset_name}_glm47_{strategy}.json"
    if cache.exists():
        c = json.load(open(cache))
        print(f"  Cached: AUC={c['auc']:.4f}", flush=True)
        return c["auc"]

    # Sanity check on 5 clients
    print("  === Sanity check on 5 clients ===", flush=True)
    rng = random.Random(42)
    for i in range(5):
        cid = int(data["cids_test"][i])
        query_emb = test_emb[i]
        demos = build_knn_few_shot(i, query_emb, train_emb, train_cids, y_train,
                                    data["pos_label"], data["neg_label"],
                                    data["serialize"], k_per_class=2, rng=random.Random(42+i))

        messages = [{"role": "system", "content": data["system_expert"]}]
        for d in demos:
            messages.append({"role": "user", "content": f"{d['profile']}\n\nClassify this client."})
            if strategy == "few_shot_cot_knn":
                reasoning = build_cot_reasoning(d["profile"], d["label"])
                messages.append({"role": "assistant", "content":
                    json.dumps({"reasoning": reasoning, "label": d["label"], "confidence": 0.9})})
            else:
                messages.append({"role": "assistant", "content":
                    json.dumps({"label": d["label"], "confidence": 0.9})})

        # Query
        profile = data["serialize"](cid)
        k = data["knn_ctx"][cid]
        enrich = (f"\nSimilar clients: {k['pos']} {data['pos_label']}, "
                 f"{k['neg']} {data['neg_label']}.")
        user_q = (f"{profile}{enrich}\n\nClassify this client. "
                 f'Output JSON: {{"label": "{data["pos_label"]}" or "{data["neg_label"]}", "confidence": 0-1}}'
                 if strategy == "few_shot_knn" else
                 f"{profile}{enrich}\n\nClassify this client. "
                 f'Output JSON: {{"reasoning": "your analysis", "label": "{data["pos_label"]}" or "{data["neg_label"]}", "confidence": 0-1}}')
        messages.append({"role": "user", "content": user_q})

        prob = call_glm(messages, api_key, data["pos_label"], data["neg_label"])
        true = data["pos_label"] if data["y_test"][i] == 1 else data["neg_label"]
        print(f"    Client {i}: true={true}, prob={prob:.3f}, {len(demos)} demos", flush=True)

    # Full run
    print(f"\n  === Full run on {len(data['cids_test'])} clients ===", flush=True)
    ckpt = OUT / f"{dataset_name}_glm47_{strategy}_ckpt.npz"
    if ckpt.exists():
        preds = list(np.load(ckpt)["preds"])
        start = len(preds)
        print(f"  Resuming from {start}", flush=True)
    else:
        preds = []; start = 0

    def process_one(i):
        cid = int(data["cids_test"][i])
        query_emb = test_emb[i]
        demos = build_knn_few_shot(i, query_emb, train_emb, train_cids, y_train,
                                    data["pos_label"], data["neg_label"],
                                    data["serialize"], k_per_class=2, rng=random.Random(42+i))

        messages = [{"role": "system", "content": data["system_expert"]}]
        for d in demos:
            messages.append({"role": "user", "content": f"{d['profile']}\n\nClassify this client."})
            if strategy == "few_shot_cot_knn":
                reasoning = build_cot_reasoning(d["profile"], d["label"])
                messages.append({"role": "assistant", "content":
                    json.dumps({"reasoning": reasoning, "label": d["label"], "confidence": 0.9})})
            else:
                messages.append({"role": "assistant", "content":
                    json.dumps({"label": d["label"], "confidence": 0.9})})

        profile = data["serialize"](cid)
        k = data["knn_ctx"][cid]
        enrich = (f"\nSimilar clients: {k['pos']} {data['pos_label']}, "
                 f"{k['neg']} {data['neg_label']}.")
        user_q = (f"{profile}{enrich}\n\nClassify this client. "
                 f'Output JSON: {{"label": "{data["pos_label"]}" or "{data["neg_label"]}", "confidence": 0-1}}'
                 if strategy == "few_shot_knn" else
                 f"{profile}{enrich}\n\nClassify this client. "
                 f'Output JSON: {{"reasoning": "your analysis", "label": "{data["pos_label"]}" or "{data["neg_label"]}", "confidence": 0-1}}')
        messages.append({"role": "user", "content": user_q})

        return i, call_glm(messages, api_key, data["pos_label"], data["neg_label"])

    BATCH = 5
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
            rate = (len(preds)-start) / max(time.time()-t0, 0.1)
            print(f"    {len(preds)}/{len(data['cids_test'])} ({rate:.1f}/s, AUC={auc:.4f})", flush=True)

        if not budget.check():
            np.savez(ckpt, preds=np.array(preds)); return 0.5

    auc = roc_auc_score(data["y_test"], preds)
    print(f"\n  {strategy}: AUC={auc:.4f}", flush=True)

    with open(cache, "w") as f:
        json.dump({"auc": auc, "method": strategy, "n_test": len(preds)}, f, indent=2)
    np.savez(OUT / f"{dataset_name}_glm47_{strategy}_preds.npz",
             preds=np.array(preds), cids=data["cids_test"], y_test=data["y_test"])
    if ckpt.exists(): ckpt.unlink()
    return auc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gender")
    ap.add_argument("--budget", type=float, default=1.5)
    args = ap.parse_args()

    budget.max_budget = args.budget
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key: exit("Set OPENROUTER_API_KEY")

    print(f"Proper few-shot for GLM-4.7, budget ${args.budget}", flush=True)

    auc_fs = run_strategy("few_shot_knn", args.dataset, api_key)
    auc_fscot = run_strategy("few_shot_cot_knn", args.dataset, api_key)

    print(f"\n{'='*50}")
    print(f"FINAL RESULTS (GLM-4.7, Gender, with kNN retrieval):")
    print(f"  Few-shot (kNN, k=4 balanced):     AUC={auc_fs:.4f}")
    print(f"  Few-shot CoT (kNN, reasoning):    AUC={auc_fscot:.4f}")
    print(f"  vs Zero-shot + JSON (prev):       AUC=0.7712")
