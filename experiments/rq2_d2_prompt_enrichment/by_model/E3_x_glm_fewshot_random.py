#!/usr/bin/env python3
"""
Few-shot 2 random baseline + Rosbank experiments for GLM-4.7.

Strategies:
1. zero_shot_knn (enrichment)
2. few_shot_random (2 random demos, no kNN, to compare with few_shot_knn)
3. few_shot_knn (k=4 balanced, kNN retrieval)
4. few_shot_cot_knn (k=4 + reasoning in demos)

Datasets: gender, rosbank
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
from run_openrouter_experiments import load_dataset, budget, OUT
from E3_x_glm_fewshot_proper import call_glm, build_cot_reasoning

# ---- Reproducibility (seed=42) ----
import random as _random, os as _os
_SEED = 42
_random.seed(_SEED); np.random.seed(_SEED)
_os.environ["PYTHONHASHSEED"] = str(_SEED)



OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def build_random_few_shot(pos_cids, neg_cids, y_train, train_cids_idx,
                           pos_label, neg_label, serialize, k_per_class=1, rng=None):
    """Select k_per_class RANDOM examples from each class (baseline for kNN)."""
    if rng is None: rng = random.Random(42)
    pos_chosen = rng.sample(list(pos_cids), k_per_class)
    neg_chosen = rng.sample(list(neg_cids), k_per_class)

    demos = []
    for cid in pos_chosen:
        demos.append({"profile": serialize(int(cid)), "label": pos_label})
    for cid in neg_chosen:
        demos.append({"profile": serialize(int(cid)), "label": neg_label})
    rng.shuffle(demos)
    return demos


def build_knn_few_shot(query_emb, train_embeddings, train_cids, y_train,
                        pos_label, neg_label, serialize, k_per_class=2, rng=None):
    sc = MaxAbsScaler()
    tr_scaled = sc.fit_transform(train_embeddings)
    q_scaled = sc.transform(query_emb.reshape(1, -1))

    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    def top_k(indices, k):
        nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn.fit(tr_scaled[indices])
        _, idxs = nn.kneighbors(q_scaled)
        return indices[idxs[0]]

    pos_nn = top_k(pos_idx, k_per_class)
    neg_nn = top_k(neg_idx, k_per_class)

    demos = []
    for idx in pos_nn:
        demos.append({"profile": serialize(int(train_cids[idx])), "label": pos_label})
    for idx in neg_nn:
        demos.append({"profile": serialize(int(train_cids[idx])), "label": neg_label})
    if rng: rng.shuffle(demos)
    return demos


def run_strategy(strategy, dataset_name, api_key):
    """
    strategy: zero_shot_knn | few_shot_random | few_shot_knn | few_shot_cot_knn
    """
    print(f"\n=== {strategy} on {dataset_name} ===", flush=True)

    data = load_dataset(dataset_name)
    train_emb = np.load(f"embeddings/{dataset_name}/emb_train_seed42.npy")
    test_emb = np.load(f"embeddings/{dataset_name}/emb_test_seed42.npy")
    train_cids = np.load(f"embeddings/{dataset_name}/cids_train_seed42.npy")
    y_train = np.load(f"embeddings/{dataset_name}/y_train_seed42.npy")
    pos_cids = train_cids[y_train == 1]
    neg_cids = train_cids[y_train == 0]

    cache = OUT / f"{dataset_name}_glm47_{strategy}.json"
    if cache.exists():
        c = json.load(open(cache))
        print(f"  Cached: AUC={c['auc']:.4f}", flush=True)
        return c["auc"]

    def process_one(i):
        cid = int(data["cids_test"][i])
        query_emb = test_emb[i]

        messages = [{"role": "system", "content": data["system_expert"]}]

        # Build demos based on strategy
        if strategy == "zero_shot_knn":
            demos = []
        elif strategy == "few_shot_random":
            demos = build_random_few_shot(pos_cids, neg_cids, y_train, train_cids,
                                           data["pos_label"], data["neg_label"],
                                           data["serialize"], k_per_class=1, rng=random.Random(42+i))
        else:  # few_shot_knn or few_shot_cot_knn
            demos = build_knn_few_shot(query_emb, train_emb, train_cids, y_train,
                                        data["pos_label"], data["neg_label"],
                                        data["serialize"], k_per_class=2, rng=random.Random(42+i))

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
        if strategy == "few_shot_cot_knn":
            user_q = (f"{profile}{enrich}\n\nClassify. Output JSON: "
                     f'{{"reasoning": "analysis", "label": "{data["pos_label"]}" or "{data["neg_label"]}", "confidence": 0-1}}')
        else:
            user_q = (f"{profile}{enrich}\n\nClassify. Output JSON: "
                     f'{{"label": "{data["pos_label"]}" or "{data["neg_label"]}", "confidence": 0-1}}')
        messages.append({"role": "user", "content": user_q})

        return i, call_glm(messages, api_key, data["pos_label"], data["neg_label"])

    ckpt = OUT / f"{dataset_name}_glm47_{strategy}_ckpt.npz"
    if ckpt.exists():
        preds = list(np.load(ckpt)["preds"])
        start = len(preds)
        print(f"  Resuming from {start}", flush=True)
    else:
        preds = []; start = 0

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
            np.savez(ckpt, preds=np.array(preds))
            return 0.5

    auc = roc_auc_score(data["y_test"], preds)
    print(f"\n  {strategy} on {dataset_name}: AUC={auc:.4f}", flush=True)

    with open(cache, "w") as f:
        json.dump({"auc": auc, "method": strategy, "dataset": dataset_name, "n_test": len(preds)}, f, indent=2)
    np.savez(OUT / f"{dataset_name}_glm47_{strategy}_preds.npz",
             preds=np.array(preds), cids=data["cids_test"], y_test=data["y_test"])
    if ckpt.exists(): ckpt.unlink()
    return auc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1.5)
    args = ap.parse_args()

    budget.max_budget = args.budget
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key: exit("Set OPENROUTER_API_KEY")

    results = {}

    # Gender: add few_shot_random (existing: zero_shot_knn=0.7712, few_shot_knn=0.7894, few_shot_cot_knn=0.7803)
    print("\n" + "="*60)
    print("GENDER: add few-shot random baseline")
    print("="*60, flush=True)
    results["gender_few_shot_random"] = run_strategy("few_shot_random", "gender", api_key)

    # Rosbank: all 4 strategies
    print("\n" + "="*60)
    print("ROSBANK: all strategies")
    print("="*60, flush=True)
    for strat in ["zero_shot_knn", "few_shot_random", "few_shot_knn", "few_shot_cot_knn"]:
        if not budget.check(): break
        results[f"rosbank_{strat}"] = run_strategy(strat, "rosbank", api_key)

    print("\n" + "="*60)
    print("FINAL SUMMARY (GLM-4.7)")
    print("="*60)
    print("\nGENDER:")
    print(f"  Zero-shot kNN:        0.7712 (cached)")
    print(f"  Few-shot random (2):  {results.get('gender_few_shot_random', '?'):.4f}")
    print(f"  Few-shot kNN (k=4):   0.7894 (cached)")
    print(f"  Few-shot CoT kNN:     0.7803 (cached)")
    print("\nROSBANK:")
    for strat in ["zero_shot_knn", "few_shot_random", "few_shot_knn", "few_shot_cot_knn"]:
        val = results.get(f'rosbank_{strat}')
        print(f"  {strat:<25} {val:.4f}" if val else f"  {strat:<25} ?")
