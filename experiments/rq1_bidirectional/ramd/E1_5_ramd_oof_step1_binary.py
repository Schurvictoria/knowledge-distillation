#!/usr/bin/env python3
"""
RAMD-KD Step 1: Get OOF kNN-CoT predictions on TRAIN via OpenRouter.

For each strong LLM teacher (Qwen3.6, GLM-4.7, DeepSeek-V3.2):
  1. 5-fold CV on train set
  2. For each fold: kNN from other folds → LLM prompt → soft prediction
  3. Save OOF predictions as teacher labels for Step 2 (RAMD-KD fine-tune)

Cost estimate:
  - Qwen3.6: 7397 × ~400 tokens × $0.10/1M = ~$0.30
  - GLM-4.7 (reasoning=off): ~$0.30
  - DeepSeek-V3.2 (reasoning mandatory, 300 tokens): ~$0.80
  Total: ~$1.40
"""
import os, json, time, requests, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import NearestNeighbors

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from run_openrouter_experiments import (
    MODELS, budget, OUT, call_openrouter_logits
)

# ---- Reproducibility (seed=42) ----
import random as _random, os as _os
_SEED = 42
_random.seed(_SEED); np.random.seed(_SEED)
_os.environ["PYTHONHASHSEED"] = str(_SEED)


OUT_OOF = Path("results/ramd_openrouter")
OUT_OOF.mkdir(parents=True, exist_ok=True)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def get_oof_predictions(model_key, dataset_name, api_key):
    """Get OOF kNN-CoT predictions on train set via OpenRouter."""
    m = MODELS[model_key]
    cache_file = OUT_OOF / f"{dataset_name}_{model_key}_oof.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        print(f"  Loaded cached OOF: {cache_file}")
        return data["probs"], data["y"]

    print(f"\n=== OOF for {m['name']} on {dataset_name} ===", flush=True)

    # Load data
    coles_train = np.load(f"embeddings/{dataset_name}/emb_train_seed42.npy")
    cids_train = np.load(f"embeddings/{dataset_name}/cids_train_seed42.npy")
    y_train = np.load(f"embeddings/{dataset_name}/y_train_seed42.npy")

    # Load dataset info for labels and serialize
    import pandas as pd
    DATA = Path("data")
    MCC_GROUPS = {range(1,1500):"Agriculture",range(4000,4800):"Transportation",
                  range(5000,5600):"Retail",range(5600,5700):"Clothing",
                  range(5800,5900):"Restaurants",range(6000,7000):"Financial",
                  range(7500,7600):"Auto",range(8000,8100):"Medical",
                  range(8200,8300):"Education"}
    def mcc_cat(mcc):
        try: mcc=int(mcc)
        except: return "Other"
        for r,n in MCC_GROUPS.items():
            if mcc in r: return n
        return "Other"

    if dataset_name == "gender":
        tx = pd.read_csv(DATA / "transactions.csv")
        labels = pd.read_csv(DATA / "gender_train.csv")
        tx = tx[tx["customer_id"].isin(labels["customer_id"])].copy()
        grouped = tx.groupby("customer_id")
        pos_label, neg_label = "male", "female"
        task_desc = "gender (male or female)"
        system_expert = ("You are an expert bank analyst specializing in customer segmentation. "
                        "You also have analysis from an ML model.")
        def serialize(cid):
            if cid not in grouped.groups: return "No txns."
            ct = grouped.get_group(cid)
            n = len(ct)
            amt = np.abs(ct["amount"].values)
            cats = ct["mcc_code"].apply(mcc_cat).value_counts()
            top = ", ".join(f"{c} ({cnt} txns, {cnt*100//n}%)" for c, cnt in cats.head(6).items())
            return (f"Client profile:\n- Transactions: {n}\n"
                    f"- Spending: avg {amt.mean():.0f} RUB, median {np.median(amt):.0f}\n"
                    f"- Top categories: {top}")
    else:  # rosbank
        df = pd.read_csv(DATA / "rosbank_train.csv")
        df["dt"] = pd.to_datetime(df["TRDATETIME"], format="%d%b%y:%H:%M:%S")
        df = df.sort_values(["cl_id", "dt"])
        grouped = df.groupby("cl_id")
        pos_label, neg_label = "churn", "stay"
        task_desc = "churn (leave or stay)"
        system_expert = ("You are an expert bank analyst specializing in customer retention. "
                        "You also have analysis from an ML model.")
        def serialize(cid):
            if cid not in grouped.groups: return "No txns."
            ct = grouped.get_group(cid)
            n = len(ct)
            amt = np.abs(ct["amount"].values)
            cats = ct["MCC"].fillna(0).astype(int).apply(mcc_cat).value_counts()
            top = ", ".join(f"{c} ({cnt} txns, {cnt*100//n}%)" for c, cnt in cats.head(6).items())
            return (f"Client profile:\n- Transactions: {n}\n"
                    f"- Spending: avg {amt.mean():.0f} RUB, median {np.median(amt):.0f}\n"
                    f"- Top categories: {top}")

    answer_fmt = f"{pos_label} or {neg_label}"

    # 5-fold CV
    sc = MaxAbsScaler()
    coles_scaled = sc.fit_transform(coles_train)
    oof_probs = np.full(len(y_train), 0.5)

    # Checkpoint
    ckpt_file = OUT_OOF / f"{dataset_name}_{model_key}_oof_ckpt.npz"
    if ckpt_file.exists():
        ckpt = np.load(ckpt_file)
        oof_probs = ckpt["probs"].copy()
        done_mask = ckpt["done_mask"].copy()
        print(f"  Resumed from checkpoint: {done_mask.sum()}/{len(y_train)} done", flush=True)
    else:
        done_mask = np.zeros(len(y_train), dtype=bool)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def process_sample(vi, nb_idx):
        nb_labels = y_train[nb_idx]
        pos_count = int(nb_labels.sum())
        neg_count = 10 - pos_count
        majority = pos_label if pos_count > 5 else neg_label
        cid = int(cids_train[vi])
        profile = serialize(cid)

        user_msg = (f"{profile}\n\n"
                   f"Similar clients: {pos_count} {pos_label}, {neg_count} {neg_label} "
                   f"(majority: {majority}).\n\n"
                   f"Respond with ONLY one word: {answer_fmt}. Write your answer after ANSWER:")
        messages = [{"role": "system", "content": system_expert},
                    {"role": "user", "content": user_msg}]
        return vi, call_openrouter_logits(m["id"], messages, api_key, pos_label, neg_label)

    t0 = time.time()
    BATCH = 10
    for fold, (tr_idx, val_idx) in enumerate(skf.split(coles_train, y_train)):
        print(f"  Fold {fold+1}/5 ({len(val_idx)} samples)...", flush=True)
        # Filter already-done
        val_todo = [vi for vi in val_idx if not done_mask[vi]]
        if len(val_todo) == 0:
            print(f"    (all done)", flush=True)
            continue

        nn = NearestNeighbors(n_neighbors=10, metric="cosine")
        nn.fit(coles_scaled[tr_idx])
        _, idxs = nn.kneighbors(coles_scaled[val_todo])

        for batch_start in range(0, len(val_todo), BATCH):
            batch_vis = val_todo[batch_start:batch_start+BATCH]
            batch_nbs = [tr_idx[idxs[i+batch_start]] for i in range(len(batch_vis))]

            with ThreadPoolExecutor(max_workers=BATCH) as ex:
                futures = [ex.submit(process_sample, vi, nb) for vi, nb in zip(batch_vis, batch_nbs)]
                for f in as_completed(futures):
                    vi, prob = f.result()
                    if prob is None:
                        # Hard failure (HTTP error, exception, budget) — don't mark done; retry on next run
                        continue
                    oof_probs[vi] = prob
                    done_mask[vi] = True

            if (batch_start // BATCH) % 10 == 0:
                np.savez(ckpt_file, probs=oof_probs, done_mask=done_mask)
                elapsed = time.time() - t0
                done = done_mask.sum()
                print(f"    {done}/{len(y_train)} ({done/max(elapsed,0.1):.1f}/s)", flush=True)

            if not budget.check():
                np.savez(ckpt_file, probs=oof_probs, done_mask=done_mask)
                print(f"  Budget exceeded. Saved ckpt.", flush=True)
                return oof_probs, y_train

    oof_auc = roc_auc_score(y_train, oof_probs)
    print(f"  OOF AUC = {oof_auc:.4f}", flush=True)

    np.savez(cache_file, probs=oof_probs, y=y_train, cids=cids_train, auc=oof_auc)
    if ckpt_file.exists():
        ckpt_file.unlink()
    return oof_probs, y_train


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gender")
    ap.add_argument("--models", nargs="*", default=["qwen36_35b", "glm47", "deepseek_v3"])
    ap.add_argument("--budget", type=float, default=3.0)
    args = ap.parse_args()

    budget.max_budget = args.budget
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY")
        exit(1)

    print(f"OOF predictions for RAMD-KD, budget ${args.budget}", flush=True)

    for mk in args.models:
        try:
            oof, y = get_oof_predictions(mk, args.dataset, api_key)
            auc = roc_auc_score(y, oof)
            print(f"\n  {MODELS[mk]['name']}: OOF AUC = {auc:.4f}", flush=True)
        except Exception as e:
            print(f"\n  FAILED {mk}: {e}", flush=True)
            import traceback; traceback.print_exc()

