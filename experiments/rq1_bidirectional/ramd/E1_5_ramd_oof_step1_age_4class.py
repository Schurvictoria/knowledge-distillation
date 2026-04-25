#!/usr/bin/env python3
"""
RAMD-KD Step 1 for Age (4-class). OOF teacher predictions via OpenRouter.
seed=42, checkpoint каждые 100 calls.
"""
import os, json, time, re, requests, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import NearestNeighbors

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from run_openrouter_experiments import MODELS, budget

# ---- Reproducibility (seed=42) ----
import random as _random, os as _os
_SEED = 42
_random.seed(_SEED); np.random.seed(_SEED)
_os.environ["PYTHONHASHSEED"] = str(_SEED)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OUT_OOF = Path("results/ramd_openrouter")
OUT_OOF.mkdir(parents=True, exist_ok=True)


def call_4class(model_id, messages, api_key, seed=42):
    schema = {"type": "object",
              "properties": {"reasoning": {"type": "string"},
                             "label": {"type": "integer", "enum": [0, 1, 2, 3]},
                             "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
              "required": ["label", "confidence"], "additionalProperties": False}
    # Choose max_tokens by model
    max_tok = 8192 if "deepseek" in model_id else (4096 if "qwen" in model_id.lower() else 500)
    reasoning = {"max_tokens": max_tok // 2} if "deepseek" in model_id or "qwen" in model_id.lower() else {"enabled": False}
    payload = {"model": model_id, "messages": messages, "max_tokens": max_tok,
               "temperature": 0.6 if reasoning != {"enabled": False} else 0, "seed": seed,
               "reasoning": reasoning,
               "response_format": {"type": "json_schema",
                   "json_schema": {"name": "cls", "strict": True, "schema": schema}}}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            if r.status_code == 429: time.sleep(5*(attempt+1)); continue
            if r.status_code == 402:
                print(f"    [402 CREDITS EXHAUSTED]", flush=True); os._exit(2)
            if r.status_code != 200:
                if attempt == 2: return None  # Sentinel: hard failure, don't mark done
                time.sleep(2); continue
            d = r.json()
            msg = d["choices"][0].get("message", {})
            content = msg.get("content", "") or ""
            u = d.get("usage", {})
            budget.add(u.get("prompt_tokens", 500), u.get("completion_tokens", 100), model_id)
            if not budget.check(): return None
            try:
                p = json.loads(content)
                label = int(p["label"])
                conf = max(0.3, min(0.95, float(p.get("confidence", 0.7))))
                probs = [(1-conf)/3] * 4
                probs[label] = conf
                return tuple(probs)
            except:
                combined = content + " " + (msg.get("reasoning") or "")
                m = re.search(r'"label"\s*:\s*(\d)', combined)
                if m:
                    label = int(m.group(1))
                    probs = [0.1] * 4; probs[label] = 0.7
                    return tuple(probs)
                return None  # parse failed — retry next run
        except:
            if attempt == 2: return None
            time.sleep(2)
    return None


def get_oof_age(model_key, api_key):
    m = MODELS[model_key]
    cache = OUT_OOF / f"age_{model_key}_oof.npz"
    if cache.exists():
        d = np.load(cache)
        print(f"  Cached: {cache}", flush=True)
        return d["probs"], d["y"]

    print(f"\n=== OOF for {m['name']} on age (4-class) ===", flush=True)
    coles_train = np.load("embeddings/age/emb_train_seed42.npy")
    cids_train = np.load("embeddings/age/cids_train_seed42.npy")
    y_train = np.load("embeddings/age/y_train_seed42.npy")

    tx = pd.read_csv("data/transactions_train.csv")
    grouped = tx.groupby("client_id")

    def serialize(cid):
        if cid not in grouped.groups: return "No txns."
        ct = grouped.get_group(cid)
        a = np.abs(ct["amount_rur"].values)
        mcc = ct["small_group"].value_counts().head(5).to_dict()
        return (f"Transactions: {len(ct)}\n"
                f"Spending: avg {a.mean():.0f} RUB, median {np.median(a):.0f}, max {a.max():.0f}\n"
                f"Top 5 categories: {list(mcc.items())[:5]}")

    AGE_LABELS = {0: "youngest (0)", 1: "young-adult (1)", 2: "middle-age (2)", 3: "senior (3)"}
    system = ("You are an expert bank analyst. Predict age group (0=youngest, 1=young-adult, "
              "2=middle-age, 3=senior) from transaction patterns. "
              "You have similar clients from a retrieval system as additional context.")

    sc = MaxAbsScaler()
    coles_scaled = sc.fit_transform(coles_train)

    oof_probs = np.full((len(y_train), 4), 0.25, dtype=np.float32)
    done_mask = np.zeros(len(y_train), dtype=bool)

    ckpt_file = OUT_OOF / f"age_{model_key}_oof_ckpt.npz"
    if ckpt_file.exists():
        ck = np.load(ckpt_file)
        if ck["probs"].ndim == 2 and ck["probs"].shape[1] == 4:
            oof_probs = ck["probs"].copy()
            done_mask = ck["done_mask"].copy()
            print(f"  Resumed: {done_mask.sum()}/{len(y_train)} done", flush=True)
        else:
            print(f"  Old binary ckpt found — ignoring (4-class fresh start)", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def process_one(vi, nb_idx):
        nb_labels = y_train[nb_idx]
        class_counts = [(nb_labels == c).sum() for c in range(4)]
        neighbors_str = ", ".join(f"{class_counts[c]} class-{c}" for c in range(4))
        cid = int(cids_train[vi])
        profile = serialize(cid)
        user = (f"{profile}\n\nSimilar clients (kNN from structured model): {neighbors_str}.\n\n"
                f"Classify age group. Output JSON: "
                f'{{"reasoning": "brief", "label": 0|1|2|3, "confidence": 0-1}}')
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        probs = call_4class(m["id"], messages, api_key, seed=42)
        return vi, probs

    t0 = time.time()
    BATCH = 15
    for fold, (tr_idx, val_idx) in enumerate(skf.split(coles_train, y_train)):
        print(f"  Fold {fold+1}/5 ({len(val_idx)} samples)...", flush=True)
        val_todo = [vi for vi in val_idx if not done_mask[vi]]
        if not val_todo: print(f"    (all done)", flush=True); continue

        nn = NearestNeighbors(n_neighbors=10, metric="cosine").fit(coles_scaled[tr_idx])
        _, idxs = nn.kneighbors(coles_scaled[val_todo])

        for bs in range(0, len(val_todo), BATCH):
            batch_vis = val_todo[bs:bs+BATCH]
            batch_nbs = [tr_idx[idxs[i+bs]] for i in range(len(batch_vis))]
            with ThreadPoolExecutor(max_workers=BATCH) as ex:
                futs = [ex.submit(process_one, vi, nb) for vi, nb in zip(batch_vis, batch_nbs)]
                for f in as_completed(futs):
                    vi, probs = f.result()
                    if probs is None:
                        # Hard failure — don't mark done; retry next run
                        continue
                    oof_probs[vi] = probs
                    done_mask[vi] = True
            if (bs // BATCH) % 20 == 0:
                np.savez(ckpt_file, probs=oof_probs, done_mask=done_mask)
                done = done_mask.sum(); rate = done / max(time.time()-t0, 0.1)
                print(f"    {done}/{len(y_train)} ({rate:.1f}/s)", flush=True)
            if not budget.check():
                np.savez(ckpt_file, probs=oof_probs, done_mask=done_mask)
                print(f"  Budget exceeded. Saved ckpt.", flush=True)
                return oof_probs, y_train

    # Final
    pred = oof_probs.argmax(axis=1)
    acc = accuracy_score(y_train[done_mask], pred[done_mask])
    print(f"  OOF acc (argmax, on {done_mask.sum()} samples) = {acc:.4f}", flush=True)
    np.savez(cache, probs=oof_probs, y=y_train, cids=cids_train, acc=acc, done_mask=done_mask)
    if ckpt_file.exists(): ckpt_file.unlink()
    return oof_probs, y_train


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["deepseek_v3"])
    ap.add_argument("--budget", type=float, default=5.0)
    args = ap.parse_args()
    budget.max_budget = args.budget
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key: exit("Set OPENROUTER_API_KEY")
    print(f"OOF age 4-class, budget ${args.budget}", flush=True)
    for mk in args.models:
        if not budget.check(): break
        try: get_oof_age(mk, api_key)
        except Exception as e:
            print(f"\n  FAILED {mk}: {e}", flush=True)
            import traceback; traceback.print_exc()

