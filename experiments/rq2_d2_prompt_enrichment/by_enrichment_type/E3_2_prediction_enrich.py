#!/usr/bin/env python3
"""
RQ2 D2 Prediction enrichment — XGBoost confidence only (без SHAP факторов).

Qwen2.5-7B-Instruct via OpenRouter (consistent with other RQ2 D2 rows).
CoT strategy. Seed=42. Checkpoints каждые 50 клиентов.
"""
import os, json, time, re, requests, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from run_openrouter_experiments import load_dataset, budget, OUT

# ---- Reproducibility (seed=42) ----
import random as _random, os as _os
_SEED = 42
_random.seed(_SEED); np.random.seed(_SEED)
_os.environ["PYTHONHASHSEED"] = str(_SEED)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "qwen/qwen-2.5-7b-instruct"


def call_qwen25(messages, api_key, pos_label, neg_label, seed=42):
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
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0,
        "seed": seed,
        "response_format": {"type": "json_schema",
            "json_schema": {"name": "cls", "strict": True, "schema": schema}},
    }
    for attempt in range(3):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429: time.sleep(5*(attempt+1)); continue
            if resp.status_code == 402:
                print(f"    [402 CREDITS EXHAUSTED] Halting.", flush=True)
                import os; os._exit(2)
            if resp.status_code != 200:
                if attempt == 2: return 0.5
                time.sleep(2); continue
            data = resp.json()
            content = data["choices"][0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            budget.add(usage.get("prompt_tokens", 500),
                      usage.get("completion_tokens", 100), MODEL_ID)
            if not budget.check(): return 0.5
            try:
                parsed = json.loads(content)
                label = parsed["label"].lower()
                conf = max(0.05, min(0.95, float(parsed.get("confidence", 0.85))))
                return conf if label == pos_label.lower() else 1 - conf
            except:
                m = re.search(r'"label"\s*:\s*"(\w+)"', content)
                if m:
                    return 0.85 if m.group(1).lower() == pos_label.lower() else 0.15
                return 0.5
        except:
            if attempt == 2: return 0.5
            time.sleep(2)
    return 0.5


def get_xgb_oof(dataset_name, data, grouped, y_train):
    """Generate OOF XGBoost predictions on train for train, and standard test predictions."""
    cache = OUT / f"xgb_oof_{dataset_name}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["oof"], d["test"]

    cids_train = np.load(f"embeddings/{dataset_name}/cids_train_seed42.npy")
    cids_test = np.load(f"embeddings/{dataset_name}/cids_test_seed42.npy")

    def agg_feat(cids):
        recs = []
        for cid in cids:
            if cid not in grouped.groups:
                recs.append([0, 0, 0, 0, 0]); continue
            ct = grouped.get_group(cid)
            amt_col = "amount" if "amount" in ct.columns else ("amount_rur" if "amount_rur" in ct.columns else ct.columns[-1])
            a = np.abs(pd.to_numeric(ct[amt_col], errors="coerce").fillna(0).values)
            mcc_col = "mcc_code" if "mcc_code" in ct.columns else ("small_group" if "small_group" in ct.columns else ("MCC" if "MCC" in ct.columns else ct.columns[2]))
            recs.append([len(ct), float(a.mean()), float(a.std()), float(np.median(a)), int(ct[mcc_col].nunique())])
        return np.array(recs, dtype=np.float32)

    X_tr = agg_feat(cids_train)
    X_te = agg_feat(cids_test)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y_train))
    for tr_idx, va_idx in skf.split(X_tr, y_train):
        m = XGBClassifier(n_estimators=300, max_depth=6, random_state=42, verbosity=0)
        m.fit(X_tr[tr_idx], y_train[tr_idx])
        oof[va_idx] = m.predict_proba(X_tr[va_idx])[:, 1]

    # Final: train on all for test preds
    m = XGBClassifier(n_estimators=300, max_depth=6, random_state=42, verbosity=0)
    m.fit(X_tr, y_train)
    test_probs = m.predict_proba(X_te)[:, 1]

    np.savez(cache, oof=oof, test=test_probs)
    return oof, test_probs


def run_dataset(dataset_name, api_key):
    print(f"\n=== Prediction-enrichment CoT on {dataset_name} (Qwen2.5-7B via OpenRouter) ===", flush=True)

    data = load_dataset(dataset_name)
    cids_test = data["cids_test"]
    y_test = data["y_test"]

    # Build XGBoost predictions on test (from agg features)
    import pandas as pd

    DATA = Path("data")
    if dataset_name == "gender":
        tx = pd.read_csv(DATA / "transactions.csv")
        labels = pd.read_csv(DATA / "gender_train.csv")
        tx = tx[tx["customer_id"].isin(labels["customer_id"])].copy()
        grouped = tx.groupby("customer_id")
    elif dataset_name == "rosbank":
        df = pd.read_csv(DATA / "rosbank_train.csv")
        grouped = df.groupby("cl_id")
    else:  # age
        tx = pd.read_csv(DATA / "transactions_train.csv")
        grouped = tx.groupby("client_id")
    y_train = np.load(f"embeddings/{dataset_name}/y_train_seed42.npy")
    _, xgb_test_probs = get_xgb_oof(dataset_name, data, grouped, y_train)

    cache = OUT / f"{dataset_name}_qwen25_7b_prediction_cot.json"
    if cache.exists():
        c = json.load(open(cache))
        print(f"  Cached: {c.get('auc', c.get('accuracy', 0)):.4f}", flush=True)
        return

    pos_label = data["pos_label"]; neg_label = data["neg_label"]

    def process_one(i):
        cid = int(cids_test[i])
        profile = data["serialize"](cid)
        xgb_p = float(xgb_test_probs[i])
        xgb_pred_label = pos_label if xgb_p >= 0.5 else neg_label
        xgb_conf = xgb_p if xgb_p >= 0.5 else 1 - xgb_p
        enrich = f"ML model: predicts {xgb_pred_label} ({xgb_conf*100:.0f}% confidence)."

        system = (data["system_expert"] + " You also have a prediction from an ML model. Think step by step.")
        user = (f"{profile}\n\n{enrich}\n\nClassify. Output JSON: "
                f'{{"reasoning": "analysis", "label": "{pos_label}" or "{neg_label}", "confidence": 0-1}}')
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        return i, call_qwen25(messages, api_key, pos_label, neg_label, seed=42)

    ckpt = OUT / f"{dataset_name}_qwen25_7b_prediction_cot_ckpt.npz"
    if ckpt.exists():
        preds = list(np.load(ckpt)["preds"])
        start = len(preds)
        print(f"  Resuming from {start}", flush=True)
    else:
        preds = []; start = 0

    if start == 0:
        print("  Sanity check...", flush=True)
        for i in range(2):
            _, p = process_one(i)
            print(f"    Client {i}: true={y_test[i]}, xgb={xgb_test_probs[i]:.3f}, prob={p:.3f}", flush=True)

    BATCH = 5
    t0 = time.time()
    for bs in range(start, len(cids_test), BATCH):
        be = min(bs + BATCH, len(cids_test))
        with ThreadPoolExecutor(max_workers=BATCH) as ex:
            futs = [ex.submit(process_one, i) for i in range(bs, be)]
            br = {}
            for f in as_completed(futs):
                i, p = f.result(); br[i] = p
        for i in range(bs, be):
            preds.append(br[i])

        if len(preds) % 50 < BATCH:
            np.savez(ckpt, preds=np.array(preds))
            auc = roc_auc_score(y_test[:len(preds)], preds)
            rate = (len(preds)-start)/max(time.time()-t0, 0.1)
            print(f"    {len(preds)}/{len(cids_test)} ({rate:.1f}/s, AUC={auc:.4f})", flush=True)

        if not budget.check():
            np.savez(ckpt, preds=np.array(preds))
            return

    preds_arr = np.array(preds)
    auc = roc_auc_score(y_test, preds_arr)
    print(f"\n  {dataset_name} Prediction-CoT: AUC={auc:.4f}", flush=True)
    with open(cache, "w") as f:
        json.dump({"auc": auc, "method": "prediction_cot", "model": "qwen2.5-7b",
                   "dataset": dataset_name, "n_test": len(preds_arr), "seed": 42}, f, indent=2)
    np.savez(OUT / f"{dataset_name}_qwen25_7b_prediction_cot_preds.npz",
             preds=preds_arr, cids=cids_test, y_test=y_test, xgb=xgb_test_probs)
    if ckpt.exists(): ckpt.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["gender", "rosbank"])
    ap.add_argument("--budget", type=float, default=2.0)
    args = ap.parse_args()
    budget.max_budget = args.budget
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key: exit("Set OPENROUTER_API_KEY")
    for ds in args.datasets:
        if not budget.check(): break
        run_dataset(ds, api_key)
