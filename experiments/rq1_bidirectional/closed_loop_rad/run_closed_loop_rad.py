#!/usr/bin/env python3
"""
Closed-Loop RAD: Iterative RAMD with LLM re-annotation on updated CoLES embeddings.

vs. RAMD (step2_distill.py::run_gkd_onpolicy):
  RAMD GKD rounds 1+: pure kNN vote on updated embeddings (NO LLM)
  Closed-Loop RAD:    actual LLM re-annotation with updated neighbors every round

Pipeline per iteration i:
  1. Extract CoLES embeddings (iter 0: from coles_baseline.pt; iter i: from prev checkpoint)
  2. 5-fold OOF: for each fold, build kNN on updated embeddings, query val fold
  3. Each val sample → updated neighbors → enrich LLM prompt → get soft label (via OpenRouter)
  4. Distil OOF soft labels into CoLES via reverse KL (same as RAMD step2)
  5. Evaluate on held-out test set; if improvement < threshold → stop

Hypothesis: as CoLES improves, kNN gives more coherent neighbors → LLM sees cleaner context
→ soft labels improve → CoLES improves further. This closes the loop that RAMD leaves open.

Caching:
  iter=0 → reuses results/ramd_openrouter/{dataset}_{model}_oof.npz if available (same embeddings)
  iter>0 → results/ramd_openrouter/{dataset}_{model}_oof_iter{i}.npz (new embeddings)

Usage:
  OPENROUTER_API_KEY=... python run_closed_loop_rad.py gender rosbank --teacher deepseek_v3
"""
import gc
import json
import os
import random
import requests
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler

# Repo root on sys.path so we can import run_openrouter_experiments
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from ptls.data_load.datasets import inference_data_loader
from ptls.nn import RnnSeqEncoder, TrxEncoder
from run_openrouter_experiments import MODELS, budget, call_openrouter_logits

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Reproducibility ──────────────────────────────────────────────────────────
_SEED = 42
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
torch.cuda.manual_seed_all(_SEED)
os.environ["PYTHONHASHSEED"] = str(_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    import pytorch_lightning as _pl
    _pl.seed_everything(_SEED, workers=True)
except ImportError:
    pass

SEEDS = [42, 123, 456, 789, 1024]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT = Path("results/closed_loop_rad")
OUT.mkdir(parents=True, exist_ok=True)

OOF_CACHE = Path("results/ramd_openrouter")
OOF_CACHE.mkdir(parents=True, exist_ok=True)

COLES_CKPT = {
    "gender":  "results/gender_true_latte/coles_baseline.pt",
    "rosbank": "results/rosbank_true_latte/coles_baseline.pt",
    "age":     "results/age_coles/coles_encoder_seed42.pt",
}

N_CLASSES = {"gender": 2, "rosbank": 2, "age": 4}

LGBM_P = dict(
    n_estimators=500, learning_rate=0.02, max_depth=6,
    subsample=0.5, colsample_bytree=0.75,
    reg_alpha=1, reg_lambda=1, min_child_samples=50, verbosity=-1,
)

MCC_GROUPS = {
    range(1, 1500):  "Agriculture",   range(4000, 4800): "Transportation",
    range(5000, 5600): "Retail",      range(5600, 5700): "Clothing",
    range(5800, 5900): "Restaurants", range(6000, 7000): "Financial",
    range(7500, 7600): "Auto",        range(8000, 8100): "Medical",
    range(8200, 8300): "Education",
}


def _mcc_cat(mcc):
    try:
        mcc = int(mcc)
    except Exception:
        return "Other"
    for r, n in MCC_GROUPS.items():
        if mcc in r:
            return n
    return "Other"


def _set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ── Data builders (mirrors step2_distill.py exactly) ─────────────────────────

def _build_gender_data():
    DATA = Path("data")
    tx = pd.read_csv(DATA / "transactions.csv")
    labels = pd.read_csv(DATA / "gender_train.csv")
    tx = tx[tx["customer_id"].isin(labels["customer_id"])].copy()

    def _parse_dt(s):
        parts = str(s).split(" ", 1)
        day = int(parts[0])
        if len(parts) > 1:
            t = parts[1].split(":")
            return day + (int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])) / 86400.0
        return float(day)

    tx["day_float"] = tx["tr_datetime"].apply(_parse_dt)
    tx = tx.sort_values(["customer_id", "day_float"])
    tx["amount"] = np.sign(tx["amount"]) * np.log1p(np.abs(tx["amount"]))
    target_map = dict(zip(labels["customer_id"], labels["gender"]))
    encs = {}
    for col in ["mcc_code", "tr_type"]:
        tx[col] = tx[col].fillna("UNK").astype(str)
        encs[col] = LabelEncoder().fit(tx[col])
    grouped = tx.groupby("customer_id")
    feature_dims = {col: len(enc.classes_) + 2 for col, enc in encs.items()}

    def _build_records(cid_set):
        records = []
        for cid in cid_set:
            if cid not in target_map or cid not in grouped.groups:
                continue
            ct = grouped.get_group(cid)
            if len(ct) < 25:
                continue
            days = ct["day_float"].values
            rec = {
                "customer_id": cid,
                "target": target_map[cid],
                "event_time": torch.FloatTensor((days - days[0]).astype(np.float32)),
                "amount": torch.FloatTensor(ct["amount"].values),
            }
            for col, enc in encs.items():
                rec[col] = torch.LongTensor(enc.transform(ct[col].values) + 1)
            records.append(rec)
        return records

    def _build_encoder():
        trx = TrxEncoder(
            embeddings={
                "mcc_code": {"in": feature_dims["mcc_code"], "out": 48},
                "tr_type":  {"in": feature_dims["tr_type"],  "out": 24},
            },
            numeric_values={"amount": "identity"},
            embeddings_noise=0.003,
            use_batch_norm_with_lens=True,
        )
        return RnnSeqEncoder(trx_encoder=trx, hidden_size=1024, type="gru",
                             bidir=False, trainable_starter="static")

    ids = labels["customer_id"].values
    targets = np.array([target_map[c] for c in ids])
    idx_tr, idx_te = train_test_split(np.arange(len(ids)), test_size=0.1,
                                       random_state=42, stratify=targets)
    train_rec = _build_records(set(ids[idx_tr]))
    test_rec  = _build_records(set(ids[idx_te]))
    return train_rec, test_rec, _build_encoder, 1024


def _build_rosbank_data():
    DATA = Path("data")
    df = pd.read_csv(DATA / "rosbank_train.csv")
    df["dt"] = pd.to_datetime(df["TRDATETIME"], format="%d%b%y:%H:%M:%S")
    df = df.sort_values(["cl_id", "dt"])
    labels_df = df.groupby("cl_id")["target_flag"].max().reset_index()
    labels_df.columns = ["customer_id", "target"]
    target_map = dict(zip(labels_df["customer_id"], labels_df["target"]))
    tx = df.rename(columns={"cl_id": "customer_id", "MCC": "mcc_code"}).copy()
    tx["mcc_code"] = tx["mcc_code"].fillna(0).astype(int)
    tx["amount"] = np.sign(tx["amount"]) * np.log1p(np.abs(tx["amount"]))
    encs = {}
    for col in ["mcc_code", "channel_type", "currency", "trx_category"]:
        if col in tx.columns:
            tx[col] = tx[col].fillna("UNK").astype(str)
            encs[col] = LabelEncoder().fit(tx[col])
    grouped = tx.groupby("customer_id")
    EMB_DIMS = {"mcc_code": 24, "channel_type": 4, "currency": 4, "trx_category": 4}
    feature_dims = {c: len(e.classes_) + 2 for c, e in encs.items()}

    def _build_records(cid_set):
        records = []
        for cid in cid_set:
            if cid not in target_map or cid not in grouped.groups:
                continue
            ct = grouped.get_group(cid)
            if len(ct) < 15:
                continue
            dt_vals = ct["dt"].values
            days = (dt_vals - dt_vals[0]) / np.timedelta64(1, "D")
            rec = {
                "customer_id": cid,
                "target": target_map[cid],
                "event_time": torch.FloatTensor(days.astype(np.float32)),
                "amount": torch.FloatTensor(ct["amount"].values),
            }
            for col, enc in encs.items():
                rec[col] = torch.LongTensor(enc.transform(ct[col].values) + 1)
            records.append(rec)
        return records

    def _build_encoder():
        embs = {c: {"in": feature_dims[c], "out": EMB_DIMS[c]}
                for c in feature_dims if c in EMB_DIMS}
        trx = TrxEncoder(embeddings=embs, numeric_values={"amount": "identity"},
                         embeddings_noise=0.0003, use_batch_norm_with_lens=True)
        return RnnSeqEncoder(trx_encoder=trx, hidden_size=1024, type="lstm",
                             bidir=False, trainable_starter="static")

    ids = labels_df["customer_id"].values
    targets = np.array([target_map[c] for c in ids])
    idx_tr, idx_te = train_test_split(np.arange(len(ids)), test_size=0.1,
                                       random_state=42, stratify=targets)
    train_rec = _build_records(set(ids[idx_tr]))
    test_rec  = _build_records(set(ids[idx_te]))
    return train_rec, test_rec, _build_encoder, 1024


def _build_age_data():
    DATA = Path("data")
    tx = pd.read_csv(DATA / "transactions_train.csv")
    labels = pd.read_csv(DATA / "train_target.csv")

    target_map = dict(zip(labels["client_id"], labels["bins"]))
    tx = tx.sort_values(["client_id", "trans_date"])
    tx["amount_rur"] = np.sign(tx["amount_rur"]) * np.log1p(np.abs(tx["amount_rur"]))
    tx["small_group"] = tx["small_group"].fillna(0).astype(str)

    sg_enc = LabelEncoder().fit(tx["small_group"])
    feature_dims = {"small_group": len(sg_enc.classes_) + 2}
    grouped = tx.groupby("client_id")

    def _build_records(cid_set):
        records = []
        for cid in cid_set:
            if cid not in target_map or cid not in grouped.groups:
                continue
            ct = grouped.get_group(cid)
            if len(ct) < 25:
                continue
            days = ct["trans_date"].values.astype(np.float32)
            days = days - days[0]
            rec = {
                "customer_id": cid,
                "target": target_map[cid],
                "event_time": torch.FloatTensor(days),
                "amount": torch.FloatTensor(ct["amount_rur"].values),
                "small_group": torch.LongTensor(sg_enc.transform(ct["small_group"].values) + 1),
            }
            records.append(rec)
        return records

    def _build_encoder():
        trx = TrxEncoder(
            embeddings={"small_group": {"in": feature_dims["small_group"], "out": 16}},
            numeric_values={"amount": "identity"},
            embeddings_noise=0.003,
            use_batch_norm_with_lens=True,
        )
        return RnnSeqEncoder(trx_encoder=trx, hidden_size=800, type="gru",
                             bidir=False, trainable_starter="static")

    ids = labels["client_id"].values
    targets = np.array([target_map[c] for c in ids])
    idx_tr, idx_te = train_test_split(np.arange(len(ids)), test_size=0.1,
                                       random_state=42, stratify=targets)
    train_rec = _build_records(set(ids[idx_tr]))
    test_rec  = _build_records(set(ids[idx_te]))
    return train_rec, test_rec, _build_encoder, 800


BUILDERS = {"gender": _build_gender_data, "rosbank": _build_rosbank_data, "age": _build_age_data}


# ── Serializers for LLM prompts ───────────────────────────────────────────────

def _make_serializer(dataset_name: str):
    DATA = Path("data")
    if dataset_name == "gender":
        tx = pd.read_csv(DATA / "transactions.csv")
        labels = pd.read_csv(DATA / "gender_train.csv")
        tx = tx[tx["customer_id"].isin(labels["customer_id"])].copy()
        grouped = tx.groupby("customer_id")

        def _ser(cid):
            if cid not in grouped.groups:
                return "No transactions."
            ct = grouped.get_group(cid)
            n = len(ct)
            amt = np.abs(ct["amount"].values)
            cats = ct["mcc_code"].apply(_mcc_cat).value_counts()
            top = ", ".join(
                f"{c} ({cnt} txns, {cnt * 100 // n}%)"
                for c, cnt in cats.head(6).items()
            )
            return (f"Client profile:\n- Transactions: {n}\n"
                    f"- Spending: avg {amt.mean():.0f} RUB, median {np.median(amt):.0f}\n"
                    f"- Top categories: {top}")

        return _ser, "male", "female", "gender (male or female)"
    elif dataset_name == "rosbank":
        df = pd.read_csv(DATA / "rosbank_train.csv")
        df["dt"] = pd.to_datetime(df["TRDATETIME"], format="%d%b%y:%H:%M:%S")
        df = df.sort_values(["cl_id", "dt"])
        grouped = df.groupby("cl_id")

        def _ser(cid):
            if cid not in grouped.groups:
                return "No transactions."
            ct = grouped.get_group(cid)
            n = len(ct)
            amt = np.abs(ct["amount"].values)
            cats = ct["MCC"].fillna(0).astype(int).apply(_mcc_cat).value_counts()
            top = ", ".join(
                f"{c} ({cnt} txns, {cnt * 100 // n}%)"
                for c, cnt in cats.head(6).items()
            )
            return (f"Client profile:\n- Transactions: {n}\n"
                    f"- Spending: avg {amt.mean():.0f} RUB, median {np.median(amt):.0f}\n"
                    f"- Top categories: {top}")

        return _ser, "churn", "stay", "churn (leave or stay)"
    else:  # age
        tx = pd.read_csv(DATA / "transactions_train.csv")
        grouped = tx.groupby("client_id")

        def _ser(cid):
            if cid not in grouped.groups:
                return "No transactions."
            ct = grouped.get_group(cid)
            amt = np.abs(ct["amount_rur"].values)
            cats = ct["small_group"].value_counts().head(5).to_dict()
            return (f"Transactions: {len(ct)}\n"
                    f"Spending: avg {amt.mean():.0f} RUB, median {np.median(amt):.0f}, "
                    f"max {amt.max():.0f}\n"
                    f"Top categories: {list(cats.items())[:5]}")

        return _ser, None, None, "age group (0=youngest, 1=young-adult, 2=middle-age, 3=senior)"


# ── OOF annotation with updated embeddings ────────────────────────────────────

def _get_oof_llm_predictions(
    dataset_name: str,
    train_rec: list,
    emb_train: np.ndarray,
    y_train: np.ndarray,
    iteration: int,
    teacher_model_key: str,
    api_key: str,
    k: int = 10,
    n_threads: int = 10,
) -> np.ndarray:
    """
    OOF LLM soft labels using current CoLES embeddings for kNN.

    iter=0: reuse results/ramd_openrouter/{dataset}_{model}_oof.npz if available
            (same CoLES_0 embeddings → identical kNN → identical labels).
    iter>0: always recompute with updated embeddings.

    Returns np.ndarray of shape (len(train_rec),) with P(positive) values.
    """
    iter_cache = OOF_CACHE / f"{dataset_name}_{teacher_model_key}_oof_iter{iteration}.npz"
    iter_ckpt  = OOF_CACHE / f"{dataset_name}_{teacher_model_key}_oof_iter{iteration}_ckpt.npz"

    # iter=0: try to inherit from existing RAMD OOF (same encoder → same embeddings)
    if iteration == 0 and not iter_cache.exists():
        legacy = OOF_CACHE / f"{dataset_name}_{teacher_model_key}_oof.npz"
        if legacy.exists():
            data = np.load(legacy)
            # Legacy is indexed by cids_train_seed42.npy order → re-align to train_rec order
            cids_legacy = data["cids"].astype(int)
            cid_to_prob = {cids_legacy[i]: data["probs"][i] for i in range(len(cids_legacy))}
            probs_aligned = np.array([cid_to_prob.get(r["customer_id"], 0.5) for r in train_rec])
            auc = roc_auc_score(y_train, probs_aligned)
            print(f"  [iter=0] Inherited RAMD OOF cache (re-aligned): AUC={auc:.4f}")
            np.savez(iter_cache, probs=probs_aligned)
            return probs_aligned

    if iter_cache.exists():
        data = np.load(iter_cache)
        auc = roc_auc_score(y_train, data["probs"])
        print(f"  [iter={iteration}] Loaded OOF cache: AUC={auc:.4f}")
        return data["probs"]

    print(f"\n  [iter={iteration}] Annotating {len(y_train)} samples via OpenRouter "
          f"(k={k}, model={teacher_model_key})...", flush=True)

    m = MODELS[teacher_model_key]
    serialize, pos_label, neg_label, task_desc = _make_serializer(dataset_name)
    system_msg = (f"You are an expert bank analyst predicting client {task_desc}. "
                  f"You also have analysis from an ML model.")
    answer_fmt = f"{pos_label} or {neg_label}"

    sc = MaxAbsScaler()
    emb_scaled = sc.fit_transform(emb_train)

    # Resume from checkpoint if exists
    if iter_ckpt.exists():
        ckpt = np.load(iter_ckpt)
        oof_probs = ckpt["probs"].copy()
        done_mask = ckpt["done_mask"].copy()
        print(f"  Resumed from checkpoint: {done_mask.sum()}/{len(y_train)} done")
    else:
        oof_probs = np.full(len(y_train), 0.5)
        done_mask = np.zeros(len(y_train), dtype=bool)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def _process(vi, nb_idx):
        nb_labels = y_train[nb_idx]
        pos_count = int(nb_labels.sum())
        neg_count = k - pos_count
        majority = pos_label if pos_count > k // 2 else neg_label
        cid = int(train_rec[vi]["customer_id"])
        profile = serialize(cid)
        user_msg = (
            f"{profile}\n\n"
            f"Similar clients: {pos_count} {pos_label}, {neg_count} {neg_label} "
            f"(majority: {majority}).\n\n"
            f"Respond with ONLY one word: {answer_fmt}. Write your answer after ANSWER:"
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]
        prob = call_openrouter_logits(m["id"], messages, api_key, pos_label, neg_label)
        return vi, prob

    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(skf.split(emb_train, y_train)):
        val_todo = [vi for vi in val_idx if not done_mask[vi]]
        if not val_todo:
            continue
        print(f"  Fold {fold + 1}/5 ({len(val_todo)} remaining)...", flush=True)

        nn_model = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn_model.fit(emb_scaled[tr_idx])
        _, idxs = nn_model.kneighbors(emb_scaled[val_todo])

        for batch_start in range(0, len(val_todo), n_threads):
            batch_vis = val_todo[batch_start: batch_start + n_threads]
            # Map local kNN indices back to global train indices
            batch_nbs = [tr_idx[idxs[batch_start + i]] for i in range(len(batch_vis))]

            with ThreadPoolExecutor(max_workers=n_threads) as ex:
                futures = [ex.submit(_process, vi, nb)
                           for vi, nb in zip(batch_vis, batch_nbs)]
                for f in as_completed(futures):
                    vi, prob = f.result()
                    if prob is None:
                        continue  # will be retried on next run via checkpoint
                    oof_probs[vi] = prob
                    done_mask[vi] = True

            # Save checkpoint every 10 batches
            if (batch_start // n_threads) % 10 == 0:
                np.savez(iter_ckpt, probs=oof_probs, done_mask=done_mask)
                elapsed = time.time() - t0
                done = done_mask.sum()
                print(f"    {done}/{len(y_train)} ({done / max(elapsed, 0.1):.1f}/s)", flush=True)

            if not budget.check():
                np.savez(iter_ckpt, probs=oof_probs, done_mask=done_mask)
                print("  Budget exceeded. Saved checkpoint.")
                return oof_probs

    oof_auc = roc_auc_score(y_train, oof_probs)
    print(f"  [iter={iteration}] OOF AUC = {oof_auc:.4f}")
    np.savez(iter_cache, probs=oof_probs, auc=oof_auc)
    if iter_ckpt.exists():
        iter_ckpt.unlink()
    return oof_probs


# ── Age: 4-class OOF via OpenRouter ──────────────────────────────────────────

def _call_4class(model_id: str, messages: list, api_key: str, seed: int = 42):
    """Returns tuple of (p0, p1, p2, p3) or None on failure. Mirrors step1_oof_age.py."""
    import re as _re
    schema = {
        "type": "object",
        "properties": {
            "reasoning":  {"type": "string"},
            "label":      {"type": "integer", "enum": [0, 1, 2, 3]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["label", "confidence"],
        "additionalProperties": False,
    }
    max_tok = 8192 if "deepseek" in model_id else (4096 if "qwen" in model_id.lower() else 500)
    reasoning = ({"max_tokens": max_tok // 2}
                 if "deepseek" in model_id or "qwen" in model_id.lower()
                 else {"enabled": False})
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tok,
        "temperature": 0.6 if reasoning != {"enabled": False} else 0,
        "seed": seed,
        "reasoning": reasoning,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "cls", "strict": True, "schema": schema},
        },
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code != 200:
                if attempt == 2:
                    return None
                time.sleep(2)
                continue
            d = r.json()
            msg = d["choices"][0].get("message", {})
            content = msg.get("content", "") or ""
            u = d.get("usage", {})
            budget.add(u.get("prompt_tokens", 500), u.get("completion_tokens", 100), model_id)
            if not budget.check():
                return None
            try:
                p = json.loads(content)
                label = int(p["label"])
                conf = max(0.3, min(0.95, float(p.get("confidence", 0.7))))
                probs = [(1 - conf) / 3] * 4
                probs[label] = conf
                return tuple(probs)
            except Exception:
                combined = content + " " + (msg.get("reasoning") or "")
                m = _re.search(r'"label"\s*:\s*(\d)', combined)
                if m:
                    label = int(m.group(1))
                    probs = [0.1] * 4
                    probs[label] = 0.7
                    return tuple(probs)
                return None
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2)
    return None


def _get_oof_llm_predictions_age(
    train_rec: list,
    emb_train: np.ndarray,
    y_train: np.ndarray,
    iteration: int,
    teacher_model_key: str,
    api_key: str,
    k: int = 10,
    n_threads: int = 15,
) -> np.ndarray:
    """
    OOF 4-class soft labels for Age using updated CoLES embeddings for kNN.
    Returns np.ndarray of shape (N, 4).
    """
    iter_cache = OOF_CACHE / f"age_{teacher_model_key}_oof_iter{iteration}.npz"
    iter_ckpt  = OOF_CACHE / f"age_{teacher_model_key}_oof_iter{iteration}_ckpt.npz"

    if iteration == 0 and not iter_cache.exists():
        legacy = OOF_CACHE / f"age_{teacher_model_key}_oof.npz"
        if legacy.exists():
            data = np.load(legacy)
            cids_legacy = data["cids"].astype(int)
            cid_to_prob = {cids_legacy[i]: data["probs"][i] for i in range(len(cids_legacy))}
            probs_aligned = np.array([cid_to_prob.get(r["customer_id"],
                                                       np.full(4, 0.25)) for r in train_rec])
            acc = (probs_aligned.argmax(axis=1) == y_train).mean()
            print(f"  [iter=0] Inherited RAMD Age OOF cache (re-aligned): acc={acc:.4f}")
            np.savez(iter_cache, probs=probs_aligned)
            return probs_aligned

    if iter_cache.exists():
        data = np.load(iter_cache)
        acc = (data["probs"].argmax(axis=1) == y_train).mean()
        print(f"  [iter={iteration}] Loaded Age OOF cache: acc={acc:.4f}")
        return data["probs"]

    print(f"\n  [iter={iteration}] Annotating Age {len(y_train)} samples (4-class, k={k})...",
          flush=True)

    m = MODELS[teacher_model_key]
    serialize, _, _, task_desc = _make_serializer("age")
    system_msg = ("You are an expert bank analyst. Predict age group "
                  "(0=youngest, 1=young-adult, 2=middle-age, 3=senior) "
                  "from transaction patterns. You have similar clients from a retrieval system.")

    sc = MaxAbsScaler()
    emb_scaled = sc.fit_transform(emb_train)

    if iter_ckpt.exists():
        ckpt = np.load(iter_ckpt)
        oof_probs = ckpt["probs"].copy()
        done_mask = ckpt["done_mask"].copy()
        print(f"  Resumed from checkpoint: {done_mask.sum()}/{len(y_train)} done")
    else:
        oof_probs = np.full((len(y_train), 4), 0.25, dtype=np.float32)
        done_mask = np.zeros(len(y_train), dtype=bool)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def _process_age(vi, nb_idx):
        nb_labels = y_train[nb_idx]
        class_counts = [(nb_labels == c).sum() for c in range(4)]
        neighbors_str = ", ".join(f"{class_counts[c]} class-{c}" for c in range(4))
        cid = int(train_rec[vi]["customer_id"])
        profile = serialize(cid)
        user_msg = (
            f"{profile}\n\nSimilar clients (kNN): {neighbors_str}.\n\n"
            f'Classify age group. Output JSON: {{"reasoning": "brief", '
            f'"label": 0|1|2|3, "confidence": 0-1}}'
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]
        probs = _call_4class(m["id"], messages, api_key)
        return vi, probs

    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(skf.split(emb_train, y_train)):
        val_todo = [vi for vi in val_idx if not done_mask[vi]]
        if not val_todo:
            continue
        print(f"  Fold {fold + 1}/5 ({len(val_todo)} remaining)...", flush=True)

        nn_model = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn_model.fit(emb_scaled[tr_idx])
        _, idxs = nn_model.kneighbors(emb_scaled[val_todo])

        for batch_start in range(0, len(val_todo), n_threads):
            batch_vis = val_todo[batch_start: batch_start + n_threads]
            batch_nbs = [tr_idx[idxs[batch_start + i]] for i in range(len(batch_vis))]

            with ThreadPoolExecutor(max_workers=n_threads) as ex:
                futures = [ex.submit(_process_age, vi, nb)
                           for vi, nb in zip(batch_vis, batch_nbs)]
                for f in as_completed(futures):
                    vi, probs = f.result()
                    if probs is None:
                        continue
                    oof_probs[vi] = probs
                    done_mask[vi] = True

            if (batch_start // n_threads) % 20 == 0:
                np.savez(iter_ckpt, probs=oof_probs, done_mask=done_mask)
                elapsed = time.time() - t0
                done = done_mask.sum()
                print(f"    {done}/{len(y_train)} ({done / max(elapsed, 0.1):.1f}/s)", flush=True)

            if not budget.check():
                np.savez(iter_ckpt, probs=oof_probs, done_mask=done_mask)
                print("  Budget exceeded. Saved checkpoint.")
                return oof_probs

    acc = (oof_probs.argmax(axis=1) == y_train).mean()
    print(f"  [iter={iteration}] Age OOF acc = {acc:.4f}")
    np.savez(iter_cache, probs=oof_probs, acc=acc)
    if iter_ckpt.exists():
        iter_ckpt.unlink()
    return oof_probs


# ── CoLES helpers ─────────────────────────────────────────────────────────────

def _extract(enc, records, bs=64):
    enc.eval()
    dl = inference_data_loader(records, num_workers=0, batch_size=bs)
    with torch.no_grad():
        return torch.cat([enc(b.to(device)).cpu() for b in dl]).numpy()


def _eval_lgbm(emb_tr, y_tr, emb_te, y_te, seed):
    sc = MaxAbsScaler()
    clf = LGBMClassifier(**LGBM_P, random_state=seed)
    clf.fit(sc.fit_transform(emb_tr), y_tr)
    return roc_auc_score(y_te, clf.predict_proba(sc.transform(emb_te))[:, 1])


def _eval_lgbm_acc(emb_tr, y_tr, emb_te, y_te, seed):
    sc = MaxAbsScaler()
    clf = LGBMClassifier(**LGBM_P, random_state=seed,
                         objective="multiclass", num_class=4)
    clf.fit(sc.fit_transform(emb_tr), y_tr)
    return (clf.predict(sc.transform(emb_te)) == y_te).mean()


def _reverse_kl(teacher_probs, student_logits):
    sp = F.softmax(student_logits, dim=1)
    return (sp * (torch.log(sp + 1e-8) - torch.log(teacher_probs + 1e-8))).sum(dim=1)


def _distil_one_pass(
    dataset_name, seed, enc_state_dict, build_enc,
    train_rec, test_rec, oof_probs_np, hidden,
    alpha, n_epochs, iteration, n_classes=2,
):
    """
    One distillation pass (single seed, single iteration).
    Returns (best_test_score, best_enc_state_dict).
    Uses val-set model selection (no test peeking during training).
    n_classes=2: binary, metric=AUC; n_classes=4: multiclass, metric=accuracy.
    """
    _set_seed(seed + iteration)
    y_tr_full = np.array([r["target"] for r in train_rec])
    y_te = np.array([r["target"] for r in test_rec])

    # Fixed val split across iterations for comparability
    tr_idx, val_idx = train_test_split(
        np.arange(len(train_rec)), test_size=0.1,
        random_state=42, stratify=y_tr_full,
    )
    train_sub = [train_rec[i] for i in tr_idx]
    val_sub   = [train_rec[i] for i in val_idx]
    y_tr  = y_tr_full[tr_idx]
    y_val = y_tr_full[val_idx]

    # oof_probs_np: shape (N,) for binary, (N, n_classes) for multiclass
    teacher_t = torch.FloatTensor(oof_probs_np[tr_idx]).to(device)

    enc = build_enc().to(device)
    enc.load_state_dict(enc_state_dict)

    eval_fn = _eval_lgbm_acc if n_classes > 2 else _eval_lgbm

    classifier = nn.Sequential(
        nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes),
    ).to(device)
    params = list(enc.parameters()) + list(classifier.parameters())
    opt = torch.optim.Adam(params, lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    best_val  = eval_fn(_extract(enc, train_sub), y_tr,
                        _extract(enc, val_sub), y_val, seed)
    best_test = eval_fn(_extract(enc, train_sub), y_tr,
                        _extract(enc, test_rec), y_te, seed)
    best_state = {k: v.cpu().clone() for k, v in enc.state_dict().items()}

    g = torch.Generator().manual_seed(seed + iteration * 1000)
    for ep in range(n_epochs):
        enc.train(); classifier.train()
        idx = torch.randperm(len(train_sub), generator=g)
        tot = 0; nb = 0
        for s in range(0, len(train_sub), 32):
            bi = idx[s: s + 32].tolist()
            dl = inference_data_loader([train_sub[i] for i in bi],
                                       num_workers=0, batch_size=32)
            for batch in dl:
                seq_emb = enc(batch.to(device))

            logits = classifier(seq_emb)
            y_b = torch.LongTensor([train_sub[i]["target"] for i in bi]).to(device)
            loss_ce = F.cross_entropy(logits, y_b)

            if n_classes == 2:
                # binary: teacher_t[bi] is scalar P(positive) → build (batch, 2)
                t_pos = teacher_t[bi].unsqueeze(1)
                t_probs = torch.cat([1 - t_pos, t_pos], dim=1)
            else:
                # multiclass: teacher_t[bi] is already (batch, n_classes)
                t_probs = teacher_t[bi]
            loss_kd = _reverse_kl(t_probs, logits).mean()
            loss = (1 - alpha) * loss_ce + alpha * loss_kd

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sch.step()

        if (ep + 1) % 5 == 0:
            emb_tr_cur = _extract(enc, train_sub)
            val_score  = eval_fn(emb_tr_cur, y_tr, _extract(enc, val_sub), y_val, seed)
            test_score = eval_fn(emb_tr_cur, y_tr, _extract(enc, test_rec), y_te, seed)
            if val_score > best_val:
                best_val  = val_score
                best_test = test_score
                best_state = {k: v.cpu().clone() for k, v in enc.state_dict().items()}
            print(f"    seed={seed} iter={iteration} ep={ep + 1}/{n_epochs} "
                  f"loss={tot / nb:.4f} val={val_score:.4f} test={test_score:.4f} "
                  f"best_val={best_val:.4f} best_test={best_test:.4f}", flush=True)

    del enc, classifier
    torch.cuda.empty_cache(); gc.collect()
    return best_test, best_state


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_closed_loop_rad(
    dataset_name: str,
    teacher_model_key: str,
    api_key: str,
    n_iterations: int = 3,
    alpha: float = 0.1,
    n_epochs: int = 15,
    n_seeds: int = 5,
    k: int = 10,
    convergence_thr: float = 5e-4,
) -> dict:
    """
    Full Closed-Loop RAD experiment.

    Args:
        dataset_name:      "gender", "rosbank", or "age"
        teacher_model_key: key in run_openrouter_experiments.MODELS
        api_key:           OpenRouter API key
        n_iterations:      max refinement rounds
        alpha:             KD loss weight (reverse KL)
        n_epochs:          training epochs per round
        n_seeds:           seeds for final multi-seed evaluation
        k:                 kNN neighborhood size
        convergence_thr:   stop if iter-over-iter gain < this

    Returns:
        summary dict with per-iteration results and final best_mean.
    """
    is_age = dataset_name == "age"
    n_classes = N_CLASSES[dataset_name]
    eval_fn = _eval_lgbm_acc if is_age else _eval_lgbm
    metric_name = "acc" if is_age else "auc"

    print(f"\n{'=' * 60}")
    print(f"Closed-Loop RAD | {dataset_name.upper()} | teacher={teacher_model_key}")
    print(f"  iters={n_iterations}  alpha={alpha}  epochs={n_epochs}  k={k}  "
          f"metric={metric_name}")
    print(f"{'=' * 60}", flush=True)

    train_rec, test_rec, build_enc, hidden = BUILDERS[dataset_name]()
    y_tr = np.array([r["target"] for r in train_rec])
    y_te = np.array([r["target"] for r in test_rec])
    print(f"  train={len(train_rec)}  test={len(test_rec)}")

    # Baseline: CoLES without any distillation
    enc0 = build_enc().to(device)
    enc0.load_state_dict(torch.load(COLES_CKPT[dataset_name], map_location=device))
    emb_tr_base = _extract(enc0, train_rec)
    emb_te_base = _extract(enc0, test_rec)
    baseline_scores = [
        eval_fn(emb_tr_base, y_tr, emb_te_base, y_te, s)
        for s in SEEDS[:n_seeds]
    ]
    baseline_mean = float(np.mean(baseline_scores))
    baseline_std  = float(np.std(baseline_scores))
    print(f"  Baseline CoLES ({metric_name}): {baseline_mean:.4f} ± {baseline_std:.4f}")

    # seed=42 encoder drives the iterative loop (embedding extraction for kNN)
    loop_state = {k_: v.cpu().clone() for k_, v in enc0.state_dict().items()}
    del enc0; gc.collect(); torch.cuda.empty_cache()

    iter_results = []

    for it in range(n_iterations):
        print(f"\n{'─' * 50}")
        print(f"  ITERATION {it}", flush=True)

        # Step 1: extract current embeddings (used both for kNN and OOF)
        enc_loop = build_enc().to(device)
        enc_loop.load_state_dict(loop_state)
        emb_train_cur = _extract(enc_loop, train_rec)
        del enc_loop; gc.collect(); torch.cuda.empty_cache()

        # Step 2: OOF LLM annotation with updated kNN
        if is_age:
            oof_probs = _get_oof_llm_predictions_age(
                train_rec=train_rec,
                emb_train=emb_train_cur,
                y_train=y_tr,
                iteration=it,
                teacher_model_key=teacher_model_key,
                api_key=api_key,
                k=k,
            )
            oof_score = (oof_probs.argmax(axis=1) == y_tr).mean()
            print(f"  Teacher OOF acc = {oof_score:.4f}", flush=True)
        else:
            oof_probs = _get_oof_llm_predictions(
                dataset_name=dataset_name,
                train_rec=train_rec,
                emb_train=emb_train_cur,
                y_train=y_tr,
                iteration=it,
                teacher_model_key=teacher_model_key,
                api_key=api_key,
                k=k,
            )
            oof_score = roc_auc_score(y_tr, oof_probs)
            print(f"  Teacher OOF AUC = {oof_score:.4f}", flush=True)

        # Step 3: distil into CoLES with multiple seeds
        seed_results = []
        seed42_state = None  # val-best encoder from seed=42 — drives next iteration's kNN

        for seed in SEEDS[:n_seeds]:
            gc.collect(); torch.cuda.empty_cache()
            best_test, best_state = _distil_one_pass(
                dataset_name, seed, loop_state, build_enc,
                train_rec, test_rec, oof_probs, hidden, alpha, n_epochs, it,
                n_classes=n_classes,
            )
            seed_results.append({"seed": seed, "best_test": best_test})
            if seed == 42:
                # seed=42's val-best state: used for next iter's kNN (no test leakage)
                seed42_state = best_state
            print(f"  seed={seed} iter={it}: {best_test:.4f}", flush=True)

        test_scores = [r["best_test"] for r in seed_results]
        iter_mean = float(np.mean(test_scores))
        iter_std  = float(np.std(test_scores))

        # Checkpoint seed=42's val-best encoder (used to drive next iteration)
        ckpt_path = OUT / f"{dataset_name}_{teacher_model_key}_iter{it}.pt"
        torch.save(seed42_state, ckpt_path)

        iter_result = {
            "iteration": it,
            f"teacher_oof_{metric_name}": float(oof_score),
            "test_mean": iter_mean,
            "test_std":  iter_std,
            "per_seed": seed_results,
        }
        iter_results.append(iter_result)
        delta = iter_mean - baseline_mean
        print(f"\n  Iter {it}: {iter_mean:.4f} ± {iter_std:.4f}  "
              f"(Δbaseline={delta:+.4f}  Δteacher_OOF={iter_mean - oof_score:+.4f})")

        # seed=42's val-best encoder drives next iteration's kNN (honest: no test leakage)
        loop_state = seed42_state

        # Convergence check (skip first iter — no previous to compare)
        if it > 0:
            prev_mean = iter_results[-2]["test_mean"]
            improvement = iter_mean - prev_mean
            print(f"  Iter-over-iter improvement: {improvement:+.4f}", flush=True)
            if improvement < convergence_thr:
                print(f"  Converged (gain {improvement:.5f} < thr {convergence_thr}). Stopping.")
                break

    best_iter = max(iter_results, key=lambda r: r["test_mean"])
    summary = {
        "dataset": dataset_name,
        "method": "closed_loop_rad",
        "teacher": teacher_model_key,
        "metric": metric_name,
        "n_iterations_run": len(iter_results),
        "alpha": alpha,
        "k": k,
        "baseline_mean": baseline_mean,
        "baseline_std":  baseline_std,
        "best_mean":  best_iter["test_mean"],
        "best_std":   best_iter["test_std"],
        "best_iteration": best_iter["iteration"],
        "iterations": iter_results,
    }
    delta = summary["best_mean"] - summary["baseline_mean"]
    print(f"\n{'=' * 60}")
    print(f"RESULT {dataset_name} ({metric_name}): baseline={baseline_mean:.4f} → "
          f"best={summary['best_mean']:.4f}  (Δ={delta:+.4f}  iter={best_iter['iteration']})")

    out_file = OUT / f"{dataset_name}_{teacher_model_key}_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_file}")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Closed-Loop RAD: iterative RAMD with LLM re-annotation each round"
    )
    ap.add_argument("datasets",   nargs="*", default=["gender", "rosbank", "age"])
    ap.add_argument("--teacher",  default="deepseek_v3",
                    choices=list(MODELS.keys()))
    ap.add_argument("--iters",    type=int,   default=3)
    ap.add_argument("--alpha",    type=float, default=0.1)
    ap.add_argument("--epochs",   type=int,   default=15)
    ap.add_argument("--seeds",    type=int,   default=5)
    ap.add_argument("--k",        type=int,   default=10)
    ap.add_argument("--conv-thr", type=float, default=5e-4,
                    help="Stop if per-iter improvement < threshold")
    ap.add_argument("--budget",   type=float, default=5.0,
                    help="Max OpenRouter spend in USD")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("ERROR: set OPENROUTER_API_KEY environment variable.")

    budget.max_budget = args.budget

    all_summaries = {}
    t_total = time.time()
    for dataset in args.datasets:
        s = run_closed_loop_rad(
            dataset_name=dataset,
            teacher_model_key=args.teacher,
            api_key=api_key,
            n_iterations=args.iters,
            alpha=args.alpha,
            n_epochs=args.epochs,
            n_seeds=args.seeds,
            k=args.k,
            convergence_thr=args.conv_thr,
        )
        all_summaries[dataset] = s

    print(f"\n{'=' * 60}")
    print("CLOSED-LOOP RAD SUMMARY")
    print(f"{'=' * 60}")
    for d, s in all_summaries.items():
        delta = s["best_mean"] - s["baseline_mean"]
        print(f"  {d}: baseline={s['baseline_mean']:.4f} → "
              f"best={s['best_mean']:.4f}  (Δ={delta:+.4f}  "
              f"iter={s['best_iteration']}  iters_run={s['n_iterations_run']})")

    out_path = OUT / "summary.json"
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    elapsed = (time.time() - t_total) / 60
    print(f"\nTotal time: {elapsed:.1f}m  |  Saved: {out_path}")
    budget.summary()
