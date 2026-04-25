#!/usr/bin/env python3
"""
Phase A: 5-seed LGBM evaluation of all existing CoLES checkpoints.
Reports mean ± std per checkpoint for the paper table.

Varies only LGBM random_state (42, 123, 456, 789, 1024).
Train/test split remains seed=42 for consistency across checkpoints.
"""

# =============================================================================
# DISABLED — not needed for current submission (2026-04-25).
# To re-enable: delete the raise SystemExit line below.
# =============================================================================
raise SystemExit("run_seeded_eval.py is temporarily disabled")

import json, warnings, gc
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

from ptls.data_load.datasets import inference_data_loader
from ptls.nn import TrxEncoder, RnnSeqEncoder

SEEDS = [42, 123, 456, 789, 1024]
DATA = Path("data")
OUT = Path("results/seeded_eval")
OUT.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_gender():
    tx = pd.read_csv(DATA / "transactions.csv")
    labels = pd.read_csv(DATA / "gender_train.csv")
    tx = tx[tx["customer_id"].isin(labels["customer_id"])].copy()

    def parse_dt(s):
        parts = str(s).split(" ", 1)
        day = int(parts[0])
        if len(parts) > 1:
            t = parts[1].split(":")
            return day + (int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])) / 86400.0
        return float(day)

    tx["day_float"] = tx["tr_datetime"].apply(parse_dt)
    tx = tx.sort_values(["customer_id", "day_float"])
    tx["amount"] = np.sign(tx["amount"]) * np.log1p(np.abs(tx["amount"]))
    target_map = dict(zip(labels["customer_id"], labels["gender"]))
    encs = {}
    for col in ["mcc_code", "tr_type"]:
        tx[col] = tx[col].fillna("UNK").astype(str)
        encs[col] = LabelEncoder().fit(tx[col])
    grouped = tx.groupby("customer_id")
    feature_dims = {col: len(enc.classes_) + 2 for col, enc in encs.items()}

    def build_records(cid_set):
        records = []
        for cid in cid_set:
            if cid not in target_map or cid not in grouped.groups:
                continue
            ct = grouped.get_group(cid)
            if len(ct) < 25:
                continue
            days = ct["day_float"].values
            rec = {"customer_id": cid, "target": target_map[cid],
                   "event_time": torch.FloatTensor((days - days[0]).astype(np.float32)),
                   "amount": torch.FloatTensor(ct["amount"].values)}
            for col, enc in encs.items():
                rec[col] = torch.LongTensor(enc.transform(ct[col].values) + 1)
            records.append(rec)
        return records

    def build_encoder():
        trx = TrxEncoder(
            embeddings={"mcc_code": {"in": feature_dims["mcc_code"], "out": 48},
                        "tr_type": {"in": feature_dims["tr_type"], "out": 24}},
            numeric_values={"amount": "identity"}, embeddings_noise=0.003,
            use_batch_norm_with_lens=True)
        return RnnSeqEncoder(trx_encoder=trx, hidden_size=1024, type="gru",
                             bidir=False, trainable_starter="static")

    ids = labels["customer_id"].values
    targets = np.array([target_map[c] for c in ids])
    idx_tr, idx_te = train_test_split(np.arange(len(ids)), test_size=0.1,
                                       random_state=42, stratify=targets)
    train_rec = build_records(set(ids[idx_tr]))
    test_rec = build_records(set(ids[idx_te]))
    return train_rec, test_rec, build_encoder, "binary"


def build_rosbank():
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

    def build_records(cid_set):
        records = []
        for cid in cid_set:
            if cid not in target_map or cid not in grouped.groups:
                continue
            ct = grouped.get_group(cid)
            if len(ct) < 15:
                continue
            dt_vals = ct["dt"].values
            days = (dt_vals - dt_vals[0]) / np.timedelta64(1, "D")
            rec = {"customer_id": cid, "target": target_map[cid],
                   "event_time": torch.FloatTensor(days.astype(np.float32)),
                   "amount": torch.FloatTensor(ct["amount"].values)}
            for col, enc in encs.items():
                rec[col] = torch.LongTensor(enc.transform(ct[col].values) + 1)
            records.append(rec)
        return records

    def build_encoder():
        embs = {c: {"in": feature_dims[c], "out": EMB_DIMS[c]} for c in feature_dims if c in EMB_DIMS}
        trx = TrxEncoder(embeddings=embs, numeric_values={"amount": "identity"},
                         embeddings_noise=0.0003, use_batch_norm_with_lens=True)
        return RnnSeqEncoder(trx_encoder=trx, hidden_size=1024, type="lstm",
                             bidir=False, trainable_starter="static")

    ids = labels_df["customer_id"].values
    targets = np.array([target_map[c] for c in ids])
    idx_tr, idx_te = train_test_split(np.arange(len(ids)), test_size=0.1,
                                       random_state=42, stratify=targets)
    train_rec = build_records(set(ids[idx_tr]))
    test_rec = build_records(set(ids[idx_te]))
    return train_rec, test_rec, build_encoder, "binary"


def build_age():
    tx = pd.read_csv(DATA / "transactions_train.csv")
    labels = pd.read_csv(DATA / "train_target.csv")
    target_map = dict(zip(labels["client_id"], labels["bins"]))
    tx = tx.sort_values(["client_id", "trans_date"])
    tx["amount_rur"] = np.sign(tx["amount_rur"]) * np.log1p(np.abs(tx["amount_rur"]))
    tx["small_group"] = tx["small_group"].fillna(0).astype(str)
    sg_enc = LabelEncoder().fit(tx["small_group"])
    grouped = tx.groupby("client_id")
    feature_dims = {"small_group": len(sg_enc.classes_) + 2}

    def build_records(cid_set):
        records = []
        for cid in cid_set:
            if cid not in target_map or cid not in grouped.groups:
                continue
            ct = grouped.get_group(cid)
            if len(ct) < 25:
                continue
            days = ct["trans_date"].values.astype(np.float32)
            records.append({"customer_id": cid, "target": target_map[cid],
                            "event_time": torch.FloatTensor(days - days[0]),
                            "amount": torch.FloatTensor(ct["amount_rur"].values),
                            "small_group": torch.LongTensor(sg_enc.transform(ct["small_group"].values) + 1)})
        return records

    def build_encoder():
        trx = TrxEncoder(embeddings={"small_group": {"in": feature_dims["small_group"], "out": 16}},
                         numeric_values={"amount": "identity"}, embeddings_noise=0.003,
                         use_batch_norm_with_lens=True)
        return RnnSeqEncoder(trx_encoder=trx, hidden_size=800, type="gru",
                             bidir=False, trainable_starter="static")

    ids = labels["client_id"].values
    targets = np.array([target_map[c] for c in ids])
    idx_tr, idx_te = train_test_split(np.arange(len(ids)), test_size=0.1,
                                       random_state=42, stratify=targets)
    train_rec = build_records(set(ids[idx_tr]))
    test_rec = build_records(set(ids[idx_te]))
    return train_rec, test_rec, build_encoder, "multi"


BUILDERS = {"gender": build_gender, "rosbank": build_rosbank, "age": build_age}

CHECKPOINTS = {
    "gender": [
        ("CoLES baseline", "results/gender_true_latte/coles_baseline.pt"),
        ("True LATTE α=0.1", "results/gender_true_latte/coles_finetuned_α0.1.pt"),
        ("True LATTE α=0.3", "results/gender_true_latte/coles_finetuned_α0.3.pt"),
        ("TAID Cross-Modal", "results/taid_crossmodal/gender/coles_taid_best.pt"),
        ("True Bidirectional", "results/gender_true_bidirectional/coles_bidir_cls0.7_con0.2_mut0.1.pt"),
    ],
    "rosbank": [
        ("CoLES baseline", "results/rosbank_true_latte/coles_baseline.pt"),
        ("True LATTE α=0.3", "results/rosbank_true_latte/coles_finetuned_α0.3.pt"),
        ("TAID Cross-Modal", "results/taid_crossmodal/rosbank/coles_taid_best.pt"),
        ("True Bidirectional", "results/rosbank_true_bidirectional/coles_bidir_best.pt"),
    ],
    "age": [
        ("CoLES baseline", "results/age_true_latte/coles_baseline.pt"),
        ("True LATTE α=0.05", "results/age_true_latte/coles_finetuned_α0.05.pt"),
        ("True LATTE α=0.1", "results/age_true_latte/coles_finetuned_α0.1.pt"),
        ("TAID Cross-Modal", "results/taid_crossmodal/age/coles_taid_best.pt"),
    ],
}


def extract(encoder, records, batch_size=64):
    encoder.eval()
    dl = inference_data_loader(records, num_workers=0, batch_size=batch_size)
    with torch.no_grad():
        return torch.cat([encoder(b.to(device)).cpu() for b in dl]).numpy()


def eval_5_seeds(emb_tr, y_tr, emb_te, y_te, task):
    sc = MaxAbsScaler()
    x_tr = sc.fit_transform(emb_tr)
    x_te = sc.transform(emb_te)
    scores = []
    for s in SEEDS:
        if task == "binary":
            p = dict(n_estimators=500, learning_rate=0.02, max_depth=6, subsample=0.5,
                     colsample_bytree=0.75, reg_alpha=1, reg_lambda=1,
                     min_child_samples=50, verbosity=-1, random_state=s)
            clf = LGBMClassifier(**p).fit(x_tr, y_tr)
            scores.append(roc_auc_score(y_te, clf.predict_proba(x_te)[:, 1]))
        else:
            p = dict(n_estimators=1000, learning_rate=0.02, objective="multiclass",
                     num_class=4, max_depth=12, num_leaves=50, subsample=0.75,
                     colsample_bytree=0.75, reg_alpha=1, reg_lambda=1,
                     min_child_samples=50, verbosity=-1, random_state=s)
            clf = LGBMClassifier(**p).fit(x_tr, y_tr)
            scores.append(accuracy_score(y_te, clf.predict(x_te)))
    return scores


def run_dataset(name: str):
    print(f"\n{'='*60}\n{name.upper()}\n{'='*60}")
    train_rec, test_rec, build_enc, task = BUILDERS[name]()
    y_tr = np.array([r["target"] for r in train_rec])
    y_te = np.array([r["target"] for r in test_rec])
    print(f"  train={len(train_rec)}, test={len(test_rec)}, task={task}")

    results = []
    for label, ckpt_path in CHECKPOINTS[name]:
        if not Path(ckpt_path).exists():
            print(f"  [skip] {label} — missing {ckpt_path}")
            continue
        enc = build_enc().to(device)
        try:
            enc.load_state_dict(torch.load(ckpt_path, map_location=device))
        except Exception as e:
            print(f"  [skip] {label} — load failed: {e}")
            del enc; torch.cuda.empty_cache(); continue
        emb_tr = extract(enc, train_rec)
        emb_te = extract(enc, test_rec)
        scores = eval_5_seeds(emb_tr, y_tr, emb_te, y_te, task)
        mean, std = float(np.mean(scores)), float(np.std(scores))
        metric = "AUC" if task == "binary" else "Acc"
        print(f"  {label:30s}  {metric} = {mean:.4f} ± {std:.4f}  (seeds: {[f'{s:.4f}' for s in scores]})")
        results.append({
            "label": label, "ckpt": ckpt_path,
            "mean": mean, "std": std, "scores": scores,
        })
        del enc; torch.cuda.empty_cache(); gc.collect()
    return {"dataset": name, "task": task, "metric": "AUC" if task == "binary" else "Acc",
            "n_train": len(train_rec), "n_test": len(test_rec), "results": results}


if __name__ == "__main__":
    import sys
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["rosbank", "gender", "age"]
    all_results = {}
    for d in datasets:
        try:
            all_results[d] = run_dataset(d)
        except Exception as e:
            print(f"FAILED {d}: {e}")
            import traceback; traceback.print_exc()
    with open(OUT / "seeded_eval.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {OUT / 'seeded_eval.json'}")

    # Pretty table
    print("\n" + "=" * 72)
    print("SUMMARY (mean ± std over 5 seeds)")
    print("=" * 72)
    for d, r in all_results.items():
        print(f"\n{d} ({r['metric']}):")
        if not r["results"]:
            continue
        base = r["results"][0]["mean"]
        for row in r["results"]:
            delta = row["mean"] - base
            sign = "+" if delta >= 0 else ""
            print(f"  {row['label']:30s} {row['mean']:.4f} ± {row['std']:.4f}   ({sign}{delta*100:.2f} pp)")
