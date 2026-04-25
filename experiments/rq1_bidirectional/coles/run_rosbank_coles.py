#!/usr/bin/env python3

# CoLES Rosbank

import subprocess, sys, time, json, warnings, gc
from pathlib import Path
from functools import partial

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.linear_model import LogisticRegression

from ptls.data_load.datasets import MemoryMapDataset, inference_data_loader
from ptls.frames.coles import CoLESModule, ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.nn import TrxEncoder, RnnSeqEncoder

# ---- Reproducibility (seed=42) ----
import random as _random, os as _os
_SEED = 42
_random.seed(_SEED); np.random.seed(_SEED)
torch.manual_seed(_SEED); torch.cuda.manual_seed_all(_SEED)
import pytorch_lightning as _pl
_pl.seed_everything(_SEED, workers=True)
_os.environ["PYTHONHASHSEED"] = str(_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---- Required input files ----
from pathlib import Path as _P
_required_inputs = [
    ("data/rosbank_train.csv", "experiments/rq1_bidirectional/coles/run_rosbank_coles.py"),
]
for _p, _hint in _required_inputs:
    assert _P(_p).exists(), f"\n  Missing input: {_p}\n  Run prerequisite: {_hint}"
# ---- end input check ----


print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

SEEDS = [42]

ROSBANK_CFG = {
    "hidden_size": 1024,
    "rnn_type": "lstm",
    "batch_size": 128,
    "lr": 0.004,
    "n_epochs": 60,
    "split_count": 5,
    "cnt_min": 15,
    "cnt_max": 150,
    "embeddings_noise": 0.0003,
    "lr_step_size": 10,
    "lr_gamma": 0.9025,
    "trx_dropout": 0.01,
}

EMB_DIMS = {
    "mcc_code":     24, 
    "channel_type":  4,
    "currency":      4,
    "trx_category":  4,
}

LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.02,
    "boosting_type": "gbdt",
    "max_depth": 6,
    "subsample": 0.5,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "min_child_samples": 50,
    "verbosity": -1,
}

OUTPUT_DIR = Path("results/rosbank_coles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR = Path("embeddings/rosbank")
EMB_DIR.mkdir(parents=True, exist_ok=True)


def download(url, path):
    if path.exists():
        return
    print(f"  downloading {path.name}...")
    subprocess.run(["curl", "-sL", url, "-o", str(path)], check=True)


def download_rosbank():
    base = "https://huggingface.co/datasets/dllllb/rosbank-churn/resolve/main"
    gz = DATA_DIR / "rosbank_train.csv.gz"
    csv = DATA_DIR / "rosbank_train.csv"
    if not csv.exists():
        download(f"{base}/train.csv.gz?download=true", gz)
        subprocess.run(["gunzip", "-c", str(gz)], stdout=open(csv, "w"), check=True)
        gz.unlink(missing_ok=True)


def load_rosbank(seed):
    download_rosbank()
    df = pd.read_csv(DATA_DIR / "rosbank_train.csv")

    labels = df.groupby("cl_id")["target_flag"].max().reset_index()
    labels.columns = ["customer_id", "target"]
    target_map = dict(zip(labels["customer_id"], labels["target"]))

    tx = df.rename(columns={"cl_id": "customer_id", "MCC": "mcc_code"}).copy()
    tx["mcc_code"] = tx["mcc_code"].fillna(0).astype(int)

    tx["dt"] = pd.to_datetime(tx["TRDATETIME"], format="%d%b%y:%H:%M:%S")
    tx = tx.sort_values(["customer_id", "dt"])

    tx["amount"] = np.sign(tx["amount"]) * np.log1p(np.abs(tx["amount"]))

    encoders = {}
    for col in ["mcc_code", "channel_type", "currency", "trx_category"]:
        if col in tx.columns:
            tx[col] = tx[col].fillna("UNK").astype(str)
            enc = LabelEncoder().fit(tx[col])
            encoders[col] = enc
        else:
            print(f"  WARNING: column {col} not found in data")

    ids = labels["customer_id"].values
    targets = np.array([target_map[c] for c in ids])

    idx_tr, idx_te = train_test_split(
        np.arange(len(ids)), test_size=0.1, random_state=seed, stratify=targets)

    train_ids = set(ids[idx_tr])
    test_ids = set(ids[idx_te])

    def build_records(customer_ids_set):
        records = []
        grouped = tx.groupby("customer_id")
        for cid in customer_ids_set:
            if cid not in target_map or cid not in grouped.groups:
                continue
            ct = grouped.get_group(cid)
            if len(ct) < 15:
                continue
            dt = ct["dt"].values
            days = (dt - dt[0]) / np.timedelta64(1, "D")
            rec = {
                "customer_id": cid,
                "target": target_map[cid],
                "event_time": torch.FloatTensor(days.astype(np.float32)),
                "amount": torch.FloatTensor(ct["amount"].values),
            }
            for col, enc in encoders.items():
                rec[col] = torch.LongTensor(enc.transform(ct[col].values) + 1)
            records.append(rec)
        return records

    train_rec = build_records(train_ids)
    test_rec = build_records(test_ids)

    feature_dims = {col: len(enc.classes_) + 2 for col, enc in encoders.items()}
    return train_rec, test_rec, feature_dims


def build_coles(feature_dims):
    cfg = ROSBANK_CFG
    embeddings = {}
    for col, n_classes in feature_dims.items():
        embeddings[col] = {"in": n_classes, "out": EMB_DIMS[col]}

    trx_encoder = TrxEncoder(
        embeddings=embeddings,
        numeric_values={"amount": "identity"},
        embeddings_noise=cfg["embeddings_noise"],
        use_batch_norm_with_lens=True,
    )

    seq_encoder = RnnSeqEncoder(
        trx_encoder=trx_encoder,
        hidden_size=cfg["hidden_size"],
        type=cfg["rnn_type"],
        bidir=False,
        trainable_starter="static",
    )

    module = CoLESModule(
        seq_encoder=seq_encoder,
        optimizer_partial=partial(torch.optim.Adam, lr=cfg["lr"]),
        lr_scheduler_partial=partial(
            torch.optim.lr_scheduler.StepLR,
            step_size=cfg["lr_step_size"],
            gamma=cfg["lr_gamma"]),
    )

    splitter = SampleSlices(
        split_count=cfg["split_count"],
        cnt_min=cfg["cnt_min"],
        cnt_max=cfg["cnt_max"],
    )

    return module, splitter


def train_coles(module, train_records, splitter):
    dataset = ColesDataset(MemoryMapDataset(train_records), splitter=splitter)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=ROSBANK_CFG["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    trainer = pl.Trainer(
        max_epochs=ROSBANK_CFG["n_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(module, loader)
    module._trainer = trainer
    return module


def extract_embeddings(module, records):
    dl = inference_data_loader(records, num_workers=0, batch_size=64)
    chunks = module._trainer.predict(module, dl)
    return torch.vstack(chunks).cpu().numpy()


def evaluate_downstream(X_train, y_train, X_test, y_test, seed):
    scaler = MaxAbsScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    results = {}

    from lightgbm import LGBMClassifier
    lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=seed)
    lgbm.fit(Xtr, y_train)
    p = lgbm.predict_proba(Xte)[:, 1]
    results["lgbm"] = {
        "roc_auc": roc_auc_score(y_test, p),
        "accuracy": accuracy_score(y_test, (p >= 0.5).astype(int)),
        "f1": f1_score(y_test, (p >= 0.5).astype(int)),
    }

    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(Xtr, y_train)
    p = lr.predict_proba(Xte)[:, 1]
    results["logreg"] = {
        "roc_auc": roc_auc_score(y_test, p),
        "accuracy": accuracy_score(y_test, (p >= 0.5).astype(int)),
        "f1": f1_score(y_test, (p >= 0.5).astype(int)),
    }

    from xgboost import XGBClassifier

    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
        random_state=seed, verbosity=0)
    xgb.fit(Xtr, y_train)
    p = xgb.predict_proba(Xte)[:, 1]
    results["xgboost"] = {
        "roc_auc": roc_auc_score(y_test, p),
        "accuracy": accuracy_score(y_test, (p >= 0.5).astype(int)),
        "f1": f1_score(y_test, (p >= 0.5).astype(int)),
    }

    return results


print("ROSBANK CoLES (aligned with coles-paper original)")
print(f"  LSTM-{ROSBANK_CFG['hidden_size']}, lr={ROSBANK_CFG['lr']}, "
      f"epochs={ROSBANK_CFG['n_epochs']}, "
      f"slices=[{ROSBANK_CFG['cnt_min']},{ROSBANK_CFG['cnt_max']}], "
      f"noise={ROSBANK_CFG['embeddings_noise']}")
print(f"  LR scheduler: step={ROSBANK_CFG['lr_step_size']}, gamma={ROSBANK_CFG['lr_gamma']}")
print(f"  amount: log-transformed, emb dims: {EMB_DIMS}")
print(f"  test_size=0.1, LGBM: n_est={LGBM_PARAMS['n_estimators']}, max_depth={LGBM_PARAMS['max_depth']}")
print(f"  Seeds: {SEEDS}")

all_results = []
t0 = time.time()

for seed in SEEDS:
    print(f"\n--- seed={seed} ---")
    ts = time.time()

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_rec, test_rec, feature_dims = load_rosbank(seed)
    y_train = np.array([r["target"] for r in train_rec])
    y_test = np.array([r["target"] for r in test_rec])
    print(f"  train={len(train_rec)}, test={len(test_rec)}, features={feature_dims}")

    module, splitter = build_coles(feature_dims)
    print(f"  training CoLES: {ROSBANK_CFG['n_epochs']} epochs, "
          f"LSTM-{ROSBANK_CFG['hidden_size']}, batch={ROSBANK_CFG['batch_size']}")

    module = train_coles(module, train_rec, splitter)

    torch.cuda.empty_cache()
    emb_train = extract_embeddings(module, train_rec)
    emb_test = extract_embeddings(module, test_rec)
    print(f"  embeddings: {emb_train.shape}")

    # Save embeddings
    np.save(EMB_DIR / f"emb_train_seed{seed}.npy", emb_train)
    np.save(EMB_DIR / f"emb_test_seed{seed}.npy", emb_test)
    np.save(EMB_DIR / f"y_train_seed{seed}.npy", y_train)
    np.save(EMB_DIR / f"y_test_seed{seed}.npy", y_test)
    cids_train = [r["customer_id"] for r in train_rec]
    cids_test = [r["customer_id"] for r in test_rec]
    np.save(EMB_DIR / f"cids_train_seed{seed}.npy", np.array(cids_train))
    np.save(EMB_DIR / f"cids_test_seed{seed}.npy", np.array(cids_test))
    if seed == SEEDS[0]:
        print(f"  saved embeddings to {EMB_DIR}")

    downstream = evaluate_downstream(emb_train, y_train, emb_test, y_test, seed)

    for model_name, metrics in downstream.items():
        all_results.append({
            "dataset": "rosbank",
            "model": model_name,
            "variant": "coles_paper_aligned",
            "seed": seed,
            **metrics,
        })
        print(f"  {model_name:<8} AUC={metrics['roc_auc']:.4f}")

    del module
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  time: {time.time() - ts:.0f}s")

elapsed = time.time() - t0

full_df = pd.DataFrame(all_results)
full_df.to_csv(OUTPUT_DIR / "rosbank_coles_per_seed.csv", index=False)

agg = []
for (m, v), g in full_df.groupby(["model", "variant"]):
    row = {"model": m, "variant": v, "n_seeds": len(g)}
    for metric in ["roc_auc", "accuracy", "f1"]:
        row[f"{metric}_mean"] = g[metric].mean()
        row[f"{metric}_std"] = g[metric].std()
    agg.append(row)
agg_df = pd.DataFrame(agg)
agg_df.to_csv(OUTPUT_DIR / "rosbank_coles_aggregated.csv", index=False)

print(f"ROSBANK RESULTS ({len(SEEDS)} seed(s))")
for _, r in agg_df.iterrows():
    print(f"  {r['model']:<8} AUC = {r['roc_auc_mean']:.4f} +/- {r['roc_auc_std']:.4f}")

print(f"\nLiterature CoLES Rosbank: 0.841")
print(f"Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")

with open(OUTPUT_DIR / "rosbank_summary.json", "w") as f:
    json.dump({
        "experiment": "CoLES Rosbank (coles-paper aligned)",
        "config": ROSBANK_CFG,
        "emb_dims": EMB_DIMS,
        "lgbm_config": LGBM_PARAMS,
        "downstream_models": ["lgbm", "logreg", "xgboost"],
        "seeds": SEEDS,
        "time": elapsed,
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "changes_vs_prev": [
            "added amount with log transform",
            "mcc emb 48->24, other embs 16->4",
            "lr_step_size 30->10",
            "test_size 0.2->0.1",
            "lgbm max_depth 12->6, n_estimators 1000->500, subsample 0.75->0.5",
        ],
    }, f, indent=2)
