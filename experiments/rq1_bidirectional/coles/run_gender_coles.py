#!/usr/bin/env python3

# CoLES Gender

import subprocess, sys, time, json, warnings, gc
from pathlib import Path
from functools import partial

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.linear_model import LogisticRegression

from ptls.data_load.datasets import MemoryMapDataset, inference_data_loader
from ptls.frames.coles import CoLESModule, ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.nn import TrxEncoder, RnnSeqEncoder

print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

SEEDS = [42]

GENDER_CFG = {
    "hidden_size": 1024,
    "rnn_type": "gru",
    "batch_size": 128,
    "lr": 0.002,
    "n_epochs": 150,
    "split_count": 5,
    "cnt_min": 15,
    "cnt_max": 75,
    "embeddings_noise": 0.003,
    "lr_step_size": 10,
    "lr_gamma": 0.9025,
}

EMB_DIMS = {"mcc_code": 48, "tr_type": 24}

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

OUTPUT_DIR = Path("results/gender_coles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR = Path("embeddings/gender")
EMB_DIR.mkdir(parents=True, exist_ok=True)


def download(url, path):
    if path.exists():
        return
    print(f"  downloading {path.name}...")
    subprocess.run(["curl", "-sL", url, "-o", str(path)], check=True)


def download_gender():
    base = "https://huggingface.co/datasets/dllllb/transactions-gender/resolve/main"
    gz = DATA_DIR / "transactions.csv.gz"
    csv = DATA_DIR / "transactions.csv"
    if not csv.exists():
        download(f"{base}/transactions.csv.gz?download=true", gz)
        subprocess.run(["gunzip", str(gz)], check=True)
    download(f"{base}/gender_train.csv?download=true", DATA_DIR / "gender_train.csv")


def parse_tr_datetime(s):
    """Parse "day_offset HH:MM:SS" -> float days."""
    parts = str(s).split(" ", 1)
    day = int(parts[0])
    if len(parts) > 1:
        t = parts[1].split(":")
        frac = (int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])) / 86400.0
    else:
        frac = 0.0
    return day + frac


def load_gender(seed):
    download_gender()
    tx = pd.read_csv(DATA_DIR / "transactions.csv")
    labels = pd.read_csv(DATA_DIR / "gender_train.csv")

    tx = tx[tx["customer_id"].isin(labels["customer_id"])].copy()

    tx["day_float"] = tx["tr_datetime"].apply(parse_tr_datetime)
    tx = tx.sort_values(["customer_id", "day_float"])

    tx["amount"] = np.sign(tx["amount"]) * np.log1p(np.abs(tx["amount"]))

    target_map = dict(zip(labels["customer_id"], labels["gender"]))

    encoders = {}
    for col in ["mcc_code", "tr_type"]:
        tx[col] = tx[col].fillna("UNK").astype(str)
        encoders[col] = LabelEncoder().fit(tx[col])

    ids = labels["customer_id"].values
    targets = np.array([target_map[c] for c in ids])
    idx_tr, idx_te = train_test_split(
        np.arange(len(ids)), test_size=0.1, random_state=seed, stratify=targets)

    train_ids, test_ids = set(ids[idx_tr]), set(ids[idx_te])

    def build_records(cid_set):
        records = []
        grouped = tx.groupby("customer_id")
        for cid in cid_set:
            if cid not in target_map or cid not in grouped.groups:
                continue
            ct = grouped.get_group(cid)
            if len(ct) < 25:
                continue
            days = ct["day_float"].values
            days = (days - days[0]).astype(np.float32)
            rec = {
                "customer_id": cid,
                "target": target_map[cid],
                "event_time": torch.FloatTensor(days),
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
    cfg = GENDER_CFG
    embeddings = {col: {"in": feature_dims[col], "out": EMB_DIMS[col]} for col in feature_dims}
    trx_encoder = TrxEncoder(
        embeddings=embeddings,
        numeric_values={"amount": "identity"},
        embeddings_noise=cfg["embeddings_noise"],
        use_batch_norm_with_lens=True,
    )
    seq_encoder = RnnSeqEncoder(
        trx_encoder=trx_encoder, hidden_size=cfg["hidden_size"],
        type=cfg["rnn_type"], bidir=False, trainable_starter="static",
    )
    module = CoLESModule(
        seq_encoder=seq_encoder,
        optimizer_partial=partial(torch.optim.Adam, lr=cfg["lr"]),
        lr_scheduler_partial=partial(
            torch.optim.lr_scheduler.StepLR,
            step_size=cfg["lr_step_size"], gamma=cfg["lr_gamma"]),
    )
    splitter = SampleSlices(
        split_count=cfg["split_count"],
        cnt_min=cfg["cnt_min"], cnt_max=cfg["cnt_max"],
    )
    return module, splitter


def train_coles(module, records, splitter):
    dataset = ColesDataset(MemoryMapDataset(records), splitter=splitter)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=GENDER_CFG["batch_size"],
        shuffle=True, num_workers=0, collate_fn=dataset.collate_fn,
    )
    trainer = pl.Trainer(
        max_epochs=GENDER_CFG["n_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, enable_progress_bar=True,
        enable_checkpointing=False, logger=False,
    )
    trainer.fit(module, loader)
    module._trainer = trainer
    return module


def extract_embeddings(module, records):
    dl = inference_data_loader(records, num_workers=0, batch_size=64)
    chunks = module._trainer.predict(module, dl)
    return torch.vstack(chunks).cpu().numpy()


def evaluate_downstream(emb_train, y_train, emb_test, y_test, seed):
    scaler = MaxAbsScaler()
    Xtr = scaler.fit_transform(emb_train)
    Xte = scaler.transform(emb_test)
    results = {}

    from lightgbm import LGBMClassifier
    lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=seed)
    lgbm.fit(Xtr, y_train)
    p = lgbm.predict_proba(Xte)[:, 1]
    results["lgbm"] = roc_auc_score(y_test, p)

    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(Xtr, y_train)
    p = lr.predict_proba(Xte)[:, 1]
    results["logreg"] = roc_auc_score(y_test, p)

    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
        random_state=seed, verbosity=0)
    xgb.fit(Xtr, y_train)
    p = xgb.predict_proba(Xte)[:, 1]
    results["xgboost"] = roc_auc_score(y_test, p)
    return results


print("GENDER CoLES (paper config)")
print(f"  GRU-{GENDER_CFG['hidden_size']}, lr={GENDER_CFG['lr']}, epochs={GENDER_CFG['n_epochs']}")
print(f"  Seeds: {SEEDS}")

all_results = []
t0 = time.time()

for seed in SEEDS:
    print(f"\n--- seed={seed} ---")
    ts = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_rec, test_rec, feature_dims = load_gender(seed)
    y_train = np.array([r["target"] for r in train_rec])
    y_test = np.array([r["target"] for r in test_rec])
    print(f"  train={len(train_rec)}, test={len(test_rec)}, features={feature_dims}")

    module, splitter = build_coles(feature_dims)
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
    for model_name, auc in downstream.items():
        all_results.append({"seed": seed, "model": model_name, "roc_auc": auc})
        print(f"  {model_name:<8} AUC={auc:.4f}")

    del module
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  time: {time.time() - ts:.0f}s")

elapsed = time.time() - t0
df = pd.DataFrame(all_results)
df.to_csv(OUTPUT_DIR / "gender_coles_per_seed.csv", index=False)

print(f"GENDER RESULTS ({len(SEEDS)} seed(s))")

for m in ["lgbm", "logreg", "xgboost"]:
    sub = df[df["model"] == m]
    print(f"  {m:<8} AUC = {sub['roc_auc'].mean():.4f} ± {sub['roc_auc'].std():.4f}")
print(f"\nLiterature CoLES Gender: 0.890")
print(f"Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")

with open(OUTPUT_DIR / "gender_summary.json", "w") as f:
    json.dump({
        "experiment": "CoLES Gender (paper config)",
        "config": GENDER_CFG, "emb_dims": EMB_DIMS,
        "lgbm_config": LGBM_PARAMS, "seeds": SEEDS,
        "time": elapsed, "date": time.strftime("%Y-%m-%d %H:%M"),
    }, f, indent=2)
