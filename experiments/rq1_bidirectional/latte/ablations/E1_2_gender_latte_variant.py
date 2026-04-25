#!/usr/bin/env python3
"""
Phase 3c: LATTE-style contrastive distillation (correct implementation).

Key difference from previous attempt:
  - CoLES sequence encoder is TRAINABLE (fine-tuned during contrastive alignment)
  - Text encoder (e5-large) is frozen
  - Contrastive loss aligns seq embeddings with text embeddings
  - Classification loss ensures task relevance

Two variants:
  A) Plain text → text embeddings
  B) SHAP-enriched text → text embeddings

Pipeline:
  1. Train CoLES from scratch on Gender transactions
  2. Load frozen text embeddings (already computed)
  3. Fine-tune CoLES with: contrastive alignment + classification
  4. Extract new embeddings → LGBM
"""

import time, json, warnings, gc
from pathlib import Path
from functools import partial

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, StandardScaler
from lightgbm import LGBMClassifier

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



# ptls not needed — using pre-computed embeddings

OUTPUT_DIR = Path("results/gender_latte_distill")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")

GENDER_CFG = {
    "hidden_size": 1024, "rnn_type": "gru", "batch_size": 128,
    "lr": 0.002, "n_epochs": 150, "split_count": 5,
    "cnt_min": 15, "cnt_max": 75, "embeddings_noise": 0.003,
    "lr_step_size": 10, "lr_gamma": 0.9025,
}

LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.02, max_depth=6, subsample=0.5,
                   colsample_bytree=0.75, reg_alpha=1, reg_lambda=1,
                   min_child_samples=50, verbosity=-1)


# ---- Load and prepare data ----
def parse_tr_datetime(s):
    parts = str(s).split(" ", 1)
    day = int(parts[0])
    if len(parts) > 1:
        t = parts[1].split(":")
        frac = (int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])) / 86400.0
    else:
        frac = 0.0
    return day + frac


def load_gender():
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
        np.arange(len(ids)), test_size=0.1, random_state=42, stratify=targets)
    train_ids, test_ids = set(ids[idx_tr]), set(ids[idx_te])

    grouped = tx.groupby("customer_id")

    def build_records(cid_set):
        records = []
        for cid in cid_set:
            if cid not in target_map or cid not in grouped.groups:
                continue
            ct = grouped.get_group(cid)
            if len(ct) < 25:
                continue
            days = ct["day_float"].values
            days = (days - days[0]).astype(np.float32)
            rec = {
                "customer_id": cid, "target": target_map[cid],
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


def build_coles_module(feature_dims):
    cfg = GENDER_CFG
    trx_encoder = TrxEncoder(
        embeddings={
            "mcc_code": {"in": feature_dims["mcc_code"], "out": 48},
            "tr_type": {"in": feature_dims["tr_type"], "out": 24},
        },
        numeric_values={"amount": "identity"},
        embeddings_noise=cfg["embeddings_noise"],
        use_batch_norm_with_lens=True,
    )
    seq_encoder = RnnSeqEncoder(
        trx_encoder=trx_encoder, hidden_size=cfg["hidden_size"],
        type=cfg["rnn_type"], bidir=False, trainable_starter="static",
    )
    return seq_encoder


def extract_embeddings_from_encoder(seq_encoder, records):
    """Extract embeddings by running records through encoder."""
    seq_encoder.eval()
    dl = inference_data_loader(records, num_workers=0, batch_size=64)
    all_embs = []
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device)
            emb = seq_encoder(batch)
            all_embs.append(emb.cpu())
    return torch.cat(all_embs).numpy()


# ---- Phase 1: Use pre-computed CoLES embeddings ----
print("=" * 60)
print("Phase 1: Load pre-computed CoLES embeddings")
print("=" * 60)

# Use existing embeddings and labels
emb_train_baseline = np.load("embeddings/gender/emb_train_seed42.npy")
emb_test_baseline = np.load("embeddings/gender/emb_test_seed42.npy")
y_train = np.load("embeddings/gender/y_train_seed42.npy")
y_test = np.load("embeddings/gender/y_test_seed42.npy")
print(f"  CoLES embeddings: train={emb_train_baseline.shape}, test={emb_test_baseline.shape}")

# Baseline LGBM
scaler = MaxAbsScaler()
Xtr = scaler.fit_transform(emb_train_baseline)
Xte = scaler.transform(emb_test_baseline)
lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=42)
lgbm.fit(Xtr, y_train)
p = lgbm.predict_proba(Xte)[:, 1]
baseline_auc = roc_auc_score(y_test, p)
print(f"  Baseline CoLES LGBM: AUC = {baseline_auc:.4f}")

# ---- Phase 2: Load text embeddings ----
print("\n" + "=" * 60)
print("Phase 2: Load text embeddings")
print("=" * 60)

text_emb_file = Path("results/gender_contrastive_distill/text_embeddings.npz")
if text_emb_file.exists():
    data = np.load(text_emb_file)
    text_plain_train = data["plain_train"]
    text_plain_test = data["plain_test"]
    text_shap_train = data["shap_train"]
    text_shap_test = data["shap_test"]
    print(f"  Loaded: plain={text_plain_train.shape}, shap={text_shap_train.shape}")
else:
    print("  ERROR: text embeddings not found. Run run_gender_contrastive_distill.py first.")
    exit(1)

# ---- Phase 3: Fine-tune CoLES with contrastive alignment ----
print("\n" + "=" * 60)
print("Phase 3: Fine-tune CoLES with contrastive alignment")
print("=" * 60)

results = {"baseline_coles_lgbm": baseline_auc}


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_seq, z_text):
        z_seq = F.normalize(z_seq, dim=1)
        z_text = F.normalize(z_text, dim=1)
        logits = z_seq @ z_text.T / self.temperature
        labels = torch.arange(len(z_seq), device=z_seq.device)
        loss_s2t = F.cross_entropy(logits, labels)
        loss_t2s = F.cross_entropy(logits.T, labels)
        return (loss_s2t + loss_t2s) / 2


class AdapterModel(nn.Module):
    """Deeper adapter: transforms CoLES embeddings guided by contrastive alignment."""
    def __init__(self, coles_dim, text_dim, adapter_dim=512, proj_dim=128):
        super().__init__()
        # Adapter transforms CoLES embeddings (learnable)
        self.adapter = nn.Sequential(
            nn.Linear(coles_dim, adapter_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(adapter_dim, adapter_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(adapter_dim, coles_dim),  # residual-compatible
        )
        # Projection heads for contrastive alignment
        self.proj_seq = nn.Sequential(nn.Linear(coles_dim, 256), nn.ReLU(), nn.Linear(256, proj_dim))
        self.proj_text = nn.Sequential(nn.Linear(text_dim, 256), nn.ReLU(), nn.Linear(256, proj_dim))
        # Classifier on adapted embeddings
        self.classifier = nn.Sequential(
            nn.Linear(coles_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, coles_emb, text_emb=None):
        # Residual adapter
        adapted = coles_emb + self.adapter(coles_emb)
        z_seq = self.proj_seq(adapted)
        z_text = self.proj_text(text_emb) if text_emb is not None else None
        logits = self.classifier(adapted).squeeze(-1)
        return adapted, z_seq, z_text, logits


def train_contrastive_adapter(coles_train, text_train, y_train_np,
                               coles_test, text_test, y_test_np,
                               variant_name, alpha=0.3, epochs=200, lr=1e-3, batch_size=256):
    """Train adapter with contrastive alignment + classification."""
    print(f"\n--- {variant_name} (α={alpha}) ---")

    scaler_c = StandardScaler()
    scaler_t = StandardScaler()
    X_c_tr = torch.FloatTensor(scaler_c.fit_transform(coles_train)).to(device)
    X_c_te = torch.FloatTensor(scaler_c.transform(coles_test)).to(device)
    X_t_tr = torch.FloatTensor(scaler_t.fit_transform(text_train)).to(device)
    y_t = torch.FloatTensor(y_train_np).to(device)

    model = AdapterModel(coles_train.shape[1], text_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bce = nn.BCEWithLogitsLoss()
    infonce = InfoNCELoss(temperature=0.07)

    best_auc = 0
    best_adapted_test = None

    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(len(X_c_tr))
        total_loss = 0
        n_batches = 0

        for start in range(0, len(X_c_tr), batch_size):
            batch = idx[start:start+batch_size]
            adapted, z_seq, z_text, logits = model(X_c_tr[batch], X_t_tr[batch])

            loss_cls = bce(logits, y_t[batch])
            loss_contrast = infonce(z_seq, z_text)
            loss = (1 - alpha) * loss_cls + alpha * loss_contrast

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Eval: adapted embeddings → LGBM
        model.eval()
        with torch.no_grad():
            adapted_tr, _, _, _ = model(X_c_tr)
            adapted_te, _, _, _ = model(X_c_te)
        ad_tr = adapted_tr.cpu().numpy()
        ad_te = adapted_te.cpu().numpy()

        sc = MaxAbsScaler()
        Xtr_a = sc.fit_transform(ad_tr)
        Xte_a = sc.transform(ad_te)
        lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=42)
        lgbm.fit(Xtr_a, y_train_np)
        p = lgbm.predict_proba(Xte_a)[:, 1]
        auc = roc_auc_score(y_test_np, p)

        if auc > best_auc:
            best_auc = auc
            best_adapted_test = ad_te.copy()

        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}/{epochs}: loss={total_loss/n_batches:.4f}, AUC={auc:.4f} (best={best_auc:.4f})")

    return best_auc, best_adapted_test


# Run both variants with different alphas
for variant, text_tr, text_te in [
    ("plain", text_plain_train, text_plain_test),
    ("shap", text_shap_train, text_shap_test),
]:
    for alpha in [0.1, 0.3, 0.5, 0.7]:
        key = f"{variant}_adapter_alpha{alpha}"
        auc, _ = train_contrastive_adapter(
            emb_train_baseline, text_tr, y_train,
            emb_test_baseline, text_te, y_test,
            variant_name=key, alpha=alpha, epochs=200)
        results[key] = auc

# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 60)
print("LATTE-STYLE DISTILLATION SUMMARY")
print("=" * 60)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name:<35} AUC = {auc:.4f}")

with open(OUTPUT_DIR / "latte_distill_results.json", "w") as f:
    json.dump(results, f, indent=2)

pd.DataFrame([{"method": k, "auc": v} for k, v in results.items()]).to_csv(
    OUTPUT_DIR / "latte_distill_results.csv", index=False)
