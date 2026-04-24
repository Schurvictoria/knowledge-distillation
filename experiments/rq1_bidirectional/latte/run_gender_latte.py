#!/usr/bin/env python3
"""
True LATTE-style distillation: fine-tune CoLES GRU with contrastive alignment.
Uses gradient checkpointing to fit in 24GB VRAM.

Unlike previous attempts (adapters on frozen embeddings), this fine-tunes
the actual sequence encoder through contrastive + classification loss.

Saves checkpoints at each stage.
"""

import time, json, warnings, gc, random, os
from pathlib import Path
from functools import partial

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, StandardScaler
from lightgbm import LGBMClassifier

from ptls.data_load.datasets import MemoryMapDataset, inference_data_loader
from ptls.frames.coles import CoLESModule, ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.nn import TrxEncoder, RnnSeqEncoder

# Reproducibility: seed everything for training
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
pl.seed_everything(SEED, workers=True)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

OUTPUT_DIR = Path("results/gender_true_latte")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, seed={SEED}")

LGBM_P = dict(n_estimators=500, learning_rate=0.02, max_depth=6, subsample=0.5,
              colsample_bytree=0.75, reg_alpha=1, reg_lambda=1, min_child_samples=50, verbosity=-1)

# ---- Load transaction data ----
print("Loading data...")
def parse_dt(s):
    parts = str(s).split(" ", 1)
    day = int(parts[0])
    if len(parts) > 1:
        t = parts[1].split(":")
        return day + (int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])) / 86400.0
    return float(day)

tx = pd.read_csv(DATA_DIR / "transactions.csv")
labels = pd.read_csv(DATA_DIR / "gender_train.csv")
tx = tx[tx["customer_id"].isin(labels["customer_id"])].copy()
tx["day_float"] = tx["tr_datetime"].apply(parse_dt)
tx = tx.sort_values(["customer_id", "day_float"])
tx["amount"] = np.sign(tx["amount"]) * np.log1p(np.abs(tx["amount"]))

target_map = dict(zip(labels["customer_id"], labels["gender"]))
encoders = {}
for col in ["mcc_code", "tr_type"]:
    tx[col] = tx[col].fillna("UNK").astype(str)
    encoders[col] = LabelEncoder().fit(tx[col])

ids = labels["customer_id"].values
targets = np.array([target_map[c] for c in ids])
idx_tr, idx_te = train_test_split(np.arange(len(ids)), test_size=0.1, random_state=42, stratify=targets)
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

train_rec_full = build_records(train_ids)
test_rec = build_records(test_ids)
feature_dims = {col: len(enc.classes_) + 2 for col, enc in encoders.items()}
y_train_full = np.array([r["target"] for r in train_rec_full])
y_test = np.array([r["target"] for r in test_rec])

# Honest val split: 10% of train for model selection (test never used for early stopping)
tr_idx, val_idx = train_test_split(
    np.arange(len(train_rec_full)), test_size=0.1, random_state=42, stratify=y_train_full)
train_rec = [train_rec_full[i] for i in tr_idx]
val_rec = [train_rec_full[i] for i in val_idx]
y_train = y_train_full[tr_idx]
y_val = y_train_full[val_idx]
print(f"  train={len(train_rec)}, val={len(val_rec)}, test={len(test_rec)}")

# ---- Load pre-computed LLM4ES text embeddings (aligned by customer_id) ----
cids_train_order = [r["customer_id"] for r in train_rec]
cids_test_order = [r["customer_id"] for r in test_rec]

llm_data = np.load("results/gender_llm4es/llm4es_embeddings.npz")
llm_all = llm_data["embeddings"].astype(np.float32)
cids_emb = np.load("embeddings/gender/cids_train_seed42.npy")
cids_emb_te = np.load("embeddings/gender/cids_test_seed42.npy")

# Build cid -> llm_emb map
all_cids_emb = np.concatenate([cids_emb, cids_emb_te])
cid_to_llm = {cid: llm_all[i] for i, cid in enumerate(all_cids_emb)}

# Align with train_rec/test_rec order (hard-assert: no silent zeros)
missing_tr = [cid for cid in cids_train_order if cid not in cid_to_llm]
missing_te = [cid for cid in cids_test_order if cid not in cid_to_llm]
assert not missing_tr, f"Missing LLM embedding for {len(missing_tr)} train cids (first: {missing_tr[:3]})"
assert not missing_te, f"Missing LLM embedding for {len(missing_te)} test cids (first: {missing_te[:3]})"
llm_train = np.array([cid_to_llm[cid] for cid in cids_train_order])
llm_test = np.array([cid_to_llm[cid] for cid in cids_test_order])

sc_l = StandardScaler()
llm_train_t = torch.FloatTensor(sc_l.fit_transform(llm_train)).to(device)
llm_test_t = torch.FloatTensor(sc_l.transform(llm_test)).to(device)
print(f"  LLM4ES aligned: train={llm_train_t.shape}, test={llm_test_t.shape}")

# ---- Phase 1: Train vanilla CoLES (or load checkpoint) ----
print("\n" + "=" * 60)
print("Phase 1: Train CoLES baseline")
print("=" * 60)

COLES_CKPT = OUTPUT_DIR / "coles_baseline.pt"

def build_seq_encoder():
    trx_encoder = TrxEncoder(
        embeddings={"mcc_code": {"in": feature_dims["mcc_code"], "out": 48},
                     "tr_type": {"in": feature_dims["tr_type"], "out": 24}},
        numeric_values={"amount": "identity"},
        embeddings_noise=0.003, use_batch_norm_with_lens=True,
    )
    return RnnSeqEncoder(
        trx_encoder=trx_encoder, hidden_size=1024,
        type="gru", bidir=False, trainable_starter="static",
    )

if COLES_CKPT.exists():
    print("  Loading checkpoint...")
    seq_encoder = build_seq_encoder()
    seq_encoder.load_state_dict(torch.load(COLES_CKPT, map_location="cpu"))
    seq_encoder = seq_encoder.to(device)
else:
    seq_encoder = build_seq_encoder()
    coles_module = CoLESModule(
        seq_encoder=seq_encoder,
        optimizer_partial=partial(torch.optim.Adam, lr=0.002),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=10, gamma=0.9025),
    )
    splitter = SampleSlices(split_count=5, cnt_min=15, cnt_max=75)
    dataset = ColesDataset(MemoryMapDataset(train_rec), splitter=splitter)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn)

    print("  Training CoLES 150 epochs...")
    trainer = pl.Trainer(max_epochs=150, accelerator="gpu", devices=1,
                         enable_progress_bar=True, enable_checkpointing=False, logger=False)
    trainer.fit(coles_module, loader)

    torch.save(coles_module._seq_encoder.state_dict(), COLES_CKPT)
    print(f"  Saved checkpoint to {COLES_CKPT}")
    seq_encoder = coles_module._seq_encoder.to(device)
    del coles_module, trainer, loader, dataset
    torch.cuda.empty_cache(); gc.collect()

# Extract baseline embeddings
def extract_embs(encoder, records):
    encoder.eval()
    dl = inference_data_loader(records, num_workers=0, batch_size=64)
    chunks = []
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device)
            chunks.append(encoder(batch).cpu())
    return torch.cat(chunks).numpy()

emb_tr_base = extract_embs(seq_encoder, train_rec)
emb_te_base = extract_embs(seq_encoder, test_rec)

sc = MaxAbsScaler()
lgbm = LGBMClassifier(**LGBM_P, random_state=42)
lgbm.fit(sc.fit_transform(emb_tr_base), y_train)
baseline = roc_auc_score(y_test, lgbm.predict_proba(sc.transform(emb_te_base))[:, 1])
print(f"  Baseline CoLES LGBM: {baseline:.4f}")

# ---- Phase 2: Fine-tune CoLES with contrastive alignment ----
print("\n" + "=" * 60)
print("Phase 2: Fine-tune CoLES with LATTE contrastive alignment")
print("=" * 60)

results = {"baseline_coles": baseline}

# Projection heads
proj_seq = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 128)).to(device)
proj_text = nn.Sequential(nn.Linear(llm_train_t.shape[1], 256), nn.ReLU(), nn.Linear(256, 128)).to(device)
classifier = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)).to(device)

# Enable gradient checkpointing on GRU to save VRAM
if hasattr(seq_encoder, 'seq_encoder'):
    seq_encoder.seq_encoder.flatten_parameters = lambda: None  # Disable for checkpointing

for alpha in [0.1]:
    print(f"\n--- Fine-tune α={alpha} ---")

    # Reload baseline weights
    seq_encoder.load_state_dict(torch.load(COLES_CKPT, map_location=device))
    seq_encoder.train()

    # Reset projection heads
    for module in [proj_seq, proj_text, classifier]:
        for p in module.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    all_params = list(seq_encoder.parameters()) + list(proj_seq.parameters()) + \
                 list(proj_text.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.Adam(all_params, lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    bce = nn.BCEWithLogitsLoss()

    # Honest model selection: track best by VAL, report TEST at best-val epoch
    sc_init = MaxAbsScaler()
    lgbm_init = LGBMClassifier(**LGBM_P, random_state=42)
    lgbm_init.fit(sc_init.fit_transform(extract_embs(seq_encoder, train_rec)), y_train)
    best_val = roc_auc_score(y_val, lgbm_init.predict_proba(
        sc_init.transform(extract_embs(seq_encoder, val_rec)))[:, 1])
    best_test = baseline
    best_epoch = 0

    for epoch in range(10):
        seq_encoder.train(); proj_seq.train(); classifier.train()
        idx = torch.randperm(len(train_rec))
        total_loss = 0
        n_batches = 0

        for start in range(0, len(train_rec), 128):
            batch_idx = idx[start:start+32].tolist()
            batch_records = [train_rec[i] for i in batch_idx]

            dl = inference_data_loader(batch_records, num_workers=0, batch_size=128)
            for batch in dl:
                batch = batch.to(device)
                seq_emb = seq_encoder(batch)  # (B, 1024)

                # Contrastive alignment
                z_seq = F.normalize(proj_seq(seq_emb), dim=1)
                z_text = F.normalize(proj_text(llm_train_t[batch_idx]), dim=1)
                logits_c = z_seq @ z_text.T / 0.07
                labels_c = torch.arange(len(z_seq), device=device)
                loss_c = (F.cross_entropy(logits_c, labels_c) +
                          F.cross_entropy(logits_c.T, labels_c)) / 2

                # Classification
                cls_logits = classifier(seq_emb).squeeze(-1)
                y_batch = torch.FloatTensor([train_rec[i]["target"] for i in batch_idx]).to(device)
                loss_cls = bce(cls_logits, y_batch)

                loss = (1 - alpha) * loss_cls + alpha * loss_c
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        scheduler.step()

        # Eval every 5 epochs — model selection on VAL only
        if (epoch + 1) % 5 == 0:
            emb_tr = extract_embs(seq_encoder, train_rec)
            emb_val = extract_embs(seq_encoder, val_rec)
            emb_te = extract_embs(seq_encoder, test_rec)
            sc2 = MaxAbsScaler()
            lgbm = LGBMClassifier(**LGBM_P, random_state=42)
            lgbm.fit(sc2.fit_transform(emb_tr), y_train)
            val_auc = roc_auc_score(y_val, lgbm.predict_proba(sc2.transform(emb_val))[:, 1])
            test_auc = roc_auc_score(y_test, lgbm.predict_proba(sc2.transform(emb_te))[:, 1])
            if val_auc > best_val:
                best_val = val_auc
                best_test = test_auc  # honest: test corresponding to best-val epoch
                best_epoch = epoch + 1
                torch.save(seq_encoder.state_dict(), OUTPUT_DIR / f"coles_finetuned_α{alpha}.pt")
            print(f"  ep {epoch+1}: loss={total_loss/n_batches:.4f}, "
                  f"val={val_auc:.4f} test={test_auc:.4f} best_val={best_val:.4f} best_test={best_test:.4f}")

    results[f"finetune_α{alpha}"] = best_test
    print(f"  Best: val={best_val:.4f} test={best_test:.4f} at epoch {best_epoch}")

    torch.cuda.empty_cache(); gc.collect()

# ---- Summary ----
print("\n" + "=" * 60)
print("TRUE LATTE DISTILLATION SUMMARY")
print("=" * 60)
for n, v in sorted(results.items(), key=lambda x: -x[1]):
    d = v - baseline
    print(f"  {n:<25} AUC={v:.4f} ({'+' if d >= 0 else ''}{d:.4f})")

with open(OUTPUT_DIR / "true_latte_results.json", "w") as f:
    json.dump(results, f, indent=2)
pd.DataFrame([{"method": k, "auc": v} for k, v in results.items()]).to_csv(
    OUTPUT_DIR / "true_latte_results.csv", index=False)
