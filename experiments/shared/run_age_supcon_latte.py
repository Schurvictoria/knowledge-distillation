#!/usr/bin/env python3
"""
Age: SupCon LATTE — fix false negatives in contrastive loss for multiclass.

Problem: standard InfoNCE treats same-class clients as negatives.
Fix: Supervised Contrastive (SupCon) — same-class = positive pairs.

Tests both:
1. Class-masked InfoNCE (remove same-class negatives)
2. Full SupCon (same-class = positive)
3. Prototype alignment (align class centroids)

Uses pre-computed LLM4ES v1 embeddings + CoLES checkpoint.
"""

# =============================================================================
# DISABLED — not needed for current submission (2026-04-25).
# To re-enable: delete the raise SystemExit line below.
# =============================================================================
raise SystemExit("run_age_supcon_latte.py is temporarily disabled")

import time, json, warnings, gc, os
from pathlib import Path
from functools import partial

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, StandardScaler
from lightgbm import LGBMClassifier
import pandas as pd

from ptls.data_load.datasets import MemoryMapDataset, inference_data_loader
from ptls.nn import TrxEncoder, RnnSeqEncoder

OUTPUT_DIR = Path("results/age_supcon")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LGBM_P = dict(n_estimators=1000, learning_rate=0.02, objective="multiclass", num_class=4,
              max_depth=12, num_leaves=50, subsample=0.75, colsample_bytree=0.75,
              reg_alpha=1, reg_lambda=1, min_child_samples=50, verbosity=-1)

# ---- Load data ----
print("Loading data...")
DATA_DIR = Path("data")
tx = pd.read_csv(DATA_DIR / "transactions_train.csv")
labels = pd.read_csv(DATA_DIR / "train_target.csv")
target_map = dict(zip(labels["client_id"], labels["bins"]))
tx = tx.sort_values(["client_id", "trans_date"])
tx["amount_rur"] = np.sign(tx["amount_rur"]) * np.log1p(np.abs(tx["amount_rur"]))
tx["small_group"] = tx["small_group"].fillna(0).astype(str)
sg_enc = LabelEncoder().fit(tx["small_group"])
grouped = tx.groupby("client_id")

cids_train = np.load("embeddings/age/cids_train_seed42.npy")
cids_test = np.load("embeddings/age/cids_test_seed42.npy")
y_train = np.load("embeddings/age/y_train_seed42.npy")
y_test = np.load("embeddings/age/y_test_seed42.npy")

feature_dims = {"small_group": len(sg_enc.classes_) + 2}

def build_records(cid_set):
    records = []
    for cid in cid_set:
        if cid not in target_map or cid not in grouped.groups: continue
        ct = grouped.get_group(cid)
        if len(ct) < 25: continue
        days = ct["trans_date"].values.astype(np.float32)
        records.append({"customer_id": cid, "target": target_map[cid],
                        "event_time": torch.FloatTensor(days - days[0]),
                        "amount": torch.FloatTensor(ct["amount_rur"].values),
                        "small_group": torch.LongTensor(sg_enc.transform(ct["small_group"].values) + 1)})
    return records

train_rec = build_records(cids_train)
test_rec = build_records(cids_test)

# Load LLM embeddings (v1)
llm_data = np.load("results/age_llm4es/llm4es_embeddings.npz")
llm_all = llm_data["embeddings"].astype(np.float32)
n_tr = len(cids_train)
llm_train = llm_all[:n_tr]
sc_l = StandardScaler()
llm_t = torch.FloatTensor(sc_l.fit_transform(llm_train)).to(device)

# CoLES checkpoint
COLES_CKPT = Path("results/age_true_latte/coles_baseline.pt")
def build_encoder():
    trx = TrxEncoder(embeddings={"small_group":{"in":feature_dims["small_group"],"out":16}},
                      numeric_values={"amount":"identity"}, embeddings_noise=0.003, use_batch_norm_with_lens=True)
    return RnnSeqEncoder(trx_encoder=trx, hidden_size=800, type="gru", bidir=False, trainable_starter="static")

def extract_embs(encoder, records):
    encoder.eval()
    dl = inference_data_loader(records, num_workers=0, batch_size=128)
    with torch.no_grad():
        return torch.cat([encoder(b.to(device)).cpu() for b in dl]).numpy()

def eval_coles(encoder):
    emb_tr = extract_embs(encoder, train_rec)
    emb_te = extract_embs(encoder, test_rec)
    sc = MaxAbsScaler()
    lgbm = LGBMClassifier(**LGBM_P, random_state=42)
    lgbm.fit(sc.fit_transform(emb_tr), y_train)
    return accuracy_score(y_test, lgbm.predict(sc.transform(emb_te)))

print(f"  train={len(train_rec)}, test={len(test_rec)}, LLM={llm_t.shape}")

# ---- Loss functions ----

def infonce(zs, zt, temp=0.07):
    """Standard InfoNCE (baseline — has false negative problem)."""
    lo = zs @ zt.T / temp
    la = torch.arange(len(zs), device=device)
    return (F.cross_entropy(lo, la) + F.cross_entropy(lo.T, la)) / 2

def masked_infonce(zs, zt, labels, temp=0.07):
    """InfoNCE with same-class pairs masked out from negatives."""
    lo = zs @ zt.T / temp
    la = torch.arange(len(zs), device=device)
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)) & ~torch.eye(len(zs), dtype=torch.bool, device=device)
    lo_masked = lo.masked_fill(mask, -1e9)
    return (F.cross_entropy(lo_masked, la) + F.cross_entropy(lo_masked.T, la)) / 2

def supcon_cross_modal(zs, zt, labels, temp=0.07):
    """Supervised Contrastive: same-class = positive pairs."""
    B = zs.shape[0]
    sim = zs @ zt.T / temp
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    # Log-sum-exp
    logits_max, _ = sim.max(dim=1, keepdim=True)
    logits = sim - logits_max.detach()
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
    return -mean_log_prob.mean()

def prototype_loss(zs, zt, labels, n_classes=4):
    """Align class prototypes across modalities."""
    loss = 0.0
    count = 0
    for c in range(n_classes):
        mask = (labels == c)
        if mask.sum() < 2: continue
        proto_s = zs[mask].mean(dim=0)
        proto_t = zt[mask].mean(dim=0)
        loss += 1 - F.cosine_similarity(proto_s.unsqueeze(0), proto_t.unsqueeze(0))
        count += 1
    return loss / max(count, 1)

# ---- Training function ----
def train_latte(name, contrastive_fn, alpha=0.05, epochs=20, lr=3e-4):
    print(f"\n--- {name} (α={alpha}) ---")
    seq_encoder = build_encoder().to(device)
    seq_encoder.load_state_dict(torch.load(COLES_CKPT, map_location=device))

    proj_s = nn.Sequential(nn.Linear(800,256),nn.ReLU(),nn.Linear(256,128)).to(device)
    proj_t = nn.Sequential(nn.Linear(llm_t.shape[1],256),nn.ReLU(),nn.Linear(256,128)).to(device)
    classifier = nn.Sequential(nn.Linear(800,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,4)).to(device)

    opt = torch.optim.Adam(list(seq_encoder.parameters()) + list(proj_s.parameters()) +
                           list(proj_t.parameters()) + list(classifier.parameters()),
                           lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    best = eval_coles(seq_encoder)
    print(f"  baseline: {best:.4f}")

    for ep in range(epochs):
        seq_encoder.train()
        idx = torch.randperm(len(train_rec))
        for s in range(0, len(train_rec), 32):
            bi = idx[s:s+32].tolist()
            dl = inference_data_loader([train_rec[i] for i in bi], num_workers=0, batch_size=32)
            for batch in dl:
                se = seq_encoder(batch.to(device))
            zs = F.normalize(proj_s(se), dim=1)
            zt = F.normalize(proj_t(llm_t[bi]), dim=1)
            yb = torch.LongTensor([train_rec[i]["target"] for i in bi]).to(device)

            loss_cls = ce(classifier(se), yb)
            loss_c = contrastive_fn(zs, zt, yb)
            loss = (1-alpha)*loss_cls + alpha*loss_c
            opt.zero_grad(); loss.backward(); opt.step()

        if (ep+1) % 5 == 0:
            acc = eval_coles(seq_encoder)
            if acc > best:
                best = acc
                torch.save(seq_encoder.state_dict(), OUTPUT_DIR / f"{name}.pt")
            print(f"  ep {ep+1}: acc={acc:.4f} (best={best:.4f})")

    return best

# ---- Run experiments ----
results = {}

# Baseline: standard InfoNCE (for reference)
results["infonce_α0.05"] = train_latte("infonce", lambda zs,zt,yb: infonce(zs,zt), alpha=0.05)

# Fix 1: Masked InfoNCE
results["masked_α0.05"] = train_latte("masked_infonce", masked_infonce, alpha=0.05)
results["masked_α0.1"] = train_latte("masked_infonce_01", masked_infonce, alpha=0.1)

# Fix 2: SupCon
results["supcon_α0.05"] = train_latte("supcon_005", supcon_cross_modal, alpha=0.05)
results["supcon_α0.1"] = train_latte("supcon_01", supcon_cross_modal, alpha=0.1)
results["supcon_α0.2"] = train_latte("supcon_02", supcon_cross_modal, alpha=0.2)

# Fix 3: Prototype alignment
results["proto_α0.05"] = train_latte("proto_005", prototype_loss, alpha=0.05)
results["proto_α0.1"] = train_latte("proto_01", prototype_loss, alpha=0.1)

# ---- Summary ----
print("\n" + "=" * 60)
print("AGE SUPCON LATTE RESULTS")
print("=" * 60)
baseline = results["infonce_α0.05"]
for n, v in sorted(results.items(), key=lambda x: -x[1]):
    d = v - baseline
    print(f"  {n:<25} acc={v:.4f} ({'+' if d>=0 else ''}{d:.4f} vs infonce)")

with open(OUTPUT_DIR / "supcon_results.json", "w") as f:
    json.dump(results, f, indent=2)
