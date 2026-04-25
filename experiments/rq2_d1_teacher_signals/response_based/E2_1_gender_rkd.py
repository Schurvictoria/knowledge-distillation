#!/usr/bin/env python3
"""
RKD (Relational Knowledge Distillation) for Gender dataset.
Transfers inter-sample relational structure from LLM embeddings to CoLES adapter.
Does NOT force point-wise alignment — preserves CoLES-specific information.

Also implements CRD (memory bank) and LATTE with orthogonal decomposition for comparison.

All on pre-computed frozen embeddings. Saves checkpoints.
"""

import time, json, warnings, gc
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
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

# ---- Required input files ----
from pathlib import Path as _P
_required_inputs = [
    ("embeddings/gender/emb_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
    ("embeddings/gender/emb_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
    ("embeddings/gender/y_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
    ("embeddings/gender/y_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
    ("results/gender_llm4es/llm4es_embeddings.npz", "experiments/rq2_d1_teacher_signals/feature_based/E2_2_gender_llm4es.py"),
]
for _p, _hint in _required_inputs:
    assert _P(_p).exists(), f"\n  Missing input: {_p}\n  Run prerequisite: {_hint}"
# ---- end input check ----



OUTPUT_DIR = Path("results/gender_rkd")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__}, device={device}")

LGBM_P = dict(n_estimators=500, learning_rate=0.02, max_depth=6, subsample=0.5,
              colsample_bytree=0.75, reg_alpha=1, reg_lambda=1, min_child_samples=50, verbosity=-1)

# ---- Load data ----
print("Loading embeddings...")
C_tr_raw = np.load("embeddings/gender/emb_train_seed42.npy")
C_te_raw = np.load("embeddings/gender/emb_test_seed42.npy")
y_train = np.load("embeddings/gender/y_train_seed42.npy")
y_test = np.load("embeddings/gender/y_test_seed42.npy")

llm_data = np.load("results/gender_llm4es/llm4es_embeddings.npz")
L_all = llm_data["embeddings"].astype(np.float32)
n_tr = len(C_tr_raw)
L_tr_raw, L_te_raw = L_all[:n_tr], L_all[n_tr:]

sc_c, sc_l = StandardScaler(), StandardScaler()
C_tr = torch.FloatTensor(sc_c.fit_transform(C_tr_raw)).to(device)
C_te = torch.FloatTensor(sc_c.transform(C_te_raw)).to(device)
L_tr = torch.FloatTensor(sc_l.fit_transform(L_tr_raw)).to(device)
L_te = torch.FloatTensor(sc_l.transform(L_te_raw)).to(device)
Y_tr = torch.FloatTensor(y_train).to(device)
print(f"  CoLES: {C_tr.shape}, LLM4ES: {L_tr.shape}")

# Baseline
sc = MaxAbsScaler()
lgbm = LGBMClassifier(**LGBM_P, random_state=42)
lgbm.fit(sc.fit_transform(C_tr_raw), y_train)
baseline = roc_auc_score(y_test, lgbm.predict_proba(sc.transform(C_te_raw))[:, 1])
print(f"  Baseline CoLES LGBM: {baseline:.4f}")


# ---- Adapter model ----
class Adapter(nn.Module):
    def __init__(self, dim, proj=128):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, dim))
        self.head = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))
        self.proj = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, proj))

    def forward(self, x):
        adapted = x + self.adapter(x)
        return adapted, self.head(adapted).squeeze(-1), F.normalize(self.proj(adapted), dim=1)


# ---- Loss functions ----
def rkd_distance(s_emb, t_emb):
    """RKD-D: match normalized pairwise distances."""
    with torch.no_grad():
        t_d = torch.cdist(t_emb, t_emb, p=2)
        t_d = t_d / (t_d.mean() + 1e-8)
    s_d = torch.cdist(s_emb, s_emb, p=2)
    s_d = s_d / (s_d.mean() + 1e-8)
    return F.smooth_l1_loss(s_d, t_d)


def rkd_angle(s_emb, t_emb):
    """RKD-A: match angles in triplets (sampled)."""
    n = len(s_emb)
    if n < 3:
        return torch.tensor(0.0, device=s_emb.device)
    # Sample triplets
    n_triplets = min(n * 4, 2048)
    idx = torch.randint(0, n, (n_triplets, 3), device=s_emb.device)

    def angles(emb, idx):
        a, b, c = emb[idx[:, 0]], emb[idx[:, 1]], emb[idx[:, 2]]
        ab = F.normalize(b - a, dim=1)
        ac = F.normalize(c - a, dim=1)
        return (ab * ac).sum(dim=1)  # cosine of angle

    with torch.no_grad():
        t_ang = angles(t_emb, idx)
    s_ang = angles(s_emb, idx)
    return F.smooth_l1_loss(s_ang, t_ang)


def infonce(z_a, z_b, temp=0.07):
    lo = z_a @ z_b.T / temp
    la = torch.arange(len(z_a), device=z_a.device)
    return (F.cross_entropy(lo, la) + F.cross_entropy(lo.T, la)) / 2


class MemoryBank:
    """CRD-style memory bank for more negatives."""
    def __init__(self, embeddings):
        self.bank = F.normalize(torch.FloatTensor(embeddings).to(device), dim=1)
        self.n = len(embeddings)

    def sample_negatives(self, n_neg=4096):
        idx = torch.randint(0, self.n, (n_neg,))
        return self.bank[idx]


def crd_loss(z_student, z_teacher, neg_bank, temp=0.07, n_neg=4096):
    """CRD: contrastive with memory bank negatives."""
    negs = neg_bank.sample_negatives(n_neg)  # (n_neg, proj_dim)
    # Positive: (student_i, teacher_i) pairs
    pos_sim = (z_student * z_teacher).sum(dim=1) / temp  # (B,)
    # Negative: student_i vs all negatives
    neg_sim = z_student @ negs.T / temp  # (B, n_neg)
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+n_neg)
    labels = torch.zeros(len(z_student), dtype=torch.long, device=device)
    return F.cross_entropy(logits, labels)


def ortho_loss(z_shared, z_spec):
    """LATTE orthogonal regularization: shared and specific should be decorrelated."""
    return torch.norm(z_shared.T @ z_spec, p='fro') ** 2 / (len(z_shared) ** 2)


class OrthoAdapter(nn.Module):
    """LATTE-style with orthogonal decomposition."""
    def __init__(self, dim, proj=128):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, dim))
        self.head = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))
        # Decomposition into shared + specific
        self.proj_shared = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, proj))
        self.proj_spec = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, proj))
        self.proj_text = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, proj))

    def forward(self, x, t=None):
        adapted = x + self.adapter(x)
        logits = self.head(adapted).squeeze(-1)
        z_shared = F.normalize(self.proj_shared(adapted), dim=1)
        z_spec = F.normalize(self.proj_spec(adapted), dim=1)
        z_text = F.normalize(self.proj_text(t), dim=1) if t is not None else None
        return adapted, logits, z_shared, z_spec, z_text


# ---- Training function ----
def train_with_loss(name, loss_fn_extra, model_cls=Adapter, epochs=200, lr=1e-3, bs=256):
    print(f"\n--- {name} ---")
    if model_cls == OrthoAdapter:
        model = OrthoAdapter(C_tr.shape[1]).to(device)
    else:
        model = Adapter(C_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bce = nn.BCEWithLogitsLoss()
    best_auc = 0

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(C_tr))
        for s in range(0, len(C_tr), bs):
            b = idx[s:s+bs]
            loss = loss_fn_extra(model, C_tr[b], L_tr[b], Y_tr[b], bce)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()

        if (ep + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                if model_cls == OrthoAdapter:
                    ad_tr, _, _, _, _ = model(C_tr)
                    ad_te, _, _, _, _ = model(C_te)
                else:
                    ad_tr, _, _ = model(C_tr)
                    ad_te, _, _ = model(C_te)
            sc_a = MaxAbsScaler()
            lgbm = LGBMClassifier(**LGBM_P, random_state=42)
            lgbm.fit(sc_a.fit_transform(ad_tr.cpu().numpy()), y_train)
            p = lgbm.predict_proba(sc_a.transform(ad_te.cpu().numpy()))[:, 1]
            auc = roc_auc_score(y_test, p)
            best_auc = max(best_auc, auc)
            print(f"  ep {ep+1}: AUC={auc:.4f} (best={best_auc:.4f})")

    torch.save(model.state_dict(), OUTPUT_DIR / f"{name}.pt")
    return best_auc


# ---- Build memory bank for CRD ----
print("\nBuilding CRD memory bank...")
# Project LLM embeddings to 128d for memory bank
proj_t = nn.Sequential(nn.Linear(L_tr.shape[1], 256), nn.ReLU(), nn.Linear(256, 128)).to(device)
with torch.no_grad():
    bank_embs = F.normalize(proj_t(L_tr), dim=1).cpu().numpy()
mem_bank = MemoryBank(bank_embs)

# ---- Run all experiments ----
results = {"baseline_coles": baseline}

# 1. RKD-D (distance only)
for alpha in [0.01, 0.05, 0.1, 0.3]:
    def loss_rkd_d(model, c, l, y, bce, a=alpha):
        ad, logits, z = model(c)
        return (1 - a) * bce(logits, y) + a * rkd_distance(ad, l)
    results[f"rkd_d_alpha{alpha}"] = train_with_loss(f"rkd_d_alpha{alpha}", loss_rkd_d)

# 2. RKD-D + RKD-A
for alpha in [0.01, 0.05, 0.1]:
    def loss_rkd_da(model, c, l, y, bce, a=alpha):
        ad, logits, z = model(c)
        return (1 - a) * bce(logits, y) + a * 0.5 * (rkd_distance(ad, l) + rkd_angle(ad, l))
    results[f"rkd_da_alpha{alpha}"] = train_with_loss(f"rkd_da_alpha{alpha}", loss_rkd_da)

# 3. CRD with memory bank
for alpha in [0.01, 0.05, 0.1]:
    def loss_crd(model, c, l, y, bce, a=alpha):
        ad, logits, z = model(c)
        # Project teacher to same space
        z_t = F.normalize(proj_t(l), dim=1)
        return (1 - a) * bce(logits, y) + a * crd_loss(z, z_t, mem_bank)
    results[f"crd_alpha{alpha}"] = train_with_loss(f"crd_alpha{alpha}", loss_crd)

# 4. LATTE with orthogonal decomposition
for alpha in [0.05, 0.1, 0.3]:
    lambda_ortho = 0.1
    def loss_ortho(model, c, l, y, bce, a=alpha, lo=lambda_ortho):
        ad, logits, z_shared, z_spec, z_text = model(c, l)
        l_cls = bce(logits, y)
        l_contrast = infonce(z_shared, z_text)
        l_ortho = ortho_loss(z_shared, z_spec)
        return (1 - a) * l_cls + a * l_contrast + lo * l_ortho
    results[f"ortho_alpha{alpha}"] = train_with_loss(f"ortho_alpha{alpha}", loss_ortho, model_cls=OrthoAdapter)

# ---- Summary ----
print("\n" + "=" * 60)
print("EMBEDDING DISTILLATION SUMMARY (Gender)")
print("=" * 60)
for n, v in sorted(results.items(), key=lambda x: -x[1]):
    d = v - baseline
    print(f"  {n:<25} AUC={v:.4f} ({'+' if d >= 0 else ''}{d:.4f})")

with open(OUTPUT_DIR / "rkd_results.json", "w") as f:
    json.dump(results, f, indent=2)
pd.DataFrame([{"method": k, "auc": v} for k, v in results.items()]).to_csv(
    OUTPUT_DIR / "rkd_results.csv", index=False)
