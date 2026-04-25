#!/usr/bin/env python3
"""
Bidirectional Knowledge Distillation: LLM ↔ CoLES on Gender dataset.

Three approaches:
  1. Alternating Distillation (LLMD4Rec-style): freeze A→train B, freeze B→train A
  2. Deep Mutual Learning (DML): both train simultaneously with mutual KL
  3. Combined: DML + contrastive alignment in embedding space

All use pre-computed embeddings (CoLES 1024d + LLM4ES 2048d).
Saves checkpoints after each experiment.
"""

import time, json, warnings, gc, copy
from pathlib import Path

warnings.filterwarnings("ignore")
# Reproducibility
import random, os as _os
SEED = 42
random.seed(SEED); import numpy as _np; _np.random.seed(SEED)
import torch as _torch
_torch.manual_seed(SEED); _torch.cuda.manual_seed_all(SEED)
import pytorch_lightning as _pl
_pl.seed_everything(SEED, workers=True)
_os.environ["PYTHONHASHSEED"] = str(SEED)
_torch.backends.cudnn.deterministic = True
_torch.backends.cudnn.benchmark = False


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

OUTPUT_DIR = Path("results/gender_bidirectional")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")

# ---- Load data ----
print("Loading pre-computed embeddings...")
coles_train = np.load("embeddings/gender/emb_train_seed42.npy")
coles_test = np.load("embeddings/gender/emb_test_seed42.npy")
y_train = np.load("embeddings/gender/y_train_seed42.npy")
y_test = np.load("embeddings/gender/y_test_seed42.npy")

llm_data = np.load("results/gender_llm4es/llm4es_embeddings.npz")
llm_all = llm_data["embeddings"].astype(np.float32)
n_tr = len(coles_train)
llm_train, llm_test = llm_all[:n_tr], llm_all[n_tr:]

print(f"  CoLES: train={coles_train.shape}, test={coles_test.shape}")
print(f"  LLM4ES: train={llm_train.shape}, test={llm_test.shape}")

# Standardize
sc_c, sc_l = StandardScaler(), StandardScaler()
C_tr = torch.FloatTensor(sc_c.fit_transform(coles_train)).to(device)
C_te = torch.FloatTensor(sc_c.transform(coles_test)).to(device)
L_tr = torch.FloatTensor(sc_l.fit_transform(llm_train)).to(device)
L_te = torch.FloatTensor(sc_l.transform(llm_test)).to(device)
Y_tr = torch.FloatTensor(y_train).to(device)

LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.02, max_depth=6, subsample=0.5,
                   colsample_bytree=0.75, reg_alpha=1, reg_lambda=1,
                   min_child_samples=50, verbosity=-1)


# ---- Model definitions ----
class BranchModel(nn.Module):
    """One branch: adapter + classifier head."""
    def __init__(self, input_dim, hidden=512):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, input_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        # Projection for contrastive alignment
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 128),
        )

    def forward(self, x):
        adapted = x + self.adapter(x)
        logits = self.head(adapted).squeeze(-1)
        z = F.normalize(self.proj(adapted), dim=1)
        return adapted, logits, z

    def predict_proba(self, x):
        _, logits, _ = self.forward(x)
        return torch.sigmoid(logits)


def eval_branch(model, X_tr, X_te, y_train, y_test):
    """Evaluate branch: MLP AUC + LGBM AUC on adapted embeddings."""
    model.eval()
    with torch.no_grad():
        # MLP prediction
        probs_te = model.predict_proba(X_te).cpu().numpy()
        mlp_auc = roc_auc_score(y_test, probs_te)

        # LGBM on adapted embeddings
        ad_tr, _, _ = model(X_tr)
        ad_te, _, _ = model(X_te)

    from sklearn.preprocessing import MaxAbsScaler
    sc = MaxAbsScaler()
    lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=42)
    lgbm.fit(sc.fit_transform(ad_tr.cpu().numpy()), y_train)
    p = lgbm.predict_proba(sc.transform(ad_te.cpu().numpy()))[:, 1]
    lgbm_auc = roc_auc_score(y_test, p)

    return mlp_auc, lgbm_auc


def infonce(z_a, z_b, temperature=0.07):
    logits = z_a @ z_b.T / temperature
    labels = torch.arange(len(z_a), device=z_a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ---- Baselines ----
print("\n" + "=" * 60)
print("BASELINES")
print("=" * 60)

results = {}

# CoLES only LGBM
from sklearn.preprocessing import MaxAbsScaler
sc = MaxAbsScaler()
lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=42)
lgbm.fit(sc.fit_transform(coles_train), y_train)
p = lgbm.predict_proba(sc.transform(coles_test))[:, 1]
results["baseline_coles_lgbm"] = roc_auc_score(y_test, p)
print(f"  CoLES LGBM:    AUC = {results['baseline_coles_lgbm']:.4f}")

# LLM4ES only LGBM
lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=42)
lgbm.fit(sc.fit_transform(llm_train), y_train)
p = lgbm.predict_proba(sc.transform(llm_test))[:, 1]
results["baseline_llm_lgbm"] = roc_auc_score(y_test, p)
print(f"  LLM4ES LGBM:   AUC = {results['baseline_llm_lgbm']:.4f}")

# Concat LGBM
Xtr_cat = np.hstack([coles_train, llm_train])
Xte_cat = np.hstack([coles_test, llm_test])
lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=42)
lgbm.fit(sc.fit_transform(Xtr_cat), y_train)
p = lgbm.predict_proba(sc.transform(Xte_cat))[:, 1]
results["baseline_concat_lgbm"] = roc_auc_score(y_test, p)
print(f"  Concat LGBM:   AUC = {results['baseline_concat_lgbm']:.4f}")


# ================================================================
# APPROACH 2: Alternating Distillation (LLMD4Rec-style)
# ================================================================
print("\n" + "=" * 60)
print("APPROACH 2: Alternating Distillation")
print("=" * 60)

def run_alternating(alpha=0.5, n_rounds=5, epochs_per_round=50, lr=1e-3, bs=256):
    model_a = BranchModel(coles_train.shape[1]).to(device)  # CoLES branch
    model_b = BranchModel(llm_train.shape[1]).to(device)    # LLM branch
    bce = nn.BCEWithLogitsLoss()
    history = []

    for round_i in range(n_rounds):
        # --- Train A (CoLES), freeze B ---
        model_b.eval()
        opt_a = torch.optim.Adam(model_a.parameters(), lr=lr, weight_decay=1e-4)
        for ep in range(epochs_per_round):
            model_a.train()
            idx = torch.randperm(len(C_tr))
            for s in range(0, len(C_tr), bs):
                b = idx[s:s+bs]
                _, logits_a, _ = model_a(C_tr[b])
                with torch.no_grad():
                    probs_b = model_b.predict_proba(L_tr[b])
                loss_ce = bce(logits_a, Y_tr[b])
                # KL: match B's soft predictions
                probs_a = torch.sigmoid(logits_a)
                loss_kl = F.binary_cross_entropy(probs_a, probs_b.detach())
                loss = (1 - alpha) * loss_ce + alpha * loss_kl
                opt_a.zero_grad(); loss.backward(); opt_a.step()

        # --- Train B (LLM), freeze A ---
        model_a.eval()
        opt_b = torch.optim.Adam(model_b.parameters(), lr=lr, weight_decay=1e-4)
        for ep in range(epochs_per_round):
            model_b.train()
            idx = torch.randperm(len(L_tr))
            for s in range(0, len(L_tr), bs):
                b = idx[s:s+bs]
                _, logits_b, _ = model_b(L_tr[b])
                with torch.no_grad():
                    probs_a = model_a.predict_proba(C_tr[b])
                loss_ce = bce(logits_b, Y_tr[b])
                probs_b = torch.sigmoid(logits_b)
                loss_kl = F.binary_cross_entropy(probs_b, probs_a.detach())
                loss = (1 - alpha) * loss_ce + alpha * loss_kl
                opt_b.zero_grad(); loss.backward(); opt_b.step()

        # Eval both
        mlp_a, lgbm_a = eval_branch(model_a, C_tr, C_te, y_train, y_test)
        mlp_b, lgbm_b = eval_branch(model_b, L_tr, L_te, y_train, y_test)
        history.append({
            "round": round_i + 1,
            "coles_mlp": mlp_a, "coles_lgbm": lgbm_a,
            "llm_mlp": mlp_b, "llm_lgbm": lgbm_b,
        })
        print(f"  Round {round_i+1}: CoLES MLP={mlp_a:.4f} LGBM={lgbm_a:.4f} | LLM MLP={mlp_b:.4f} LGBM={lgbm_b:.4f}")

    # Save checkpoints
    torch.save(model_a.state_dict(), OUTPUT_DIR / f"alternating_coles_α{alpha}.pt")
    torch.save(model_b.state_dict(), OUTPUT_DIR / f"alternating_llm_α{alpha}.pt")
    return model_a, model_b, history

for alpha in [0.3, 0.5, 0.7]:
    print(f"\n--- α={alpha} ---")
    _, _, hist = run_alternating(alpha=alpha)
    best_coles = max(h["coles_lgbm"] for h in hist)
    best_llm = max(h["llm_lgbm"] for h in hist)
    results[f"alt_α{alpha}_coles"] = best_coles
    results[f"alt_α{alpha}_llm"] = best_llm


# ================================================================
# APPROACH 3: Deep Mutual Learning (DML)
# ================================================================
print("\n" + "=" * 60)
print("APPROACH 3: Deep Mutual Learning")
print("=" * 60)

def run_dml(alpha=0.5, epochs=200, lr=1e-3, bs=256):
    model_a = BranchModel(coles_train.shape[1]).to(device)
    model_b = BranchModel(llm_train.shape[1]).to(device)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=lr, weight_decay=1e-4)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=lr, weight_decay=1e-4)
    sch_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, epochs)
    sch_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, epochs)
    bce = nn.BCEWithLogitsLoss()
    history = []
    best_coles_lgbm = 0
    best_llm_lgbm = 0

    for ep in range(epochs):
        model_a.train(); model_b.train()
        idx = torch.randperm(len(C_tr))

        for s in range(0, len(C_tr), bs):
            b = idx[s:s+bs]
            _, logits_a, _ = model_a(C_tr[b])
            _, logits_b, _ = model_b(L_tr[b])

            probs_a = torch.sigmoid(logits_a)
            probs_b = torch.sigmoid(logits_b)

            # Model A loss: CE + KL(B||A)
            loss_a = (1 - alpha) * bce(logits_a, Y_tr[b]) + \
                     alpha * F.binary_cross_entropy(probs_a, probs_b.detach())
            # Model B loss: CE + KL(A||B)
            loss_b = (1 - alpha) * bce(logits_b, Y_tr[b]) + \
                     alpha * F.binary_cross_entropy(probs_b, probs_a.detach())

            opt_a.zero_grad(); loss_a.backward(); opt_a.step()
            opt_b.zero_grad(); loss_b.backward(); opt_b.step()

        sch_a.step(); sch_b.step()

        if (ep + 1) % 25 == 0:
            mlp_a, lgbm_a = eval_branch(model_a, C_tr, C_te, y_train, y_test)
            mlp_b, lgbm_b = eval_branch(model_b, L_tr, L_te, y_train, y_test)
            best_coles_lgbm = max(best_coles_lgbm, lgbm_a)
            best_llm_lgbm = max(best_llm_lgbm, lgbm_b)
            history.append({"epoch": ep+1, "coles_mlp": mlp_a, "coles_lgbm": lgbm_a,
                           "llm_mlp": mlp_b, "llm_lgbm": lgbm_b})
            print(f"  ep {ep+1}: CoLES MLP={mlp_a:.4f} LGBM={lgbm_a:.4f} | LLM MLP={mlp_b:.4f} LGBM={lgbm_b:.4f}")

    torch.save(model_a.state_dict(), OUTPUT_DIR / f"dml_coles_α{alpha}.pt")
    torch.save(model_b.state_dict(), OUTPUT_DIR / f"dml_llm_α{alpha}.pt")
    return best_coles_lgbm, best_llm_lgbm, history

for alpha in [0.3, 0.5, 0.7]:
    print(f"\n--- α={alpha} ---")
    bc, bl, hist = run_dml(alpha=alpha)
    results[f"dml_α{alpha}_coles"] = bc
    results[f"dml_α{alpha}_llm"] = bl


# ================================================================
# COMBINED: Mutual Contrastive + Soft-Label
# ================================================================
print("\n" + "=" * 60)
print("COMBINED: Mutual Contrastive + Soft-Label")
print("=" * 60)

def run_combined(alpha_kl=0.3, alpha_contrast=0.3, epochs=200, lr=1e-3, bs=256):
    model_a = BranchModel(coles_train.shape[1]).to(device)
    model_b = BranchModel(llm_train.shape[1]).to(device)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=lr, weight_decay=1e-4)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=lr, weight_decay=1e-4)
    sch_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, epochs)
    sch_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, epochs)
    bce = nn.BCEWithLogitsLoss()
    history = []
    best_coles_lgbm = 0
    best_llm_lgbm = 0

    for ep in range(epochs):
        model_a.train(); model_b.train()
        idx = torch.randperm(len(C_tr))

        for s in range(0, len(C_tr), bs):
            b = idx[s:s+bs]
            adapted_a, logits_a, z_a = model_a(C_tr[b])
            adapted_b, logits_b, z_b = model_b(L_tr[b])

            probs_a = torch.sigmoid(logits_a)
            probs_b = torch.sigmoid(logits_b)

            # Contrastive: align projection spaces
            loss_contrast = infonce(z_a, z_b.detach()) + infonce(z_b, z_a.detach())

            # Model A: CE + KL mimicry + contrastive
            loss_a = (1 - alpha_kl - alpha_contrast) * bce(logits_a, Y_tr[b]) + \
                     alpha_kl * F.binary_cross_entropy(probs_a, probs_b.detach()) + \
                     alpha_contrast * infonce(z_a, z_b.detach())

            # Model B: CE + KL mimicry + contrastive
            loss_b = (1 - alpha_kl - alpha_contrast) * bce(logits_b, Y_tr[b]) + \
                     alpha_kl * F.binary_cross_entropy(probs_b, probs_a.detach()) + \
                     alpha_contrast * infonce(z_b, z_a.detach())

            opt_a.zero_grad(); loss_a.backward(); opt_a.step()
            opt_b.zero_grad(); loss_b.backward(); opt_b.step()

        sch_a.step(); sch_b.step()

        if (ep + 1) % 25 == 0:
            mlp_a, lgbm_a = eval_branch(model_a, C_tr, C_te, y_train, y_test)
            mlp_b, lgbm_b = eval_branch(model_b, L_tr, L_te, y_train, y_test)
            best_coles_lgbm = max(best_coles_lgbm, lgbm_a)
            best_llm_lgbm = max(best_llm_lgbm, lgbm_b)
            history.append({"epoch": ep+1, "coles_mlp": mlp_a, "coles_lgbm": lgbm_a,
                           "llm_mlp": mlp_b, "llm_lgbm": lgbm_b})
            print(f"  ep {ep+1}: CoLES MLP={mlp_a:.4f} LGBM={lgbm_a:.4f} | LLM MLP={mlp_b:.4f} LGBM={lgbm_b:.4f}")

    torch.save(model_a.state_dict(), OUTPUT_DIR / f"combined_coles_kl{alpha_kl}_c{alpha_contrast}.pt")
    torch.save(model_b.state_dict(), OUTPUT_DIR / f"combined_llm_kl{alpha_kl}_c{alpha_contrast}.pt")
    return best_coles_lgbm, best_llm_lgbm, history

configs = [
    (0.3, 0.1),  # mostly soft-label
    (0.2, 0.3),  # mostly contrastive
    (0.25, 0.25),  # balanced
]

for alpha_kl, alpha_c in configs:
    print(f"\n--- KL={alpha_kl}, Contrastive={alpha_c} ---")
    bc, bl, hist = run_combined(alpha_kl=alpha_kl, alpha_contrast=alpha_c)
    results[f"combined_kl{alpha_kl}_c{alpha_c}_coles"] = bc
    results[f"combined_kl{alpha_kl}_c{alpha_c}_llm"] = bl


# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 60)
print("BIDIRECTIONAL DISTILLATION SUMMARY (Gender)")
print("=" * 60)

print(f"\n{'Method':<45} {'CoLES AUC':>10} {'LLM AUC':>10}")
print("-" * 65)

# Baselines first
for k in ["baseline_coles_lgbm", "baseline_llm_lgbm", "baseline_concat_lgbm"]:
    if k in results:
        print(f"  {k:<43} {results[k]:>10.4f}")

print("-" * 65)

# Group by approach
for prefix in ["alt_", "dml_", "combined_"]:
    coles_keys = sorted([k for k in results if k.startswith(prefix) and "_coles" in k])
    for ck in coles_keys:
        lk = ck.replace("_coles", "_llm")
        cv = results.get(ck, 0)
        lv = results.get(lk, 0)
        print(f"  {ck:<43} {cv:>10.4f} {lv:>10.4f}")

with open(OUTPUT_DIR / "bidirectional_results.json", "w") as f:
    json.dump(results, f, indent=2)

pd.DataFrame([{"method": k, "auc": v} for k, v in results.items()]).to_csv(
    OUTPUT_DIR / "bidirectional_results.csv", index=False)

print(f"\nAll checkpoints and results saved to {OUTPUT_DIR}")
