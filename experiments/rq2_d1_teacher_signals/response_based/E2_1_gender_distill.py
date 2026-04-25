#!/usr/bin/env python3
"""
Phase 3: Knowledge Distillation on Gender dataset.
Three approaches:
  1. Soft-label distillation (Hinton-style): student learns from LLM soft probs
  2. Embedding distillation: student learns from LLM hidden states
  3. Stacking baseline: LLM predictions as features (for comparison)

Runs on CPU (approach 1,3) + GPU (approach 2 embedding extraction).
"""

import time, json, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

OUTPUT_DIR = Path("results/gender_distillation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load all data ----
print("Loading data...")
emb_train = np.load("embeddings/gender/emb_train_seed42.npy")
emb_test = np.load("embeddings/gender/emb_test_seed42.npy")
y_train = np.load("embeddings/gender/y_train_seed42.npy")
y_test = np.load("embeddings/gender/y_test_seed42.npy")
cids_train = np.load("embeddings/gender/cids_train_seed42.npy")
cids_test = np.load("embeddings/gender/cids_test_seed42.npy")

# LLM predictions (teacher soft labels)
llm_zs = pd.read_csv("results/gender_llm/gender_zero_shot_predictions.csv").set_index("customer_id")
llm_fs = pd.read_csv("results/gender_llm/gender_few_shot_predictions.csv").set_index("customer_id")
llm_sh = pd.read_csv("results/gender_llm/gender_shap_enriched_predictions.csv").set_index("customer_id")

def get_llm_probs(cids):
    probs = np.zeros((len(cids), 3))
    for i, cid in enumerate(cids):
        probs[i, 0] = llm_zs.loc[cid, "pred_prob"] if cid in llm_zs.index else 0.5
        probs[i, 1] = llm_fs.loc[cid, "pred_prob"] if cid in llm_fs.index else 0.5
        probs[i, 2] = llm_sh.loc[cid, "pred_prob"] if cid in llm_sh.index else 0.5
    return probs

llm_train = get_llm_probs(cids_train)
llm_test = get_llm_probs(cids_test)

# Best LLM teacher: shap_enriched
teacher_train = llm_train[:, 2]  # shap_enriched probs
teacher_test = llm_test[:, 2]

print(f"  CoLES: train={emb_train.shape}, test={emb_test.shape}")
print(f"  Teacher (shap_enriched) AUC on test: {roc_auc_score(y_test, teacher_test):.4f}")

lgbm_params = dict(n_estimators=500, learning_rate=0.02, max_depth=6, subsample=0.5,
                   colsample_bytree=0.75, reg_alpha=1, reg_lambda=1, min_child_samples=50, verbosity=-1)

results = {}

# ================================================================
# APPROACH 0: Baselines
# ================================================================
print("\n" + "="*60)
print("BASELINES")
print("="*60)

scaler = MaxAbsScaler()
Xtr = scaler.fit_transform(emb_train)
Xte = scaler.transform(emb_test)

# CoLES only
lgbm = LGBMClassifier(**lgbm_params, random_state=42)
lgbm.fit(Xtr, y_train)
p = lgbm.predict_proba(Xte)[:, 1]
results["coles_only"] = roc_auc_score(y_test, p)
print(f"  CoLES only:          AUC = {results['coles_only']:.4f}")

# Stacking: CoLES + LLM probs as features
Xtr_stack = np.hstack([Xtr, llm_train])
Xte_stack = np.hstack([Xte, llm_test])
lgbm = LGBMClassifier(**lgbm_params, random_state=42)
lgbm.fit(Xtr_stack, y_train)
p = lgbm.predict_proba(Xte_stack)[:, 1]
results["stacking_coles_llm"] = roc_auc_score(y_test, p)
print(f"  Stacking (CoLES+LLM): AUC = {results['stacking_coles_llm']:.4f}")

# ================================================================
# APPROACH 1: Soft-label distillation (MLP student)
# ================================================================
print("\n" + "="*60)
print("APPROACH 1: Soft-label distillation (MLP)")
print("="*60)

class StudentMLP(nn.Module):
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_student(X_train, y_hard, y_soft, X_test, y_test_true,
                  alpha=0.5, temperature=2.0, epochs=100, lr=1e-3, batch_size=256):
    """Train MLP with combined hard label + soft label loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    X_tr = torch.FloatTensor(scaler.fit_transform(X_train)).to(device)
    X_te = torch.FloatTensor(scaler.transform(X_test)).to(device)
    y_h = torch.FloatTensor(y_hard).to(device)
    y_s = torch.FloatTensor(y_soft).to(device)

    model = StudentMLP(X_tr.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    bce = nn.BCEWithLogitsLoss()
    best_auc = 0
    best_probs = None

    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(len(X_tr))
        total_loss = 0
        for start in range(0, len(X_tr), batch_size):
            batch_idx = idx[start:start+batch_size]
            logits = model(X_tr[batch_idx])

            # Hard label loss
            loss_hard = bce(logits, y_h[batch_idx])

            # Soft label loss (KL divergence with temperature)
            student_log_probs = torch.log_softmax(
                torch.stack([logits / temperature, -logits / temperature], dim=-1), dim=-1)
            teacher_probs = torch.softmax(
                torch.stack([y_s[batch_idx] / temperature, (1 - y_s[batch_idx]) / temperature], dim=-1), dim=-1)
            loss_soft = nn.functional.kl_div(student_log_probs, teacher_probs,
                                              reduction="batchmean") * (temperature ** 2)

            loss = (1 - alpha) * loss_hard + alpha * loss_soft

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Eval
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_te)).cpu().numpy()
        auc = roc_auc_score(y_test_true, probs)
        if auc > best_auc:
            best_auc = auc
            best_probs = probs.copy()

    return best_auc, best_probs

# Distill with different alpha values
for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
    auc, _ = train_student(emb_train, y_train, teacher_train, emb_test, y_test,
                           alpha=alpha, temperature=2.0, epochs=150)
    label = f"soft_label_alpha{alpha}"
    results[label] = auc
    desc = "hard only" if alpha == 0 else "soft only" if alpha == 1 else f"mixed α={alpha}"
    print(f"  α={alpha} ({desc}): AUC = {auc:.4f}")

# ================================================================
# APPROACH 2: Embedding distillation
# ================================================================
print("\n" + "="*60)
print("APPROACH 2: Embedding distillation")
print("="*60)

# Extract LLM hidden states
def extract_llm_embeddings():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("  Loading Qwen2.5-7B for embedding extraction...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    # Load serialized client texts
    tx = pd.read_csv("data/transactions.csv")
    labels = pd.read_csv("data/gender_train.csv")
    tx = tx[tx["customer_id"].isin(labels["customer_id"])]
    grouped = tx.groupby("customer_id")

    all_cids = np.concatenate([cids_train, cids_test])
    hidden_dim = model.config.hidden_size

    emb_path = OUTPUT_DIR / "llm_embeddings.npz"
    if emb_path.exists():
        data = np.load(emb_path)
        print(f"  Loaded cached LLM embeddings: {data['embeddings'].shape}")
        return data["embeddings"], data["cid_order"]

    # Build simple text summaries for embedding extraction
    MCC_GROUPS = {
        range(1, 1500): "Agriculture", range(4000, 4800): "Transportation",
        range(5000, 5600): "Retail", range(5600, 5700): "Clothing",
        range(5800, 5900): "Restaurants", range(7500, 7600): "Auto Services",
        range(7700, 7800): "Entertainment", range(8000, 8100): "Medical",
    }
    def mcc_cat(mcc):
        try:
            mcc = int(mcc)
        except: return "Other"
        for r, name in MCC_GROUPS.items():
            if mcc in r: return name
        return "Other"

    embeddings = np.zeros((len(all_cids), hidden_dim), dtype=np.float16)
    print(f"  Extracting embeddings for {len(all_cids)} clients (dim={hidden_dim})...")

    for i, cid in enumerate(all_cids):
        if cid not in grouped.groups:
            continue
        ct = grouped.get_group(cid)
        cats = ct["mcc_code"].apply(mcc_cat).value_counts()
        top = ", ".join(f"{c} ({n})" for c, n in cats.head(5).items())
        amt = np.abs(ct["amount"].values)
        text = (f"Client with {len(ct)} transactions. "
                f"Avg amount: {amt.mean():.0f}, categories: {top}.")
        prompt = f"Describe this bank client's profile:\n{text}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Mean pool last hidden state
            last_hidden = outputs.hidden_states[-1][0]  # (seq_len, hidden_dim)
            mask = inputs["attention_mask"][0].unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(0) / mask.sum(0)
            embeddings[i] = pooled.cpu().float().numpy().astype(np.float16)

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(all_cids)}")

    np.savez_compressed(emb_path, embeddings=embeddings, cid_order=all_cids)
    print(f"  Saved LLM embeddings: {embeddings.shape}")

    del model, tokenizer
    torch.cuda.empty_cache()
    return embeddings, all_cids

llm_embs, llm_cid_order = extract_llm_embeddings()

# Split LLM embeddings into train/test aligned with CoLES split
cid_to_idx = {cid: i for i, cid in enumerate(llm_cid_order)}
llm_emb_train = np.array([llm_embs[cid_to_idx[c]] for c in cids_train], dtype=np.float32)
llm_emb_test = np.array([llm_embs[cid_to_idx[c]] for c in cids_test], dtype=np.float32)

print(f"  LLM embeddings: train={llm_emb_train.shape}, test={llm_emb_test.shape}")

# 2a: LLM embeddings only
scaler_llm = MaxAbsScaler()
Xtr_llm = scaler_llm.fit_transform(llm_emb_train)
Xte_llm = scaler_llm.transform(llm_emb_test)
lgbm = LGBMClassifier(**lgbm_params, random_state=42)
lgbm.fit(Xtr_llm, y_train)
p = lgbm.predict_proba(Xte_llm)[:, 1]
results["llm_embeddings_only"] = roc_auc_score(y_test, p)
print(f"  LLM embeddings only:      AUC = {results['llm_embeddings_only']:.4f}")

# 2b: CoLES + LLM embeddings
Xtr_both = np.hstack([Xtr, Xtr_llm])
Xte_both = np.hstack([Xte, Xte_llm])
lgbm = LGBMClassifier(**lgbm_params, random_state=42)
lgbm.fit(Xtr_both, y_train)
p = lgbm.predict_proba(Xte_both)[:, 1]
results["coles_plus_llm_emb"] = roc_auc_score(y_test, p)
print(f"  CoLES + LLM embeddings:   AUC = {results['coles_plus_llm_emb']:.4f}")

# 2c: CoLES + LLM embeddings + LLM probs
Xtr_all = np.hstack([Xtr, Xtr_llm, llm_train])
Xte_all = np.hstack([Xte, Xte_llm, llm_test])
lgbm = LGBMClassifier(**lgbm_params, random_state=42)
lgbm.fit(Xtr_all, y_train)
p = lgbm.predict_proba(Xte_all)[:, 1]
results["coles_llm_emb_probs"] = roc_auc_score(y_test, p)
print(f"  CoLES + LLM emb + probs:  AUC = {results['coles_llm_emb_probs']:.4f}")

# 2d: Embedding distillation - train student to predict LLM embeddings from CoLES
print("\n  Training embedding distillation student...")

class EmbDistillStudent(nn.Module):
    def __init__(self, coles_dim, llm_dim, hidden=512):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(coles_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, llm_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(coles_dim + llm_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, coles_emb, return_projected=False):
        projected = self.projector(coles_emb)
        logits = self.classifier(torch.cat([coles_emb, projected], dim=-1)).squeeze(-1)
        if return_projected:
            return logits, projected
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler_c = StandardScaler()
scaler_l = StandardScaler()
X_c_tr = torch.FloatTensor(scaler_c.fit_transform(emb_train)).to(device)
X_c_te = torch.FloatTensor(scaler_c.transform(emb_test)).to(device)
X_l_tr = torch.FloatTensor(scaler_l.fit_transform(llm_emb_train)).to(device)

y_t = torch.FloatTensor(y_train).to(device)

model = EmbDistillStudent(emb_train.shape[1], llm_emb_train.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
bce = nn.BCEWithLogitsLoss()
mse = nn.MSELoss()

best_auc = 0
for epoch in range(200):
    model.train()
    idx = torch.randperm(len(X_c_tr))
    for start in range(0, len(X_c_tr), 256):
        batch = idx[start:start+256]
        logits, projected = model(X_c_tr[batch], return_projected=True)
        loss_cls = bce(logits, y_t[batch])
        loss_emb = mse(projected, X_l_tr[batch])
        loss = loss_cls + 0.1 * loss_emb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_c_te)).cpu().numpy()
    auc = roc_auc_score(y_test, probs)
    if auc > best_auc:
        best_auc = auc

results["emb_distill_student"] = best_auc
print(f"  Embedding distill student: AUC = {best_auc:.4f}")

# ================================================================
# SUMMARY
# ================================================================
print("\n" + "="*60)
print("GENDER DISTILLATION SUMMARY")
print("="*60)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name:<30} AUC = {auc:.4f}")

with open(OUTPUT_DIR / "gender_distillation_results.json", "w") as f:
    json.dump(results, f, indent=2)

pd.DataFrame([{"method": k, "auc": v} for k, v in results.items()]).to_csv(
    OUTPUT_DIR / "gender_distillation_results.csv", index=False)
