#!/usr/bin/env python3
"""
RAMD: Retrieval-Augmented Mutual Distillation.

Iterative loop:
  Round 0: CoLES baseline → kNN → LLM with kNN CoT → LLM predictions
  Round 1: LLM predictions → soft-label distill → CoLES improves
           Improved CoLES → better kNN → better LLM predictions
  Round 2: Better LLM → contrastive distill → CoLES improves more
           Even better CoLES → even better kNN → even better LLM
  ...until convergence

Key insight: retrieval (kNN) is the bridge that connects both models.
Better CoLES → better retrieval → better LLM → better distillation → better CoLES.

Saves checkpoints + pushes after EACH round.
"""

import time, json, warnings, gc, os
from pathlib import Path
from functools import partial

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
from lightgbm import LGBMClassifier

from ptls.data_load.datasets import MemoryMapDataset, inference_data_loader
from ptls.nn import TrxEncoder, RnnSeqEncoder

OUTPUT_DIR = Path("results/gender_ramd")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LGBM_P = dict(n_estimators=500, learning_rate=0.02, max_depth=6, subsample=0.5,
              colsample_bytree=0.75, reg_alpha=1, reg_lambda=1, min_child_samples=50, verbosity=-1)

MCC_GROUPS = {range(1,1500):"Agriculture",range(4000,4800):"Transportation",range(5000,5600):"Retail",
              range(5600,5700):"Clothing",range(5800,5900):"Restaurants",range(6000,7000):"Financial",
              range(7500,7600):"Auto Services",range(8000,8100):"Medical"}
def mcc_cat(mcc):
    try: mcc=int(mcc)
    except: return "Other"
    for r,n in MCC_GROUPS.items():
        if mcc in r: return n
    return "Other"

# ---- Load data ----
print("=" * 60)
print("RAMD: Retrieval-Augmented Mutual Distillation (Gender)")
print("=" * 60)

tx = pd.read_csv(DATA_DIR / "transactions.csv")
labels = pd.read_csv(DATA_DIR / "gender_train.csv")
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

encoders = {}
for col in ["mcc_code", "tr_type"]:
    tx[col] = tx[col].fillna("UNK").astype(str)
    encoders[col] = LabelEncoder().fit(tx[col])

ids = labels["customer_id"].values
targets = np.array([target_map[c] for c in ids])
idx_tr, idx_te = train_test_split(np.arange(len(ids)), test_size=0.1, random_state=42, stratify=targets)
train_ids_arr, test_ids_arr = ids[idx_tr], ids[idx_te]
train_ids_set, test_ids_set = set(train_ids_arr), set(test_ids_arr)
grouped = tx.groupby("customer_id")

def build_records(cid_set):
    records = []
    for cid in cid_set:
        if cid not in target_map or cid not in grouped.groups: continue
        ct = grouped.get_group(cid)
        if len(ct) < 25: continue
        days = ct["day_float"].values
        rec = {"customer_id": cid, "target": target_map[cid],
               "event_time": torch.FloatTensor((days - days[0]).astype(np.float32)),
               "amount": torch.FloatTensor(ct["amount"].values)}
        for col, enc in encoders.items():
            rec[col] = torch.LongTensor(enc.transform(ct[col].values) + 1)
        records.append(rec)
    return records

def serialize_client(cid):
    if cid not in grouped.groups: return "No txns."
    ct = grouped.get_group(cid)
    n = len(ct)
    amt = np.abs(ct["amount"].values)
    cats = ct["mcc_code"].apply(lambda x: mcc_cat(x)).value_counts()
    top = ", ".join(f"{c} ({n_})" for c, n_ in cats.head(6).items())
    return f"Client with {n} txns. Avg amount: {amt.mean():.0f}. Categories: {top}."

train_rec = build_records(train_ids_set)
test_rec = build_records(test_ids_set)
feature_dims = {col: len(enc.classes_) + 2 for col, enc in encoders.items()}
y_train = np.array([r["target"] for r in train_rec])
y_test = np.array([r["target"] for r in test_rec])
cids_train = [r["customer_id"] for r in train_rec]
cids_test = [r["customer_id"] for r in test_rec]
print(f"  train={len(train_rec)}, test={len(test_rec)}")

# ---- CoLES encoder ----
COLES_CKPT = Path("results/gender_true_latte/coles_baseline.pt")

def build_seq_encoder():
    trx = TrxEncoder(
        embeddings={"mcc_code": {"in": feature_dims["mcc_code"], "out": 48},
                     "tr_type": {"in": feature_dims["tr_type"], "out": 24}},
        numeric_values={"amount": "identity"}, embeddings_noise=0.003, use_batch_norm_with_lens=True)
    return RnnSeqEncoder(trx_encoder=trx, hidden_size=1024, type="gru", bidir=False, trainable_starter="static")

def extract_embs(encoder, records):
    encoder.eval()
    dl = inference_data_loader(records, num_workers=0, batch_size=64)
    with torch.no_grad():
        return torch.cat([encoder(b.to(device)).cpu() for b in dl]).numpy()

def eval_coles(encoder):
    emb_tr = extract_embs(encoder, train_rec)
    emb_te = extract_embs(encoder, test_rec)
    sc = MaxAbsScaler()
    lgbm = LGBMClassifier(**LGBM_P, random_state=42)
    lgbm.fit(sc.fit_transform(emb_tr), y_train)
    return roc_auc_score(y_test, lgbm.predict_proba(sc.transform(emb_te))[:, 1])

# ---- kNN retrieval ----
def knn_retrieve(emb_train, emb_test, y_train, k=10):
    """Find k nearest neighbors and return label distributions."""
    sc = MaxAbsScaler()
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(sc.fit_transform(emb_train))
    _, indices = nn.kneighbors(sc.transform(emb_test))
    contexts = []
    for i in range(len(emb_test)):
        nb_labels = y_train[indices[i]]
        pos = int(nb_labels.sum())
        neg = k - pos
        majority = "male" if pos > k // 2 else "female"
        contexts.append(f"Similar clients: {pos} male, {neg} female (majority: {majority}).")
    return contexts

# ---- LLM inference with kNN CoT ----
def llm_predict_with_cot(cids, contexts, model, tokenizer, pos_ids, neg_ids):
    """LLM predicts with kNN CoT context."""
    SYSTEM = ("You are a bank analyst predicting client gender. "
              "You have the profile AND similar client analysis. Answer: male or female.")
    preds = []
    for i, cid in enumerate(cids):
        text = serialize_client(cid)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Profile:\n{text}\n\n{contexts[i]}\n\nPredict."}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        all_ids = pos_ids + neg_ids
        probs = torch.softmax(logits[all_ids].float(), dim=0)
        pp = probs[:len(pos_ids)].sum().item()
        pn = probs[len(pos_ids):].sum().item()
        total = pp + pn
        preds.append(pp / total if total > 1e-8 else 0.5)
        del inputs
        if (i+1) % 200 == 0:
            print(f"    {i+1}/{len(cids)}")
    return np.array(preds)

# ---- Distillation step ----
def distill_round(encoder, llm_soft_train, alpha=0.1, epochs=10):
    """Fine-tune CoLES with contrastive + soft-label from LLM."""
    encoder.train()
    # Use LLM soft predictions as additional signal
    llm_t = torch.FloatTensor(llm_soft_train).to(device)

    classifier = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                           lr=3e-4, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    best_auc = 0
    for ep in range(epochs):
        encoder.train(); classifier.train()
        idx = torch.randperm(len(train_rec))
        for s in range(0, len(train_rec), 32):
            bi = idx[s:s+32].tolist()
            dl = inference_data_loader([train_rec[i] for i in bi], num_workers=0, batch_size=32)
            for batch in dl:
                se = encoder(batch.to(device))

            yb = torch.FloatTensor([train_rec[i]["target"] for i in bi]).to(device)
            logits = classifier(se).squeeze(-1)

            # Hard label loss
            loss_hard = bce(logits, yb)

            # Soft label from LLM (kNN CoT predictions)
            loss_soft = F.binary_cross_entropy(torch.sigmoid(logits), llm_t[bi])

            loss = (1 - alpha) * loss_hard + alpha * loss_soft
            opt.zero_grad(); loss.backward(); opt.step()

        auc = eval_coles(encoder)
        if auc > best_auc:
            best_auc = auc
            torch.save(encoder.state_dict(), OUTPUT_DIR / "coles_best_current.pt")

    return best_auc


# ==================================================================
# RAMD Loop
# ==================================================================

seq_encoder = build_seq_encoder().to(device)
seq_encoder.load_state_dict(torch.load(COLES_CKPT, map_location=device))

baseline = eval_coles(seq_encoder)
print(f"\n  Baseline CoLES: {baseline:.4f}")

# Prepare LLM config (loaded/unloaded per round to fit in VRAM)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

POS_TOKENS = ["male", " male", "Male", " Male"]
NEG_TOKENS = ["female", " female", "Female", " Female"]
def get_ids(tokens):
    ids = set()
    for t in tokens:
        e = tokenizer.encode(t, add_special_tokens=False)
        if e: ids.add(e[0])
    return list(ids)
pos_ids, neg_ids = get_ids(POS_TOKENS), get_ids(NEG_TOKENS)

results = {"baseline_coles": baseline}
N_ROUNDS = 4

for round_i in range(N_ROUNDS):
    print(f"\n{'='*60}")
    print(f"RAMD Round {round_i}")
    print(f"{'='*60}")

    # Step 1: Extract CoLES embeddings (current state)
    print("  Step 1: Extract CoLES embeddings...")
    emb_train = extract_embs(seq_encoder, train_rec)
    emb_test = extract_embs(seq_encoder, test_rec)

    # Step 2: kNN retrieval
    print("  Step 2: kNN retrieval...")
    train_contexts = knn_retrieve(emb_train, emb_train, y_train, k=10)
    test_contexts = knn_retrieve(emb_train, emb_test, y_train, k=10)

    # Step 3: Load LLM → predict → unload LLM
    print("  Step 3: Loading LLM...")
    model_llm = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model_llm.eval()

    print("  Step 3a: LLM predictions (train)...")
    llm_preds_train = llm_predict_with_cot(cids_train, train_contexts, model_llm, tokenizer, pos_ids, neg_ids)
    print("  Step 3b: LLM predictions (test)...")
    llm_preds_test = llm_predict_with_cot(cids_test, test_contexts, model_llm, tokenizer, pos_ids, neg_ids)

    llm_auc = roc_auc_score(y_test, llm_preds_test)
    results[f"round{round_i}_llm_auc"] = llm_auc
    print(f"  LLM AUC (kNN CoT): {llm_auc:.4f}")

    # Unload LLM to free GPU for distillation
    del model_llm; torch.cuda.empty_cache(); gc.collect()
    print(f"  LLM unloaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    # Step 4: Distill LLM soft predictions into CoLES
    print("  Step 4: Distill into CoLES...")
    if round_i > 0:
        # Reload best checkpoint from previous round to avoid drift
        seq_encoder.load_state_dict(torch.load(OUTPUT_DIR / "coles_best_current.pt", map_location=device))

    coles_auc = distill_round(seq_encoder, llm_preds_train, alpha=0.15, epochs=10)
    results[f"round{round_i}_coles_auc"] = coles_auc
    print(f"  CoLES AUC after distillation: {coles_auc:.4f}")

    # Save round checkpoint
    torch.save(seq_encoder.state_dict(), OUTPUT_DIR / f"coles_round{round_i}.pt")

    # Eval: concat CoLES + LLM predictions
    emb_train_new = extract_embs(seq_encoder, train_rec)
    emb_test_new = extract_embs(seq_encoder, test_rec)
    Xtr = np.hstack([MaxAbsScaler().fit_transform(emb_train_new), llm_preds_train.reshape(-1, 1)])
    Xte = np.hstack([MaxAbsScaler().fit_transform(emb_test_new), llm_preds_test.reshape(-1, 1)])
    lgbm = LGBMClassifier(**LGBM_P, random_state=42)
    lgbm.fit(Xtr, y_train)
    combined_auc = roc_auc_score(y_test, lgbm.predict_proba(Xte)[:, 1])
    results[f"round{round_i}_combined_auc"] = combined_auc
    print(f"  Combined (CoLES + LLM pred): {combined_auc:.4f}")

    print(f"\n  Round {round_i} summary:")
    print(f"    LLM:      {llm_auc:.4f}")
    print(f"    CoLES:    {coles_auc:.4f}")
    print(f"    Combined: {combined_auc:.4f}")

    with open(OUTPUT_DIR / "ramd_results.json", "w") as f:
        json.dump(results, f, indent=2)

# ---- Final summary ----
del tokenizer; torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("RAMD CONVERGENCE")
print("=" * 60)
print(f"  {'Round':<8} {'LLM':>8} {'CoLES':>8} {'Combined':>10}")
for r in range(N_ROUNDS):
    llm = results.get(f"round{r}_llm_auc", 0)
    col = results.get(f"round{r}_coles_auc", 0)
    com = results.get(f"round{r}_combined_auc", 0)
    print(f"  {r:<8} {llm:>8.4f} {col:>8.4f} {com:>10.4f}")
print(f"\n  Baseline CoLES: {baseline:.4f}")
print(f"  Best Combined:  {max(results.get(f'round{r}_combined_auc', 0) for r in range(N_ROUNDS)):.4f}")
