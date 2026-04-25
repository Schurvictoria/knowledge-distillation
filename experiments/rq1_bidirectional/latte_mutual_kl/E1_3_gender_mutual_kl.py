#!/usr/bin/env python3
"""
TRUE Bidirectional Distillation: fine-tune BOTH CoLES GRU AND LLM (LoRA) simultaneously.

Both models process the same clients:
- CoLES: raw transactions → GRU → seq_embedding (1024d)
- LLM: serialized text → Qwen2.5-3B LoRA → text_embedding (2048d)

Joint training with:
- Classification loss (both models predict gender)
- Contrastive alignment (InfoNCE between seq and text embeddings)
- Mutual soft-label (each model matches the other's predictions)

At inference: only CoLES needed (no LLM). LLM knowledge distilled into GRU.

Saves checkpoints after each round.
"""

import time, json, warnings, gc, os
from pathlib import Path
from functools import partial

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

# ---- Required input files ----
from pathlib import Path as _P
_required_inputs = [
    ("data/gender_train.csv", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
    ("data/transactions.csv", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
]
for _p, _hint in _required_inputs:
    assert _P(_p).exists(), f"\n  Missing input: {_p}\n  Run prerequisite: {_hint}"
# ---- end input check ----

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from lightgbm import LGBMClassifier

from ptls.data_load.datasets import MemoryMapDataset, inference_data_loader
from ptls.nn import TrxEncoder, RnnSeqEncoder

OUTPUT_DIR = Path("results/gender_true_bidirectional")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LGBM_P = dict(n_estimators=500, learning_rate=0.02, max_depth=6, subsample=0.5,
              colsample_bytree=0.75, reg_alpha=1, reg_lambda=1, min_child_samples=50, verbosity=-1)

# ---- Load transaction data ----
print("=" * 60)
print("STEP 1: Load data")
print("=" * 60)

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

MCC_GROUPS = {
    range(1, 1500): "Agriculture", range(4000, 4800): "Transportation",
    range(5000, 5600): "Retail", range(5600, 5700): "Clothing",
    range(5800, 5900): "Restaurants", range(6000, 7000): "Financial",
    range(7500, 7600): "Auto", range(8000, 8100): "Medical",
}

def mcc_cat(mcc):
    try:
        mcc = int(mcc)
    except: return "Other"
    for r, n in MCC_GROUPS.items():
        if mcc in r: return n
    return "Other"

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

def serialize_client(cid, max_txns=30):
    if cid not in grouped.groups: return "No transactions."
    ct = grouped.get_group(cid)
    if len(ct) > max_txns: ct = ct.tail(max_txns)
    lines = [f"Client ({len(ct)} txns):"]
    for _, r in ct.iterrows():
        d = "spent" if r["amount"] < 0 else "received"
        lines.append(f"Day {int(r['day_float'])}: {d} {abs(r['amount']):.0f} at {mcc_cat(r['mcc_code'])}")
    return "\n".join(lines)

train_rec_full = build_records(train_ids)
test_rec = build_records(test_ids)
# Honest val split (10% from train, seed=42, stratified)
_y_full = np.array([r["target"] for r in train_rec_full])
_tr_idx, _val_idx = train_test_split(
    np.arange(len(train_rec_full)), test_size=0.1, random_state=42, stratify=_y_full)
train_rec = [train_rec_full[i] for i in _tr_idx]
val_rec = [train_rec_full[i] for i in _val_idx]
y_val = _y_full[_val_idx]
feature_dims = {col: len(enc.classes_) + 2 for col, enc in encoders.items()}
y_train = np.array([r["target"] for r in train_rec])
y_test = np.array([r["target"] for r in test_rec])

# Pre-serialize texts
train_texts = [serialize_client(r["customer_id"]) for r in train_rec]
test_texts = [serialize_client(r["customer_id"]) for r in test_rec]
print(f"  train={len(train_rec)}, test={len(test_rec)}")

# ---- Load both models ----
print("\n" + "=" * 60)
print("STEP 2: Load CoLES + LLM")
print("=" * 60)

# CoLES
COLES_CKPT = Path("results/gender_true_latte/coles_baseline.pt")
def build_seq_encoder():
    trx = TrxEncoder(
        embeddings={"mcc_code": {"in": feature_dims["mcc_code"], "out": 48},
                     "tr_type": {"in": feature_dims["tr_type"], "out": 24}},
        numeric_values={"amount": "identity"}, embeddings_noise=0.003, use_batch_norm_with_lens=True)
    return RnnSeqEncoder(trx_encoder=trx, hidden_size=1024, type="gru", bidir=False, trainable_starter="static")

seq_encoder = build_seq_encoder().to(device)
if COLES_CKPT.exists():
    seq_encoder.load_state_dict(torch.load(COLES_CKPT, map_location=device))
    print(f"  CoLES loaded from {COLES_CKPT}")
else:
    print("  ERROR: CoLES checkpoint not found!")
    exit(1)

# LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

LLM_CKPT = Path("results/gender_llm4es/checkpoints/llm4es_lora")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", quantization_config=bnb, device_map="auto")
if (LLM_CKPT / "adapter_model.safetensors").exists():
    llm = PeftModel.from_pretrained(llm, str(LLM_CKPT))
    print(f"  LLM loaded with LoRA from {LLM_CKPT}")
else:
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
                           target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])
    llm = get_peft_model(llm, lora_cfg)
    print("  LLM with fresh LoRA")

print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

# ---- Projection heads + classifiers ----
hidden_llm = llm.config.hidden_size  # 2048
proj_seq = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 128)).to(device)
proj_text = nn.Sequential(nn.Linear(hidden_llm, 256), nn.ReLU(), nn.Linear(256, 128)).to(device)
cls_seq = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)).to(device)
cls_text = nn.Sequential(nn.Linear(hidden_llm, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)).to(device)

# ---- Extract LLM embedding for a text ----
def get_llm_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=False).to(device)
    with torch.set_grad_enabled(llm.training):
        out = llm(**inputs, output_hidden_states=True)
        hidden = torch.stack(out.hidden_states[-4:]).mean(0)  # last 4 layers
        mask = inputs["attention_mask"][0].unsqueeze(-1).float()
        return (hidden[0] * mask).sum(0) / mask.sum(0)

# ---- Eval function: returns (val_auc, test_auc) — model selection on VAL ----
def _extract(records):
    seq_encoder.eval()
    dl = inference_data_loader(records, num_workers=0, batch_size=64)
    chunks = []
    with torch.no_grad():
        for batch in dl:
            chunks.append(seq_encoder(batch.to(device)).cpu())
    return torch.cat(chunks).numpy()

def eval_coles():
    sc = MaxAbsScaler()
    lgbm = LGBMClassifier(**LGBM_P, random_state=42)
    emb_tr = _extract(train_rec)
    emb_val = _extract(val_rec)
    emb_te = _extract(test_rec)
    lgbm.fit(sc.fit_transform(emb_tr), y_train)
    val_auc = roc_auc_score(y_val, lgbm.predict_proba(sc.transform(emb_val))[:, 1])
    test_auc = roc_auc_score(y_test, lgbm.predict_proba(sc.transform(emb_te))[:, 1])
    return val_auc, test_auc

# Baseline
baseline_val, baseline_test = eval_coles()
print(f"\n  Baseline CoLES: val={baseline_val:.4f} test={baseline_test:.4f}")

# ---- Bidirectional fine-tuning ----
print("\n" + "=" * 60)
print("STEP 3: True Bidirectional Fine-tuning")
print("=" * 60)

results = {"baseline_coles_val": baseline_val, "baseline_coles_test": baseline_test}

for alpha_cls, alpha_contrast, alpha_mutual in [(0.7, 0.2, 0.1), (0.5, 0.3, 0.2), (0.8, 0.1, 0.1)]:
    config_name = f"cls{alpha_cls}_con{alpha_contrast}_mut{alpha_mutual}"
    print(f"\n--- {config_name} ---")

    # Reset models
    seq_encoder.load_state_dict(torch.load(COLES_CKPT, map_location=device))
    # Reset heads
    for m in [proj_seq, proj_text, cls_seq, cls_text]:
        for p in m.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    # Trainable params: CoLES + projections + classifiers + LLM LoRA
    trainable = list(seq_encoder.parameters()) + list(proj_seq.parameters()) + \
                list(proj_text.parameters()) + list(cls_seq.parameters()) + list(cls_text.parameters())
    # Add LLM LoRA params
    for name, param in llm.named_parameters():
        if param.requires_grad:
            trainable.append(param)

    opt = torch.optim.Adam(trainable, lr=3e-4, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    best_val = baseline_val
    best_test = baseline_test

    for epoch in range(10):
        seq_encoder.train(); llm.train()
        proj_seq.train(); proj_text.train(); cls_seq.train(); cls_text.train()

        idx = torch.randperm(len(train_rec))
        total_loss = 0
        n_batches = 0

        for start in range(0, len(train_rec), 16):
            b_idx = idx[start:start+16].tolist()
            batch_records = [train_rec[i] for i in b_idx]
            batch_texts = [train_texts[i] for i in b_idx]
            y_batch = torch.FloatTensor([r["target"] for r in batch_records]).to(device)

            # CoLES forward
            dl = inference_data_loader(batch_records, num_workers=0, batch_size=32)
            for batch in dl:
                seq_emb = seq_encoder(batch.to(device))  # (B, 1024)

            # LLM forward (one by one due to variable length)
            text_embs = []
            for text in batch_texts:
                text_embs.append(get_llm_embedding(text))
            text_emb = torch.stack(text_embs)  # (B, 2048)

            # Projections
            z_seq = F.normalize(proj_seq(seq_emb), dim=1)
            z_text = F.normalize(proj_text(text_emb), dim=1)

            # Classification losses
            logits_seq = cls_seq(seq_emb).squeeze(-1)
            logits_text = cls_text(text_emb).squeeze(-1)
            loss_cls_seq = bce(logits_seq, y_batch)
            loss_cls_text = bce(logits_text, y_batch)

            # Contrastive alignment
            lo = z_seq @ z_text.T / 0.07
            la = torch.arange(len(z_seq), device=device)
            loss_contrast = (F.cross_entropy(lo, la) + F.cross_entropy(lo.T, la)) / 2

            # Mutual soft-label
            p_seq = torch.sigmoid(logits_seq)
            p_text = torch.sigmoid(logits_text)
            loss_mutual = (F.binary_cross_entropy(p_seq, p_text.detach()) +
                          F.binary_cross_entropy(p_text, p_seq.detach())) / 2

            # Total
            loss = alpha_cls * (loss_cls_seq + loss_cls_text) / 2 + \
                   alpha_contrast * loss_contrast + \
                   alpha_mutual * loss_mutual

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        # Eval CoLES — model selection on VAL only (no test peeking)
        val_auc, test_auc = eval_coles()
        if val_auc > best_val:
            best_val = val_auc
            best_test = test_auc
            torch.save(seq_encoder.state_dict(), OUTPUT_DIR / f"coles_bidir_{config_name}.pt")

        print(f"  ep {epoch+1}: loss={total_loss/n_batches:.4f}, "
              f"val={val_auc:.4f} test={test_auc:.4f} best_val={best_val:.4f} best_test={best_test:.4f}")

    results[config_name] = best_test
    torch.cuda.empty_cache(); gc.collect()

# ---- Summary ----
print("\n" + "=" * 60)
print("TRUE BIDIRECTIONAL SUMMARY")
print("=" * 60)
for n, v in sorted(results.items(), key=lambda x: -x[1]):
    d = v - baseline_test
    print(f"  {n:<35} AUC={v:.4f} ({'+' if d >= 0 else ''}{d:.4f})")

with open(OUTPUT_DIR / "true_bidir_results.json", "w") as f:
    json.dump(results, f, indent=2)

del llm, tokenizer; torch.cuda.empty_cache(); gc.collect()
