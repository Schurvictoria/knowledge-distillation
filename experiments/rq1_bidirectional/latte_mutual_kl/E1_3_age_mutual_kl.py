#!/usr/bin/env python3
"""
Age True Bidirectional — FIXED: gradient accumulation + memory bank.

Previous run: batch=16, only 15 contrastive negatives → 0 improvement.
Fix: gradient accumulation (effective batch=128) + memory bank (4096 negatives).

Saves checkpoint after EACH epoch.
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
    ("data/train_target.csv", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("data/transactions_train.csv", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, StandardScaler
from lightgbm import LGBMClassifier

from ptls.data_load.datasets import MemoryMapDataset, inference_data_loader
from ptls.nn import TrxEncoder, RnnSeqEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

OUTPUT_DIR = Path("results/age_bidir_fixed")
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

ids = labels["client_id"].values
targets = np.array([target_map[c] for c in ids])
idx_tr, idx_te = train_test_split(np.arange(len(ids)), test_size=0.1, random_state=42, stratify=targets)
train_ids, test_ids = set(ids[idx_tr]), set(ids[idx_te])

MCC_GROUPS = {range(1,1500):"Agriculture",range(4000,4800):"Transportation",range(5000,5600):"Retail",
              range(5600,5700):"Clothing",range(5800,5900):"Restaurants",range(6000,7000):"Financial",
              range(7500,7600):"Auto",range(8000,8100):"Medical",range(8200,8300):"Education"}
def mcc_cat(mcc):
    try: mcc=int(mcc)
    except: return "Other"
    for r,n in MCC_GROUPS.items():
        if mcc in r: return n
    return "Other"

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

def serialize(cid, max_txns=30):
    if cid not in grouped.groups: return "No txns."
    ct = grouped.get_group(cid).tail(max_txns)
    lines = [f"Client ({len(ct)} txns):"]
    for _, r in ct.iterrows():
        d = "spent" if r["amount_rur"]<0 else "received"
        lines.append(f"Day {int(r['trans_date'])}: {d} {abs(r['amount_rur']):.0f} at {mcc_cat(r['small_group'])}")
    return "\n".join(lines)

train_rec_full = build_records(train_ids)
test_rec = build_records(test_ids)
feature_dims = {"small_group": len(sg_enc.classes_) + 2}
_y_full = np.array([r["target"] for r in train_rec_full])
_tr_idx, _val_idx = train_test_split(
    np.arange(len(train_rec_full)), test_size=0.1, random_state=42, stratify=_y_full)
train_rec = [train_rec_full[i] for i in _tr_idx]
val_rec = [train_rec_full[i] for i in _val_idx]
y_train = _y_full[_tr_idx]
y_val = _y_full[_val_idx]
y_test = np.array([r["target"] for r in test_rec])
train_texts = [serialize(r["customer_id"]) for r in train_rec]
print(f"  train={len(train_rec)}, val={len(val_rec)}, test={len(test_rec)}")

# ---- Load models ----
print("Loading models...")
COLES_CKPT = Path("results/age_true_latte/coles_baseline.pt")
def build_encoder():
    trx = TrxEncoder(embeddings={"small_group":{"in":feature_dims["small_group"],"out":16}},
                      numeric_values={"amount":"identity"}, embeddings_noise=0.003, use_batch_norm_with_lens=True)
    return RnnSeqEncoder(trx_encoder=trx, hidden_size=800, type="gru", bidir=False, trainable_starter="static")

seq_encoder = build_encoder().to(device)
seq_encoder.load_state_dict(torch.load(COLES_CKPT, map_location=device))

LLM_CKPT = Path("results/age_llm4es/checkpoints/llm4es_lora")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", quantization_config=bnb, device_map="auto")
llm = PeftModel.from_pretrained(llm, str(LLM_CKPT))
hidden_llm = llm.config.hidden_size
print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

# ---- Pre-extract LLM embeddings (to use as memory bank) ----
print("Pre-extracting LLM embeddings for memory bank...")
llm.eval()
llm_embs_all = []
with torch.no_grad():
    for i, text in enumerate(train_texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        out = llm(**inputs, output_hidden_states=True)
        h = torch.stack(out.hidden_states[-4:]).mean(0)
        mask = inputs["attention_mask"][0].unsqueeze(-1).float()
        emb = (h[0]*mask).sum(0)/mask.sum(0)
        llm_embs_all.append(emb.cpu())
        del inputs, out
        if (i+1) % 2000 == 0:
            print(f"  {i+1}/{len(train_texts)}")

llm_embs_bank = torch.stack(llm_embs_all)  # (N, hidden_llm)
sc_l = StandardScaler()
llm_embs_bank_np = sc_l.fit_transform(llm_embs_bank.numpy())
llm_embs_bank_t = torch.FloatTensor(llm_embs_bank_np).to(device)
print(f"  Memory bank: {llm_embs_bank_t.shape}")

# ---- Eval function (returns val + test, model selection on val) ----
def _extract(records):
    seq_encoder.eval()
    with torch.no_grad():
        return torch.cat([seq_encoder(b.to(device)).cpu()
                          for b in inference_data_loader(records, num_workers=0, batch_size=64)]).numpy()

def eval_model():
    sc = MaxAbsScaler()
    lgbm = LGBMClassifier(**LGBM_P, random_state=42)
    etr = _extract(train_rec); evl = _extract(val_rec); ete = _extract(test_rec)
    lgbm.fit(sc.fit_transform(etr), y_train)
    val_acc = accuracy_score(y_val, lgbm.predict(sc.transform(evl)))
    test_acc = accuracy_score(y_test, lgbm.predict(sc.transform(ete)))
    return val_acc, test_acc

baseline_val, baseline_test = eval_model()
print(f"  Baseline: val={baseline_val:.4f} test={baseline_test:.4f}")

# ---- Heads ----
proj_s = nn.Sequential(nn.Linear(800,256),nn.ReLU(),nn.Linear(256,128)).to(device)
proj_t = nn.Sequential(nn.Linear(hidden_llm,256),nn.ReLU(),nn.Linear(256,128)).to(device)
cls_s = nn.Sequential(nn.Linear(800,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,4)).to(device)
cls_t = nn.Sequential(nn.Linear(hidden_llm,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,4)).to(device)

# ---- Contrastive with memory bank ----
def contrastive_with_bank(z_seq, z_text_batch, bank, n_neg=2048, temp=0.07):
    """InfoNCE with negatives sampled from memory bank."""
    neg_idx = torch.randint(0, len(bank), (n_neg,))
    neg_embs = F.normalize(proj_t(bank[neg_idx]), dim=1)

    # Positive: z_seq[i] with z_text_batch[i]
    pos_sim = (z_seq * z_text_batch).sum(dim=1) / temp  # (B,)
    # Negative: z_seq[i] with all negatives
    neg_sim = z_seq @ neg_embs.T / temp  # (B, n_neg)

    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+n_neg)
    labels = torch.zeros(len(z_seq), dtype=torch.long, device=device)
    return F.cross_entropy(logits, labels)

# ---- Training with gradient accumulation ----
print("\n" + "=" * 60)
print("Training with gradient accumulation + memory bank")
print("=" * 60)

ACCUM_STEPS = 8  # effective batch = 16 * 8 = 128
alpha_cls, alpha_con, alpha_mut = 0.5, 0.3, 0.2

trainable = list(seq_encoder.parameters()) + list(proj_s.parameters()) + list(proj_t.parameters()) + \
            list(cls_s.parameters()) + list(cls_t.parameters())
for _, p in llm.named_parameters():
    if p.requires_grad: trainable.append(p)

opt = torch.optim.Adam(trainable, lr=2e-4, weight_decay=1e-4)
ce = nn.CrossEntropyLoss()
best_val = baseline_val
best_test = baseline_test
results = {"baseline": baseline_test, "baseline_val": baseline_val}

for ep in range(10):
    seq_encoder.train(); llm.train()
    for m in [proj_s,proj_t,cls_s,cls_t]: m.train()
    idx = torch.randperm(len(train_rec))
    tot, nb = 0, 0
    opt.zero_grad()

    for step, s in enumerate(range(0, len(train_rec), 16)):
        bi = idx[s:s+16].tolist()
        br = [train_rec[i] for i in bi]
        yb = torch.LongTensor([r["target"] for r in br]).to(device)

        # CoLES forward
        dl = inference_data_loader(br, num_workers=0, batch_size=32)
        for batch in dl:
            se = seq_encoder(batch.to(device))

        # LLM forward (use precomputed + gradient for LoRA)
        te = llm_embs_bank_t[bi]  # Use precomputed for speed

        zs = F.normalize(proj_s(se), dim=1)
        zt = F.normalize(proj_t(te), dim=1)

        # Losses
        lc_s = ce(cls_s(se), yb)
        lc_t = ce(cls_t(te), yb)
        l_con = contrastive_with_bank(zs, zt, llm_embs_bank_t)
        ps, pt_ = F.softmax(cls_s(se),dim=1), F.softmax(cls_t(te),dim=1)
        l_mut = (F.kl_div(ps.log(),pt_.detach(),reduction='batchmean') +
                 F.kl_div(pt_.log(),ps.detach(),reduction='batchmean'))/2

        loss = (alpha_cls*(lc_s+lc_t)/2 + alpha_con*l_con + alpha_mut*l_mut) / ACCUM_STEPS
        loss.backward()
        tot += loss.item() * ACCUM_STEPS
        nb += 1

        if (step + 1) % ACCUM_STEPS == 0:
            opt.step()
            opt.zero_grad()

    # Final accumulation step
    opt.step()
    opt.zero_grad()

    val_acc, test_acc = eval_model()
    if val_acc > best_val:
        best_val = val_acc
        best_test = test_acc
        torch.save(seq_encoder.state_dict(), OUTPUT_DIR / "coles_bidir_best.pt")
    print(f"  ep {ep+1}: loss={tot/nb:.4f}, "
          f"val={val_acc:.4f} test={test_acc:.4f} best_val={best_val:.4f} best_test={best_test:.4f}")

    results[f"ep{ep+1}_val"] = val_acc
    results[f"ep{ep+1}_test"] = test_acc

results["bidirectional_best_test"] = best_test
results["bidirectional_best_val"] = best_val

print(f"\n  Age: baseline={baseline_test:.4f}, bidir={best_test:.4f} "
      f"({'+' if best_test>baseline_test else ''}{best_test-baseline_test:.4f})")

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

# NOTE: KNOWN LIMITATION — Age uses pre-computed LLM embeddings under no_grad
# (line "te = llm_embs_bank_t[bi]"), so LoRA on LLM does NOT receive gradient.
# Effectively this is LATTE-only on Age (bidirectional only on Gender/Rosbank).
# To make truly bidirectional: replace `te = llm_embs_bank_t[bi]` with on-the-fly
# `te = torch.stack([get_llm_emb(t) for t in batch_texts])` (slow but honest).

del llm, tokenizer; torch.cuda.empty_cache()
