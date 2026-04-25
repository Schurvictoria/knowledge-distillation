#!/usr/bin/env python3
"""
Phase 3: LLM4ES fine-tuning on Rosbank + TAID/DA-KD distillation.
Full pipeline: fine-tune Qwen2.5-3B → extract embeddings → adaptive distillation.
Saves all checkpoints.
"""

import time, json, warnings, gc, os
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from lightgbm import LGBMClassifier
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from datasets import Dataset

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
    ("data/rosbank_train.csv", "experiments/rq1_bidirectional/coles/run_rosbank_coles.py"),
    ("embeddings/rosbank/cids_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_rosbank_coles.py"),
    ("embeddings/rosbank/cids_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_rosbank_coles.py"),
    ("embeddings/rosbank/emb_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_rosbank_coles.py"),
    ("embeddings/rosbank/emb_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_rosbank_coles.py"),
    ("embeddings/rosbank/y_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_rosbank_coles.py"),
    ("embeddings/rosbank/y_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_rosbank_coles.py"),
]
for _p, _hint in _required_inputs:
    assert _P(_p).exists(), f"\n  Missing input: {_p}\n  Run prerequisite: {_hint}"
# ---- end input check ----



OUTPUT_DIR = Path("results/rosbank_llm4es")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUTPUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")

LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.02, max_depth=6, subsample=0.5,
                   colsample_bytree=0.75, reg_alpha=1, reg_lambda=1,
                   min_child_samples=50, verbosity=-1)

MCC_GROUPS = {
    range(1, 1500): "Agriculture", range(1500, 3000): "Construction",
    range(3000, 3300): "Airlines", range(3300, 3500): "Car Rental",
    range(3500, 4000): "Hotels", range(4000, 4800): "Transportation",
    range(4800, 5000): "Utilities and Telecom", range(5000, 5600): "Retail Stores",
    range(5600, 5700): "Clothing Stores", range(5700, 5800): "Home Furnishing",
    range(5800, 5900): "Restaurants and Food", range(5900, 6000): "Pharmacies",
    range(6000, 7000): "Financial Services", range(7000, 7300): "Personal Services",
    range(7300, 7500): "Business Services", range(7500, 7600): "Auto Services",
    range(7600, 7700): "Repair Services", range(7700, 7800): "Entertainment",
    range(7800, 8000): "Recreation", range(8000, 8100): "Medical Services",
}

def mcc_cat(mcc):
    try:
        mcc = int(mcc)
    except (ValueError, TypeError):
        return "Other"
    for r, name in MCC_GROUPS.items():
        if mcc in r:
            return name
    return "Other"

# ---- Load data ----
print("=" * 60)
print("STEP 1: Prepare Rosbank transaction texts")
print("=" * 60)

df = pd.read_csv(DATA_DIR / "rosbank_train.csv")
df["dt"] = pd.to_datetime(df["TRDATETIME"], format="%d%b%y:%H:%M:%S")
df = df.sort_values(["cl_id", "dt"])
df["mcc_desc"] = df["MCC"].fillna(0).astype(int).apply(mcc_cat)
df["amount_log"] = np.sign(df["amount"]) * np.log1p(np.abs(df["amount"]))

cids_train = np.load("embeddings/rosbank/cids_train_seed42.npy")
cids_test = np.load("embeddings/rosbank/cids_test_seed42.npy")
y_train = np.load("embeddings/rosbank/y_train_seed42.npy")
y_test = np.load("embeddings/rosbank/y_test_seed42.npy")
all_cids = np.concatenate([cids_train, cids_test])

grouped = df.groupby("cl_id")

def serialize_client(cid, max_txns=50):
    if cid not in grouped.groups:
        return "No transactions."
    ct = grouped.get_group(cid)
    if len(ct) > max_txns:
        ct = ct.tail(max_txns)
    lines = [f"Bank client transaction history ({len(ct)} transactions):"]
    for _, row in ct.iterrows():
        direction = "spent" if row["amount"] < 0 else "received"
        lines.append(f"{row['dt'].strftime('%Y-%m-%d')}: {direction} {abs(row['amount']):.0f} at {row['mcc_desc']}")
    return "\n".join(lines)

print("Serializing clients...")
client_texts = {cid: serialize_client(cid) for cid in all_cids}
print(f"  {len(client_texts)} clients, avg len: {np.mean([len(t) for t in client_texts.values()]):.0f} chars")

# ---- Fine-tune ----
print("\n" + "=" * 60)
print("STEP 2: Fine-tune Qwen2.5-3B (QLoRA)")
print("=" * 60)

MODEL_ID = "Qwen/Qwen2.5-3B"
LORA_CKPT = CKPT_DIR / "llm4es_lora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if (LORA_CKPT / "adapter_model.safetensors").exists():
    print(f"  Checkpoint found, loading LoRA from {LORA_CKPT}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    model = PeftModel.from_pretrained(model, str(LORA_CKPT))
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                              lora_dropout=0.05, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_texts = [client_texts[cid] for cid in cids_train]
    ds = Dataset.from_dict({"text": train_texts})
    ds_tok = ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=512, padding=False),
                    batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=str(CKPT_DIR / "ft_runs"), num_train_epochs=3,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, warmup_steps=50, logging_steps=50,
        save_strategy="epoch", bf16=True, report_to="none", dataloader_num_workers=0)

    trainer = Trainer(model=model, args=args, train_dataset=ds_tok,
                      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    print("Fine-tuning...")
    trainer.train()
    model.save_pretrained(str(LORA_CKPT))
    tokenizer.save_pretrained(str(LORA_CKPT))
    print(f"  Saved checkpoint to {LORA_CKPT}")
    del trainer; torch.cuda.empty_cache(); gc.collect()

model.eval()
print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

# ---- Extract embeddings ----
print("\n" + "=" * 60)
print("STEP 3: Extract embeddings")
print("=" * 60)

EMB_PATH = OUTPUT_DIR / "llm4es_embeddings.npz"
if EMB_PATH.exists():
    llm_all = np.load(EMB_PATH)["embeddings"].astype(np.float32)
    print(f"  Loaded cached: {llm_all.shape}")
else:
    hidden_size = model.config.hidden_size
    llm_all = np.zeros((len(all_cids), hidden_size), dtype=np.float16)
    print(f"  Extracting for {len(all_cids)} clients (dim={hidden_size})...")
    for i, cid in enumerate(all_cids):
        inputs = tokenizer(client_texts[cid], return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            hidden = torch.stack(out.hidden_states[-8:]).mean(0)
            mask = inputs["attention_mask"][0].unsqueeze(-1).float()
            pooled = (hidden[0] * mask).sum(0) / mask.sum(0)
            llm_all[i] = pooled.cpu().float().numpy().astype(np.float16)
        if (i+1) % 500 == 0:
            print(f"    {i+1}/{len(all_cids)}")
    np.savez_compressed(EMB_PATH, embeddings=llm_all, cid_order=all_cids)
    print(f"  Saved: {llm_all.shape}")
    llm_all = llm_all.astype(np.float32)

del model, tokenizer; torch.cuda.empty_cache(); gc.collect()

n_train = len(cids_train)
llm_train, llm_test = llm_all[:n_train], llm_all[n_train:]

# ---- Distillation experiments ----
print("\n" + "=" * 60)
print("STEP 4: TAID + DA-KD Distillation")
print("=" * 60)

coles_train = np.load("embeddings/rosbank/emb_train_seed42.npy")
coles_test = np.load("embeddings/rosbank/emb_test_seed42.npy")

# Baseline
scaler = MaxAbsScaler()
Xtr = scaler.fit_transform(coles_train)
Xte = scaler.transform(coles_test)
lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=42)
lgbm.fit(Xtr, y_train)
baseline = roc_auc_score(y_test, lgbm.predict_proba(Xte)[:, 1])
print(f"  Baseline CoLES: AUC = {baseline:.4f}")

# LLM4ES only
sc_l = MaxAbsScaler()
lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=42)
lgbm.fit(sc_l.fit_transform(llm_train), y_train)
llm_only = roc_auc_score(y_test, lgbm.predict_proba(sc_l.transform(llm_test))[:, 1])
print(f"  LLM4ES only: AUC = {llm_only:.4f}")

# Concat
lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=42)
Xtr_c = np.hstack([Xtr, sc_l.fit_transform(llm_train)])
Xte_c = np.hstack([Xte, sc_l.transform(llm_test)])
lgbm.fit(Xtr_c, y_train)
concat_auc = roc_auc_score(y_test, lgbm.predict_proba(Xte_c)[:, 1])
print(f"  CoLES + LLM4ES concat: AUC = {concat_auc:.4f}")

results = {"baseline_coles": baseline, "llm4es_only": llm_only, "coles_llm4es_concat": concat_auc}

class Adapter(nn.Module):
    def __init__(self, cd, ld, proj=128):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(cd,512), nn.GELU(), nn.Dropout(0.2),
                                      nn.Linear(512,512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512,cd))
        self.proj_s = nn.Sequential(nn.Linear(cd,256), nn.ReLU(), nn.Linear(256,proj))
        self.proj_t = nn.Sequential(nn.Linear(ld,256), nn.ReLU(), nn.Linear(256,proj))
        self.cls = nn.Sequential(nn.Linear(cd,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,1))
    def forward(self, c, l=None):
        a = c + self.adapter(c)
        zs = F.normalize(self.proj_s(a), dim=1)
        zt = F.normalize(self.proj_t(l), dim=1) if l is not None else None
        return a, zs, zt, self.cls(a).squeeze(-1)

def infonce(za, zb, t=0.07):
    lo = za @ zb.T / t
    la = torch.arange(len(za), device=za.device)
    return (F.cross_entropy(lo,la) + F.cross_entropy(lo.T,la)) / 2

def weighted_infonce(za, zb, w, t=0.07):
    lo = za @ zb.T / t
    la = torch.arange(len(za), device=za.device)
    return (F.cross_entropy(lo,la,reduction='none')*w).mean() + (F.cross_entropy(lo.T,la,reduction='none')*w).mean()

def dakd_weights(c, l):
    """Compute per-sample difficulty weights. Projects to same dim if needed."""
    with torch.no_grad():
        # Project to same dimension for distance computation
        c_norm = F.normalize(c, dim=1)
        if c.shape[1] != l.shape[1]:
            # Use first min(c,l) dims for distance
            dim = min(c.shape[1], l.shape[1])
            c_norm = F.normalize(c[:, :dim], dim=1)
            l_norm = F.normalize(l[:, :dim], dim=1)
        else:
            l_norm = F.normalize(l, dim=1)
        d = torch.norm(c_norm - l_norm, dim=1)
        w = (d - d.min()) / (d.max() - d.min() + 1e-8)
        return torch.softmax(w, dim=0) * len(w)

def taid_alpha(ep, mx, tgt, warm=0.2):
    w = int(mx * warm)
    return tgt * min(ep / w, 1.0) if w > 0 else tgt

experiments = [
    ("fixed_alpha0.5",      0.5, False, False),
    ("taid_alpha0.5",        0.5, True,  False),
    ("dakd_alpha0.5",        0.5, False, True),
    ("taid_dakd_alpha0.5",   0.5, True,  True),
    ("taid_dakd_alpha0.3",   0.3, True,  True),
    ("taid_dakd_alpha0.7",   0.7, True,  True),
]

for name, a_tgt, use_taid, use_dakd in experiments:
    print(f"\n--- {name} ---")
    sc_c, sc_l2 = StandardScaler(), StandardScaler()
    Xc = torch.FloatTensor(sc_c.fit_transform(coles_train)).to(device)
    Xc_te = torch.FloatTensor(sc_c.transform(coles_test)).to(device)
    Xl = torch.FloatTensor(sc_l2.fit_transform(llm_train)).to(device)
    yt = torch.FloatTensor(y_train).to(device)

    m = Adapter(coles_train.shape[1], llm_train.shape[1]).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 200)
    bce = nn.BCEWithLogitsLoss()
    best = 0

    for ep in range(200):
        m.train()
        alpha = taid_alpha(ep, 200, a_tgt) if use_taid else a_tgt
        idx = torch.randperm(len(Xc))
        for s in range(0, len(Xc), 256):
            b = idx[s:s+256]
            _, zs, zt, lo = m(Xc[b], Xl[b])
            lc = bce(lo, yt[b])
            if use_dakd:
                w = dakd_weights(Xc[b], Xl[b])
                la = weighted_infonce(zs, zt, w)
            else:
                la = infonce(zs, zt)
            ((1-alpha)*lc + alpha*la).backward()
            opt.step(); opt.zero_grad()
        sch.step()

        if (ep+1) % 50 == 0:
            m.eval()
            with torch.no_grad():
                at, _, _, _ = m(Xc)
                ae, _, _, _ = m(Xc_te)
            sc_a = MaxAbsScaler()
            lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=42)
            lgbm.fit(sc_a.fit_transform(at.cpu().numpy()), y_train)
            auc = roc_auc_score(y_test, lgbm.predict_proba(sc_a.transform(ae.cpu().numpy()))[:, 1])
            if auc > best: best = auc
            print(f"  {name} ep {ep+1}: AUC={auc:.4f} (best={best:.4f})")

    results[name] = best
    torch.save(m.state_dict(), OUTPUT_DIR / f"{name}_adapter.pt")
    del m; torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("ROSBANK RESULTS")
print("=" * 60)
for n, a in sorted(results.items(), key=lambda x: -x[1]):
    d = a - baseline
    print(f"  {n:<25} AUC={a:.4f}  ({'+' if d>=0 else ''}{d:.4f})")

with open(OUTPUT_DIR / "rosbank_llm4es_results.json", "w") as f:
    json.dump(results, f, indent=2)
pd.DataFrame([{"method":k,"auc":v} for k,v in results.items()]).to_csv(
    OUTPUT_DIR / "rosbank_llm4es_results.csv", index=False)
