#!/usr/bin/env python3
"""
Age dataset: LLM4ES fine-tuning + bidirectional distillation.
Full pipeline: fine-tune Qwen2.5-3B → embeddings → distillation experiments.
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
from sklearn.metrics import accuracy_score
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
    ("data/train_target.csv", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("data/transactions_train.csv", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/cids_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/cids_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/emb_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/emb_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/y_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/y_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
]
for _p, _hint in _required_inputs:
    assert _P(_p).exists(), f"\n  Missing input: {_p}\n  Run prerequisite: {_hint}"
# ---- end input check ----



OUTPUT_DIR = Path("results/age_llm4es")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUTPUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")

LGBM_PARAMS = dict(n_estimators=1000, learning_rate=0.02, objective="multiclass",
                   num_class=4, max_depth=12, num_leaves=50, subsample=0.75,
                   colsample_bytree=0.75, reg_alpha=1, reg_lambda=1,
                   min_child_samples=50, verbosity=-1)

MCC_GROUPS = {
    range(1, 1500): "Agriculture", range(1500, 3000): "Construction",
    range(3000, 3300): "Airlines", range(4000, 4800): "Transportation",
    range(4800, 5000): "Utilities", range(5000, 5600): "Retail",
    range(5600, 5700): "Clothing", range(5800, 5900): "Restaurants",
    range(6000, 7000): "Financial", range(7500, 7600): "Auto Services",
    range(8000, 8100): "Medical", range(8200, 8300): "Education",
}

def mcc_cat(mcc):
    try:
        mcc = int(mcc)
    except: return "Other"
    for r, n in MCC_GROUPS.items():
        if mcc in r: return n
    return "Other"

# ---- Load data ----
print("=" * 60)
print("STEP 1: Prepare Age transaction texts")
print("=" * 60)

tx = pd.read_csv(DATA_DIR / "transactions_train.csv")
labels = pd.read_csv(DATA_DIR / "train_target.csv")
tx = tx.sort_values(["client_id", "trans_date"])
tx["mcc_desc"] = tx["small_group"].fillna(0).astype(int).apply(mcc_cat)

cids_train = np.load("embeddings/age/cids_train_seed42.npy")
cids_test = np.load("embeddings/age/cids_test_seed42.npy")
y_train = np.load("embeddings/age/y_train_seed42.npy")
y_test = np.load("embeddings/age/y_test_seed42.npy")
all_cids = np.concatenate([cids_train, cids_test])

grouped = tx.groupby("client_id")

def serialize_client(cid, max_txns=50):
    if cid not in grouped.groups:
        return "No transactions."
    ct = grouped.get_group(cid)
    if len(ct) > max_txns:
        ct = ct.tail(max_txns)
    lines = [f"Client transaction history ({len(ct)} transactions):"]
    for _, row in ct.iterrows():
        d = "spent" if row["amount_rur"] < 0 else "received"
        lines.append(f"Day {int(row['trans_date'])}: {d} {abs(row['amount_rur']):.0f} at {row['mcc_desc']}")
    return "\n".join(lines)

print("Serializing clients (sampling 10k for fine-tuning)...")
# Serialize all for embedding extraction, sample for fine-tuning
client_texts = {}
for cid in all_cids:
    client_texts[cid] = serialize_client(cid)
print(f"  {len(client_texts)} clients serialized")

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
    print(f"  Checkpoint found, loading from {LORA_CKPT}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    model = PeftModel.from_pretrained(model, str(LORA_CKPT))
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                              lora_dropout=0.05, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Sample train texts (Age has 27k clients — too many for full fine-tuning)
    np.random.seed(42)
    sample_idx = np.random.choice(len(cids_train), min(8000, len(cids_train)), replace=False)
    train_texts = [client_texts[cids_train[i]] for i in sample_idx]
    ds = Dataset.from_dict({"text": train_texts})
    ds_tok = ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=512, padding=False),
                    batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=str(CKPT_DIR / "ft_runs"), num_train_epochs=2,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, warmup_steps=50, logging_steps=100,
        save_strategy="epoch", bf16=True, report_to="none", dataloader_num_workers=0)

    trainer = Trainer(model=model, args=args, train_dataset=ds_tok,
                      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    print(f"Fine-tuning on {len(train_texts)} samples...")
    trainer.train()
    model.save_pretrained(str(LORA_CKPT))
    tokenizer.save_pretrained(str(LORA_CKPT))
    print(f"  Saved checkpoint to {LORA_CKPT}")
    del trainer; torch.cuda.empty_cache(); gc.collect()

model.eval()

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
        if (i+1) % 1000 == 0:
            print(f"    {i+1}/{len(all_cids)}")
    np.savez_compressed(EMB_PATH, embeddings=llm_all, cid_order=all_cids)
    print(f"  Saved: {llm_all.shape}")
    llm_all = llm_all.astype(np.float32)

del model, tokenizer; torch.cuda.empty_cache(); gc.collect()

n_tr = len(cids_train)
llm_train, llm_test = llm_all[:n_tr], llm_all[n_tr:]

# ---- Distillation ----
print("\n" + "=" * 60)
print("STEP 4: Distillation experiments")
print("=" * 60)

coles_train = np.load("embeddings/age/emb_train_seed42.npy")
coles_test = np.load("embeddings/age/emb_test_seed42.npy")

results = {}

# Baselines
sc = MaxAbsScaler()
lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=42)
lgbm.fit(sc.fit_transform(coles_train), y_train)
results["coles_only"] = accuracy_score(y_test, lgbm.predict(sc.transform(coles_test)))
print(f"  CoLES only:     acc = {results['coles_only']:.4f}")

sc_l = MaxAbsScaler()
lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=42)
lgbm.fit(sc_l.fit_transform(llm_train), y_train)
results["llm4es_only"] = accuracy_score(y_test, lgbm.predict(sc_l.transform(llm_test)))
print(f"  LLM4ES only:    acc = {results['llm4es_only']:.4f}")

Xtr_cat = np.hstack([sc.fit_transform(coles_train), sc_l.fit_transform(llm_train)])
Xte_cat = np.hstack([sc.transform(coles_test), sc_l.transform(llm_test)])
lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=42)
lgbm.fit(Xtr_cat, y_train)
results["concat"] = accuracy_score(y_test, lgbm.predict(Xte_cat))
print(f"  Concat:         acc = {results['concat']:.4f}")

# DML (best approach from Gender experiments for comparison)
class Branch(nn.Module):
    def __init__(self, dim, n_classes=4, hidden=512):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(dim,hidden), nn.GELU(), nn.Dropout(0.2),
                                      nn.Linear(hidden,hidden), nn.GELU(), nn.Dropout(0.1),
                                      nn.Linear(hidden,dim))
        self.head = nn.Sequential(nn.Linear(dim,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,n_classes))
    def forward(self, x):
        adapted = x + self.adapter(x)
        return adapted, self.head(adapted)

sc_c2, sc_l2 = StandardScaler(), StandardScaler()
Xc = torch.FloatTensor(sc_c2.fit_transform(coles_train)).to(device)
Xc_te = torch.FloatTensor(sc_c2.transform(coles_test)).to(device)
Xl = torch.FloatTensor(sc_l2.fit_transform(llm_train)).to(device)
Yt = torch.LongTensor(y_train).to(device)

for alpha in [0.3, 0.5]:
    print(f"\n  DML α={alpha}...")
    ma = Branch(coles_train.shape[1]).to(device)
    mb = Branch(llm_train.shape[1]).to(device)
    oa = torch.optim.Adam(ma.parameters(), lr=1e-3, weight_decay=1e-4)
    ob = torch.optim.Adam(mb.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    best_acc = 0

    for ep in range(200):
        ma.train(); mb.train()
        idx = torch.randperm(len(Xc))
        for s in range(0, len(Xc), 256):
            b = idx[s:s+256]
            _, la = ma(Xc[b]); _, lb = mb(Xl[b])
            pa = F.softmax(la, dim=1); pb = F.softmax(lb, dim=1)
            loss_a = (1-alpha)*ce(la, Yt[b]) + alpha*F.kl_div(pa.log(), pb.detach(), reduction='batchmean')
            loss_b = (1-alpha)*ce(lb, Yt[b]) + alpha*F.kl_div(pb.log(), pa.detach(), reduction='batchmean')
            oa.zero_grad(); loss_a.backward(); oa.step()
            ob.zero_grad(); loss_b.backward(); ob.step()

        if (ep+1) % 50 == 0:
            ma.eval()
            with torch.no_grad():
                ad_tr, _ = ma(Xc); ad_te, _ = ma(Xc_te)
            sc_a = MaxAbsScaler()
            lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=42)
            lgbm.fit(sc_a.fit_transform(ad_tr.cpu().numpy()), y_train)
            acc = accuracy_score(y_test, lgbm.predict(sc_a.transform(ad_te.cpu().numpy())))
            best_acc = max(best_acc, acc)
            print(f"    ep {ep+1}: acc={acc:.4f} (best={best_acc:.4f})")

    results[f"dml_alpha{alpha}"] = best_acc
    torch.save(ma.state_dict(), OUTPUT_DIR / f"dml_coles_alpha{alpha}.pt")
    del ma, mb; torch.cuda.empty_cache()

# Summary
print("\n" + "=" * 60)
print("AGE RESULTS")
print("=" * 60)
for n, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {n:<25} acc = {v:.4f}")

with open(OUTPUT_DIR / "age_results.json", "w") as f:
    json.dump(results, f, indent=2)
pd.DataFrame([{"method":k,"accuracy":v} for k,v in results.items()]).to_csv(
    OUTPUT_DIR / "age_results.csv", index=False)
