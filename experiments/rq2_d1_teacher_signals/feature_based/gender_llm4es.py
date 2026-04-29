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
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

import random as _random, os as _os
_SEED = 42
_random.seed(_SEED); np.random.seed(_SEED)
torch.manual_seed(_SEED); torch.cuda.manual_seed_all(_SEED)
import pytorch_lightning as _pl
_pl.seed_everything(_SEED, workers=True)
_os.environ["PYTHONHASHSEED"] = str(_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from pathlib import Path as _P
_required_inputs = [
    ("data/gender_train.csv", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
    ("data/transactions.csv", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
    ("embeddings/gender/cids_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
    ("embeddings/gender/cids_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
    ("embeddings/gender/emb_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
    ("embeddings/gender/emb_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
    ("embeddings/gender/y_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
    ("embeddings/gender/y_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
]
for _p, _hint in _required_inputs:
    assert _P(_p).exists(), f"\n  Missing input: {_p}\n  Run prerequisite: {_hint}"

OUTPUT_DIR = Path("results/gender_llm4es")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUTPUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    range(8100, 8200): "Legal Services", range(8200, 8300): "Education",
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

print("STEP 1: Prepare transaction texts")

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
tx["mcc_desc"] = tx["mcc_code"].apply(mcc_cat)

target_map = dict(zip(labels["customer_id"], labels["gender"]))
grouped = tx.groupby("customer_id")

cids_train = np.load("embeddings/gender/cids_train_seed42.npy")
cids_test = np.load("embeddings/gender/cids_test_seed42.npy")
y_train = np.load("embeddings/gender/y_train_seed42.npy")
y_test = np.load("embeddings/gender/y_test_seed42.npy")
all_cids = np.concatenate([cids_train, cids_test])

def serialize_client_llm4es(cid, max_txns=50):
    if cid not in grouped.groups:
        return "No transactions."
    ct = grouped.get_group(cid)
    if len(ct) > max_txns:
        ct = ct.tail(max_txns)

    lines = [f"Bank client transaction history ({len(ct)} transactions):"]
    for _, row in ct.iterrows():
        amt = row["amount"]
        direction = "spent" if amt < 0 else "received"
        lines.append(f"Day {int(row['day_float'])}: {direction} {abs(amt):.0f} at {row['mcc_desc']}")

    return "\n".join(lines)

print("Serializing clients...")
client_texts = {}
for cid in all_cids:
    client_texts[cid] = serialize_client_llm4es(cid)

avg_len = np.mean([len(t) for t in client_texts.values()])
print(f"  {len(client_texts)} clients, avg text len: {avg_len:.0f} chars")

print("STEP 2: Fine-tune LLaMA-3.2-3B (QLoRA)")

MODEL_ID = "Qwen/Qwen2.5-3B"
LORA_CKPT = CKPT_DIR / "llm4es_lora"

if (LORA_CKPT / "adapter_model.safetensors").exists():
    print(f"  Checkpoint found at {LORA_CKPT}, skipping fine-tuning")
    SKIP_FINETUNE = True
else:
    SKIP_FINETUNE = False

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config,
    device_map="auto",
)
print(f"  Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

if not SKIP_FINETUNE:
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_texts = [client_texts[cid] for cid in cids_train]
    ds = Dataset.from_dict({"text": train_texts})

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)

    ds_tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=str(CKPT_DIR / "ft_runs"),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=50,
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Fine-tuning...")
    trainer.train()

    model.save_pretrained(str(LORA_CKPT))
    tokenizer.save_pretrained(str(LORA_CKPT))
    print(f"  LoRA checkpoint saved to {LORA_CKPT}")

    del trainer
    torch.cuda.empty_cache()
    gc.collect()
else:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(LORA_CKPT))
    print(f"  LoRA weights loaded from {LORA_CKPT}")

model.eval()

print("STEP 3: Extract embeddings (mean pool last 8 layers)")

EMB_PATH = OUTPUT_DIR / "llm4es_embeddings.npz"

if EMB_PATH.exists():
    data = np.load(EMB_PATH)
    llm_emb_all = data["embeddings"]
    print(f"  Loaded cached embeddings: {llm_emb_all.shape}")
else:
    hidden_size = model.config.hidden_size
    n_layers_pool = 8
    llm_emb_all = np.zeros((len(all_cids), hidden_size), dtype=np.float16)

    print(f"  Extracting for {len(all_cids)} clients (dim={hidden_size}, pool last {n_layers_pool} layers)...")
    for i, cid in enumerate(all_cids):
        text = client_texts[cid]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states[-n_layers_pool:]
            stacked = torch.stack(hidden_states).mean(0)
            mask = inputs["attention_mask"][0].unsqueeze(-1).float()
            pooled = (stacked[0] * mask).sum(0) / mask.sum(0)
            llm_emb_all[i] = pooled.cpu().float().numpy().astype(np.float16)

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(all_cids)}")

    np.savez_compressed(EMB_PATH, embeddings=llm_emb_all, cid_order=all_cids)
    print(f"  Saved embeddings: {llm_emb_all.shape}")

n_train = len(cids_train)
llm_emb_train = llm_emb_all[:n_train].astype(np.float32)
llm_emb_test = llm_emb_all[n_train:].astype(np.float32)

del model, tokenizer
torch.cuda.empty_cache()
gc.collect()

print("STEP 4: Downstream evaluation")

coles_train = np.load("embeddings/gender/emb_train_seed42.npy")
coles_test = np.load("embeddings/gender/emb_test_seed42.npy")

results = {}

scaler = MaxAbsScaler()
Xtr = scaler.fit_transform(coles_train)
Xte = scaler.transform(coles_test)
lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=42)
lgbm.fit(Xtr, y_train)
p = lgbm.predict_proba(Xte)[:, 1]
results["coles_only"] = roc_auc_score(y_test, p)
print(f"  CoLES only:                AUC = {results['coles_only']:.4f}")

scaler_l = MaxAbsScaler()
Xtr_l = scaler_l.fit_transform(llm_emb_train)
Xte_l = scaler_l.transform(llm_emb_test)
lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=42)
lgbm.fit(Xtr_l, y_train)
p = lgbm.predict_proba(Xte_l)[:, 1]
results["llm4es_only"] = roc_auc_score(y_test, p)
print(f"  LLM4ES emb only:           AUC = {results['llm4es_only']:.4f}")

Xtr_c = np.hstack([Xtr, Xtr_l])
Xte_c = np.hstack([Xte, Xte_l])
lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=42)
lgbm.fit(Xtr_c, y_train)
p = lgbm.predict_proba(Xte_c)[:, 1]
results["coles_plus_llm4es"] = roc_auc_score(y_test, p)
print(f"  CoLES + LLM4ES:            AUC = {results['coles_plus_llm4es']:.4f}")

print("\n  Training contrastive adapter with LLM4ES embeddings...")

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, z_a, z_b):
        z_a = F.normalize(z_a, dim=1)
        z_b = F.normalize(z_b, dim=1)
        logits = z_a @ z_b.T / self.temperature
        labels = torch.arange(len(z_a), device=z_a.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

class Adapter(nn.Module):
    def __init__(self, coles_dim, text_dim, proj_dim=128):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(coles_dim, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, coles_dim),
        )
        self.proj_seq = nn.Sequential(nn.Linear(coles_dim, 256), nn.ReLU(), nn.Linear(256, proj_dim))
        self.proj_text = nn.Sequential(nn.Linear(text_dim, 256), nn.ReLU(), nn.Linear(256, proj_dim))
        self.classifier = nn.Sequential(nn.Linear(coles_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, coles, text=None):
        adapted = coles + self.adapter(coles)
        z_seq = self.proj_seq(adapted)
        z_text = self.proj_text(text) if text is not None else None
        logits = self.classifier(adapted).squeeze(-1)
        return adapted, z_seq, z_text, logits

for alpha in [0.1, 0.3, 0.5]:
    sc_c = StandardScaler()
    sc_t = StandardScaler()
    X_c = torch.FloatTensor(sc_c.fit_transform(coles_train)).to(device)
    X_c_te = torch.FloatTensor(sc_c.transform(coles_test)).to(device)
    X_t = torch.FloatTensor(sc_t.fit_transform(llm_emb_train)).to(device)
    y_t = torch.FloatTensor(y_train).to(device)

    adapter = Adapter(coles_train.shape[1], llm_emb_train.shape[1]).to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    bce = nn.BCEWithLogitsLoss()
    infonce = InfoNCELoss()

    best_auc = 0
    for epoch in range(200):
        adapter.train()
        idx = torch.randperm(len(X_c))
        for s in range(0, len(X_c), 256):
            b = idx[s:s+256]
            adapted, z_s, z_t, logits = adapter(X_c[b], X_t[b])
            loss = (1-alpha) * bce(logits, y_t[b]) + alpha * infonce(z_s, z_t)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        adapter.eval()
        with torch.no_grad():
            ad_tr, _, _, _ = adapter(X_c)
            ad_te, _, _, _ = adapter(X_c_te)
        sc_a = MaxAbsScaler()
        lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=42)
        lgbm.fit(sc_a.fit_transform(ad_tr.cpu().numpy()), y_train)
        p = lgbm.predict_proba(sc_a.transform(ad_te.cpu().numpy()))[:, 1]
        auc = roc_auc_score(y_test, p)
        if auc > best_auc:
            best_auc = auc

        if (epoch+1) % 50 == 0:
            print(f"    α={alpha} epoch {epoch+1}: AUC={auc:.4f} (best={best_auc:.4f})")

    results[f"llm4es_adapter_alpha{alpha}"] = best_auc
    del adapter; torch.cuda.empty_cache()

print("LLM4ES DISTILLATION SUMMARY")
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name:<30} AUC = {auc:.4f}")

with open(OUTPUT_DIR / "llm4es_results.json", "w") as f:
    json.dump(results, f, indent=2)
pd.DataFrame([{"method": k, "auc": v} for k, v in results.items()]).to_csv(
    OUTPUT_DIR / "llm4es_results.csv", index=False)
