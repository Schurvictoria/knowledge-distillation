import time, json, warnings, gc, os
from pathlib import Path

warnings.filterwarnings("ignore")
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
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from datasets import Dataset

from ptls.data_load.datasets import MemoryMapDataset, inference_data_loader
from ptls.nn import TrxEncoder, RnnSeqEncoder

OUTPUT_DIR = Path("results/age_llm4es")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUTPUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LGBM_P = dict(n_estimators=1000, learning_rate=0.02, objective="multiclass", num_class=4,
              max_depth=12, num_leaves=50, subsample=0.75, colsample_bytree=0.75,
              reg_alpha=1, reg_lambda=1, min_child_samples=50, verbosity=-1)

MCC_GROUPS = {range(1,1500):"Agriculture",range(4000,4800):"Transportation",range(5000,5600):"Retail",
              range(5600,5700):"Clothing",range(5800,5900):"Restaurants",range(6000,7000):"Financial",
              range(7500,7600):"Auto Services",range(8000,8100):"Medical",range(8200,8300):"Education"}
def mcc_cat(mcc):
    try: mcc=int(mcc)
    except: return "Other"
    for r,n in MCC_GROUPS.items():
        if mcc in r: return n
    return "Other"

print("STEP 1: Load Age data")

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

cids_train = np.load("embeddings/age/cids_train_seed42.npy")
cids_test = np.load("embeddings/age/cids_test_seed42.npy")
y_train = np.load("embeddings/age/y_train_seed42.npy")
y_test = np.load("embeddings/age/y_test_seed42.npy")
all_cids = np.concatenate([cids_train, cids_test])

def serialize(cid, max_txns=50):
    if cid not in grouped.groups: return "No transactions."
    ct = grouped.get_group(cid).tail(max_txns)
    lines = [f"Client transaction history ({len(ct)} transactions):"]
    for _, r in ct.iterrows():
        d = "spent" if r["amount_rur"]<0 else "received"
        lines.append(f"Day {int(r['trans_date'])}: {d} {abs(r['amount_rur']):.0f} at {mcc_cat(r['small_group'])}")
    return "\n".join(lines)


print("STEP 2: Fine-tune Qwen2.5-3B (5 epochs, 27k clients)")

MODEL_ID = "Qwen/Qwen2.5-3B"
LORA_CKPT = CKPT_DIR / "llm4es_lora"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

if (LORA_CKPT / "adapter_model.safetensors").exists():
    print(f"  Checkpoint found at {LORA_CKPT}, skipping fine-tuning")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(model, str(LORA_CKPT))
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                              lora_dropout=0.05, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("  Serializing ALL train clients...")
    train_texts = [serialize(cid) for cid in cids_train]
    ds = Dataset.from_dict({"text": train_texts})
    ds_tok = ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=512, padding=False),
                    batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=str(CKPT_DIR / "ft_runs"),
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=200,
        save_strategy="epoch",
        bf16=True, report_to="none", dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_tok,
                      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    print(f"  Fine-tuning 5 epochs on {len(train_texts)} clients...")
    trainer.train()
    model.save_pretrained(str(LORA_CKPT))
    tokenizer.save_pretrained(str(LORA_CKPT))
    print(f"  Saved to {LORA_CKPT}")
    del trainer; torch.cuda.empty_cache(); gc.collect()

model.eval()

print("STEP 3: Extract embeddings (mean pool last 8 layers)")

EMB_PATH = OUTPUT_DIR / "llm4es_embeddings.npz"
if EMB_PATH.exists():
    llm_all = np.load(EMB_PATH)["embeddings"].astype(np.float32)
    print(f"  Loaded cached: {llm_all.shape}")
else:
    hidden_size = model.config.hidden_size
    llm_all = np.zeros((len(all_cids), hidden_size), dtype=np.float16)
    print(f"  Extracting for {len(all_cids)} clients (dim={hidden_size})...")
    for i, cid in enumerate(all_cids):
        text = serialize(cid)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            hidden = torch.stack(out.hidden_states[-8:]).mean(0)
            mask = inputs["attention_mask"][0].unsqueeze(-1).float()
            pooled = (hidden[0] * mask).sum(0) / mask.sum(0)
            llm_all[i] = pooled.cpu().float().numpy().astype(np.float16)
        if (i+1) % 2000 == 0:
            print(f"    {i+1}/{len(all_cids)}")
    np.savez_compressed(EMB_PATH, embeddings=llm_all, cid_order=all_cids)
    print(f"  Saved: {llm_all.shape}")
    llm_all = llm_all.astype(np.float32)

del model, tokenizer; torch.cuda.empty_cache(); gc.collect()

n_tr = len(cids_train)
llm_train, llm_test = llm_all[:n_tr], llm_all[n_tr:]

print("STEP 4: Evaluate v2 vs v1")

coles_train = np.load("embeddings/age/emb_train_seed42.npy")
coles_test = np.load("embeddings/age/emb_test_seed42.npy")

results = {}

sc = MaxAbsScaler()
lgbm = LGBMClassifier(**LGBM_P, random_state=42)
lgbm.fit(sc.fit_transform(coles_train), y_train)
results["coles_only"] = accuracy_score(y_test, lgbm.predict(sc.transform(coles_test)))

sc_l = MaxAbsScaler()
lgbm = LGBMClassifier(**LGBM_P, random_state=42)
lgbm.fit(sc_l.fit_transform(llm_train), y_train)
results["llm4es_only"] = accuracy_score(y_test, lgbm.predict(sc_l.transform(llm_test)))

Xtr = np.hstack([sc.fit_transform(coles_train), sc_l.fit_transform(llm_train)])
Xte = np.hstack([sc.transform(coles_test), sc_l.transform(llm_test)])
lgbm = LGBMClassifier(**LGBM_P, random_state=42)
lgbm.fit(Xtr, y_train)
results["concat"] = accuracy_score(y_test, lgbm.predict(Xte))

print(f"  CoLES only:    {results['coles_only']:.4f}")
print(f"  LLM4ES only:   {results['llm4es_only']:.4f}")
print(f"  Concat:        {results['concat']:.4f}")

print("STEP 5: True LATTE + Bidirectional with v2 embeddings")

from functools import partial

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
    ("data/train_target.csv", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("data/transactions_train.csv", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/cids_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/cids_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/emb_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/emb_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/y_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/y_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("results/age_llm4es/llm4es_embeddings.npz", "experiments/rq2_d1_teacher_signals/feature_based/E2_2_age_llm4es.py"),
]
for _p, _hint in _required_inputs:
    assert _P(_p).exists(), f"\n  Missing input: {_p}\n  Run prerequisite: {_hint}"

def build_encoder():
    trx = TrxEncoder(embeddings={"small_group":{"in": len(sg_enc.classes_)+2, "out":16}},
                      numeric_values={"amount":"identity"}, embeddings_noise=0.003, use_batch_norm_with_lens=True)
    return RnnSeqEncoder(trx_encoder=trx, hidden_size=800, type="gru", bidir=False, trainable_starter="static")

def build_records_from_cids(cid_set):
    records = []
    for cid in cid_set:
        if cid not in grouped.groups: continue
        ct = grouped.get_group(cid)
        if len(ct) < 25: continue
        days = ct["trans_date"].values.astype(np.float32)
        records.append({"customer_id": cid, "target": target_map[cid],
                        "event_time": torch.FloatTensor(days - days[0]),
                        "amount": torch.FloatTensor(ct["amount_rur"].values),
                        "small_group": torch.LongTensor(sg_enc.transform(ct["small_group"].values) + 1)})
    return records

train_rec = build_records_from_cids(cids_train)
test_rec = build_records_from_cids(cids_test)

COLES_CKPT = Path("results/age_true_latte/coles_baseline.pt")
seq_encoder = build_encoder().to(device)
seq_encoder.load_state_dict(torch.load(COLES_CKPT, map_location=device))

sc_l2 = StandardScaler()
llm_t = torch.FloatTensor(sc_l2.fit_transform(llm_train)).to(device)

proj_seq = nn.Sequential(nn.Linear(800, 256), nn.ReLU(), nn.Linear(256, 128)).to(device)
proj_text = nn.Sequential(nn.Linear(llm_train.shape[1], 256), nn.ReLU(), nn.Linear(256, 128)).to(device)
classifier = nn.Sequential(nn.Linear(800, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 4)).to(device)

def extract_embs(encoder, records):
    encoder.eval()
    dl = inference_data_loader(records, num_workers=0, batch_size=128)
    with torch.no_grad():
        return torch.cat([encoder(b.to(device)).cpu() for b in dl]).numpy()

def eval_coles():
    emb_tr = extract_embs(seq_encoder, train_rec)
    emb_te = extract_embs(seq_encoder, test_rec)
    sc2 = MaxAbsScaler()
    lgbm = LGBMClassifier(**LGBM_P, random_state=42)
    lgbm.fit(sc2.fit_transform(emb_tr), y_train)
    return accuracy_score(y_test, lgbm.predict(sc2.transform(emb_te)))

for alpha in [0.05, 0.1]:
    print(f"\n  True LATTE α={alpha} with v2 embeddings...")
    seq_encoder.load_state_dict(torch.load(COLES_CKPT, map_location=device))
    for m in [proj_seq, proj_text, classifier]:
        for p in m.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    opt = torch.optim.Adam(list(seq_encoder.parameters()) + list(proj_seq.parameters()) +
                           list(proj_text.parameters()) + list(classifier.parameters()),
                           lr=3e-4, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    best = results["coles_only"]

    for ep in range(20):
        seq_encoder.train()
        idx = torch.randperm(len(train_rec))
        for s in range(0, len(train_rec), 32):
            bi = idx[s:s+32].tolist()
            dl = inference_data_loader([train_rec[i] for i in bi], num_workers=0, batch_size=32)
            for batch in dl:
                se = seq_encoder(batch.to(device))
            zs = F.normalize(proj_seq(se), dim=1)
            zt = F.normalize(proj_text(llm_t[bi]), dim=1)
            lo = zs @ zt.T / 0.07
            la = torch.arange(len(zs), device=device)
            yb = torch.LongTensor([train_rec[i]["target"] for i in bi]).to(device)
            loss = (1-alpha)*ce(classifier(se), yb) + alpha*(F.cross_entropy(lo,la)+F.cross_entropy(lo.T,la))/2
            opt.zero_grad(); loss.backward(); opt.step()

        if (ep+1) % 5 == 0:
            acc = eval_coles()
            if acc > best:
                best = acc
                torch.save(seq_encoder.state_dict(), OUTPUT_DIR / f"coles_latte_alpha{alpha}.pt")
            print(f"    ep {ep+1}: acc={acc:.4f} (best={best:.4f})")

    results[f"latte_alpha{alpha}"] = best

print("AGE LLM4ES v2 SUMMARY")
for n, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {n:<25} acc={v:.4f}")

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)
