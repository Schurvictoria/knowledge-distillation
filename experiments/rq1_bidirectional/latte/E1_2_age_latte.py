#!/usr/bin/env python3
"""
True LATTE distillation on Age: fine-tune CoLES GRU with contrastive alignment.
Age: 30k clients, 4-class, GRU-800. Uses pre-computed LLM4ES embeddings.
"""

import time, json, warnings, gc, random, os
from pathlib import Path
from functools import partial

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, StandardScaler
from lightgbm import LGBMClassifier

from ptls.data_load.datasets import MemoryMapDataset, inference_data_loader
from ptls.frames.coles import CoLESModule, ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.nn import TrxEncoder, RnnSeqEncoder

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
pl.seed_everything(SEED, workers=True)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---- File existence checks ----
from pathlib import Path as _P
_required = [
    _P("data/transactions_train.csv"),
    _P("embeddings/age/cids_train_seed42.npy"),
    _P("embeddings/age/cids_test_seed42.npy"),
    _P("results/age_llm4es/llm4es_embeddings.npz"),
]
for _p in _required:
    assert _p.exists(), (
        f"\n  Missing: {_p}"
        f"\n  -> Run baseline:    python experiments/infrastructure/run_age_coles.py"
        f"\n  -> Run LLM4ES gen:  python experiments/rq2_d1_teacher_signals/feature_based/E2_2_age_llm4es.py"
    )

OUTPUT_DIR = Path("results/age_true_latte")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LGBM_P = dict(n_estimators=1000, learning_rate=0.02, objective="multiclass", num_class=4,
              max_depth=12, num_leaves=50, subsample=0.75, colsample_bytree=0.75,
              reg_alpha=1, reg_lambda=1, min_child_samples=50, verbosity=-1)

# ---- Load Age data ----
print("Loading Age data...")
tx = pd.read_csv(DATA_DIR / "transactions_train.csv")
labels = pd.read_csv(DATA_DIR / "train_target.csv")
target_map = dict(zip(labels["client_id"], labels["bins"]))
tx = tx.sort_values(["client_id", "trans_date"])
tx["amount_rur"] = np.sign(tx["amount_rur"]) * np.log1p(np.abs(tx["amount_rur"]))
tx["small_group"] = tx["small_group"].fillna(0).astype(str)
sg_enc = LabelEncoder().fit(tx["small_group"])

ids = labels["client_id"].values
targets = np.array([target_map[c] for c in ids])
idx_tr, idx_te = train_test_split(np.arange(len(ids)), test_size=0.1, random_state=42, stratify=targets)
train_ids, test_ids = set(ids[idx_tr]), set(ids[idx_te])
grouped = tx.groupby("client_id")

def build_records(cid_set):
    records = []
    for cid in cid_set:
        if cid not in target_map or cid not in grouped.groups: continue
        ct = grouped.get_group(cid)
        if len(ct) < 25: continue
        days = ct["trans_date"].values.astype(np.float32)
        records.append({
            "customer_id": cid, "target": target_map[cid],
            "event_time": torch.FloatTensor(days - days[0]),
            "amount": torch.FloatTensor(ct["amount_rur"].values),
            "small_group": torch.LongTensor(sg_enc.transform(ct["small_group"].values) + 1),
        })
    return records

train_rec_full = build_records(train_ids)
test_rec = build_records(test_ids)
feature_dims = {"small_group": len(sg_enc.classes_) + 2}
y_train_full = np.array([r["target"] for r in train_rec_full])
y_test = np.array([r["target"] for r in test_rec])

tr_idx, val_idx = train_test_split(
    np.arange(len(train_rec_full)), test_size=0.1, random_state=42, stratify=y_train_full)
train_rec = [train_rec_full[i] for i in tr_idx]
val_rec = [train_rec_full[i] for i in val_idx]
y_train = y_train_full[tr_idx]
y_val = y_train_full[val_idx]
print(f"  train={len(train_rec)}, val={len(val_rec)}, test={len(test_rec)}")

# Load LLM4ES embeddings
cids_train_order = [r["customer_id"] for r in train_rec]
cids_test_order = [r["customer_id"] for r in test_rec]
cids_emb_tr = np.load("embeddings/age/cids_train_seed42.npy")
cids_emb_te = np.load("embeddings/age/cids_test_seed42.npy")
llm_all = np.load("results/age_llm4es/llm4es_embeddings.npz")["embeddings"].astype(np.float32)
all_cids_emb = np.concatenate([cids_emb_tr, cids_emb_te])
assert len(llm_all) == len(all_cids_emb), \
    f'LLM emb count ({len(llm_all)}) != cids count ({len(all_cids_emb)}) — bad alignment'
cid_to_llm = {cid: llm_all[i] for i, cid in enumerate(all_cids_emb)}
missing_tr = [c for c in cids_train_order if c not in cid_to_llm]
missing_te = [c for c in cids_test_order if c not in cid_to_llm]
assert not missing_tr, f"Missing LLM embedding for {len(missing_tr)} train cids"
assert not missing_te, f"Missing LLM embedding for {len(missing_te)} test cids"
llm_train = np.array([cid_to_llm[c] for c in cids_train_order])
llm_test = np.array([cid_to_llm[c] for c in cids_test_order])
sc_l = StandardScaler()
llm_train_t = torch.FloatTensor(sc_l.fit_transform(llm_train)).to(device)
print(f"  train={len(train_rec)}, test={len(test_rec)}, LLM={llm_train_t.shape}")

# ---- Build CoLES ----
COLES_CKPT = OUTPUT_DIR / "coles_baseline.pt"

def build_seq_encoder():
    trx_encoder = TrxEncoder(
        embeddings={"small_group": {"in": feature_dims["small_group"], "out": 16}},
        numeric_values={"amount": "identity"}, embeddings_noise=0.003, use_batch_norm_with_lens=True)
    return RnnSeqEncoder(trx_encoder=trx_encoder, hidden_size=800,
                         type="gru", bidir=False, trainable_starter="static")

if COLES_CKPT.exists():
    print("  Loading CoLES checkpoint...")
    seq_encoder = build_seq_encoder()
    seq_encoder.load_state_dict(torch.load(COLES_CKPT, map_location="cpu"))
    seq_encoder = seq_encoder.to(device)
else:
    seq_encoder = build_seq_encoder()
    coles_module = CoLESModule(
        seq_encoder=seq_encoder,
        optimizer_partial=partial(torch.optim.Adam, lr=0.001),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=30, gamma=0.9025))
    splitter = SampleSlices(split_count=5, cnt_min=25, cnt_max=200)
    dataset = ColesDataset(MemoryMapDataset(train_rec), splitter=splitter)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn)
    print("  Training CoLES 150 epochs...")
    trainer = pl.Trainer(max_epochs=150, accelerator="gpu", devices=1, enable_progress_bar=True,
                         enable_checkpointing=False, logger=False)
    trainer.fit(coles_module, loader)
    torch.save(coles_module._seq_encoder.state_dict(), COLES_CKPT)
    seq_encoder = coles_module._seq_encoder.to(device)
    del coles_module, trainer, loader, dataset; torch.cuda.empty_cache(); gc.collect()

def extract_embs(encoder, records):
    encoder.eval()
    dl = inference_data_loader(records, num_workers=0, batch_size=128)
    chunks = []
    with torch.no_grad():
        for batch in dl:
            chunks.append(encoder(batch.to(device)).cpu())
    return torch.cat(chunks).numpy()

emb_base_tr = extract_embs(seq_encoder, train_rec)
emb_base_te = extract_embs(seq_encoder, test_rec)
sc = MaxAbsScaler()
lgbm = LGBMClassifier(**LGBM_P, random_state=42)
lgbm.fit(sc.fit_transform(emb_base_tr), y_train)
baseline = accuracy_score(y_test, lgbm.predict(sc.transform(emb_base_te)))
print(f"  Baseline CoLES: acc={baseline:.4f}")

# ---- Fine-tune ----
print("\nFine-tuning CoLES with contrastive alignment...")
results = {"baseline_coles": baseline}

proj_seq = nn.Sequential(nn.Linear(800, 256), nn.ReLU(), nn.Linear(256, 128)).to(device)
proj_text = nn.Sequential(nn.Linear(llm_train_t.shape[1], 256), nn.ReLU(), nn.Linear(256, 128)).to(device)
classifier = nn.Sequential(nn.Linear(800, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 4)).to(device)

for alpha in [0.1]:
    print(f"\n--- α={alpha} ---")
    seq_encoder.load_state_dict(torch.load(COLES_CKPT, map_location=device))
    seq_encoder.train()
    for m in [proj_seq, proj_text, classifier]:
        for p in m.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    opt = torch.optim.Adam(list(seq_encoder.parameters()) + list(proj_seq.parameters()) +
                           list(proj_text.parameters()) + list(classifier.parameters()),
                           lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20)
    ce = nn.CrossEntropyLoss()
    sc_init = MaxAbsScaler()
    lgbm_init = LGBMClassifier(**LGBM_P, random_state=42)
    lgbm_init.fit(sc_init.fit_transform(extract_embs(seq_encoder, train_rec)), y_train)
    best_val = accuracy_score(y_val, lgbm_init.predict(
        sc_init.transform(extract_embs(seq_encoder, val_rec))))
    best_test = baseline

    for ep in range(20):
        seq_encoder.train(); proj_seq.train(); classifier.train()
        idx = torch.randperm(len(train_rec))
        for s in range(0, len(train_rec), 128):
            b_idx = idx[s:s+128].tolist()
            dl = inference_data_loader([train_rec[i] for i in b_idx], num_workers=0, batch_size=128)
            for batch in dl:
                seq_emb = seq_encoder(batch.to(device))
                z_s = F.normalize(proj_seq(seq_emb), dim=1)
                z_t = F.normalize(proj_text(llm_train_t[b_idx]), dim=1)
                lo = z_s @ z_t.T / 0.07
                la = torch.arange(len(z_s), device=device)
                loss_c = (F.cross_entropy(lo, la) + F.cross_entropy(lo.T, la)) / 2
                y_b = torch.LongTensor([train_rec[i]["target"] for i in b_idx]).to(device)
                loss_cls = ce(classifier(seq_emb), y_b)
                loss = (1-alpha)*loss_cls + alpha*loss_c
                opt.zero_grad(); loss.backward(); opt.step()
        sch.step()

        if (ep+1) % 5 == 0:
            emb_tr = extract_embs(seq_encoder, train_rec)
            emb_val = extract_embs(seq_encoder, val_rec)
            emb_te = extract_embs(seq_encoder, test_rec)
            sc2 = MaxAbsScaler()
            lgbm = LGBMClassifier(**LGBM_P, random_state=42)
            lgbm.fit(sc2.fit_transform(emb_tr), y_train)
            val_acc = accuracy_score(y_val, lgbm.predict(sc2.transform(emb_val)))
            test_acc = accuracy_score(y_test, lgbm.predict(sc2.transform(emb_te)))
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
                torch.save(seq_encoder.state_dict(), OUTPUT_DIR / f"coles_finetuned_α{alpha}.pt")
            print(f"  ep {ep+1}: val={val_acc:.4f} test={test_acc:.4f} best_val={best_val:.4f} best_test={best_test:.4f}")

    results[f"finetune_α{alpha}"] = best_test
    torch.cuda.empty_cache(); gc.collect()

print("\n" + "=" * 60)
for n, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {n:<25} acc={v:.4f} ({'+' if v-baseline>=0 else ''}{v-baseline:.4f})")

with open(OUTPUT_DIR / "true_latte_results.json", "w") as f:
    json.dump(results, f, indent=2)
