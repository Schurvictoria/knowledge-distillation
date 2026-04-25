#!/usr/bin/env python3
"""
True LATTE α=0.05 on Gender, 5 seeds.
Hypothesis: lower α helps (pattern from Age where α=0.05 >> α=0.1).
Reuses existing CoLES baseline checkpoint + LLM4ES embeddings.
"""
import json, warnings, gc, random
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, StandardScaler
from lightgbm import LGBMClassifier

from ptls.data_load.datasets import inference_data_loader
from ptls.nn import TrxEncoder, RnnSeqEncoder

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
    ("data/gender_train.csv", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
    ("data/transactions.csv", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
    ("embeddings/gender/cids_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
    ("embeddings/gender/cids_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
    ("results/gender_llm4es/llm4es_embeddings.npz", "experiments/rq2_d1_teacher_signals/feature_based/E2_2_gender_llm4es.py"),
]
for _p, _hint in _required_inputs:
    assert _P(_p).exists(), f"\n  Missing input: {_p}\n  Run prerequisite: {_hint}"
# ---- end input check ----



SEEDS = [42, 123, 456, 789, 1024]
ALPHA = 0.05
N_EPOCHS = 30
OUT = Path("results/seeded_eval")
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LGBM_P = dict(n_estimators=500, learning_rate=0.02, max_depth=6, subsample=0.5,
              colsample_bytree=0.75, reg_alpha=1, reg_lambda=1, min_child_samples=50, verbosity=-1)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---- Load data (once) ----
print("Loading Gender data...")
tx = pd.read_csv(DATA / "transactions.csv")
labels = pd.read_csv(DATA / "gender_train.csv")
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
encs = {}
for col in ["mcc_code", "tr_type"]:
    tx[col] = tx[col].fillna("UNK").astype(str)
    encs[col] = LabelEncoder().fit(tx[col])
grouped = tx.groupby("customer_id")
feature_dims = {col: len(enc.classes_) + 2 for col, enc in encs.items()}

def build_records(cid_set):
    records = []
    for cid in cid_set:
        if cid not in target_map or cid not in grouped.groups:
            continue
        ct = grouped.get_group(cid)
        if len(ct) < 25:
            continue
        days = ct["day_float"].values
        rec = {"customer_id": cid, "target": target_map[cid],
               "event_time": torch.FloatTensor((days - days[0]).astype(np.float32)),
               "amount": torch.FloatTensor(ct["amount"].values)}
        for col, enc in encs.items():
            rec[col] = torch.LongTensor(enc.transform(ct[col].values) + 1)
        records.append(rec)
    return records

ids = labels["customer_id"].values
targets = np.array([target_map[c] for c in ids])
idx_tr, idx_te = train_test_split(np.arange(len(ids)), test_size=0.1, random_state=42, stratify=targets)
train_rec = build_records(set(ids[idx_tr]))
test_rec = build_records(set(ids[idx_te]))
y_train = np.array([r["target"] for r in train_rec])
y_test = np.array([r["target"] for r in test_rec])

# LLM4ES embeddings
cids_train_order = [r["customer_id"] for r in train_rec]
llm_all = np.load("results/gender_llm4es/llm4es_embeddings.npz")["embeddings"].astype(np.float32)
cids_emb = np.concatenate([np.load("embeddings/gender/cids_train_seed42.npy"),
                           np.load("embeddings/gender/cids_test_seed42.npy")])
cid_to_llm = {cid: llm_all[i] for i, cid in enumerate(cids_emb)}
llm_train = np.array([cid_to_llm.get(c, np.zeros(llm_all.shape[1])) for c in cids_train_order])
sc_l = StandardScaler()
llm_train_t = torch.FloatTensor(sc_l.fit_transform(llm_train)).to(device)

COLES_CKPT = "results/gender_true_latte/coles_baseline.pt"
print(f"  train={len(train_rec)}, test={len(test_rec)}, LLM dim={llm_train_t.shape[1]}")


def build_encoder():
    trx = TrxEncoder(
        embeddings={"mcc_code": {"in": feature_dims["mcc_code"], "out": 48},
                    "tr_type": {"in": feature_dims["tr_type"], "out": 24}},
        numeric_values={"amount": "identity"}, embeddings_noise=0.003,
        use_batch_norm_with_lens=True)
    return RnnSeqEncoder(trx_encoder=trx, hidden_size=1024, type="gru",
                         bidir=False, trainable_starter="static")


def extract(enc, records, bs=64):
    enc.eval()
    dl = inference_data_loader(records, num_workers=0, batch_size=bs)
    with torch.no_grad():
        return torch.cat([enc(b.to(device)).cpu() for b in dl]).numpy()


def eval_lgbm(emb_tr, emb_te, seed):
    sc = MaxAbsScaler()
    clf = LGBMClassifier(**LGBM_P, random_state=seed)
    clf.fit(sc.fit_transform(emb_tr), y_train)
    return roc_auc_score(y_test, clf.predict_proba(sc.transform(emb_te))[:, 1])


def train_one_seed(seed):
    set_seed(seed)
    enc = build_encoder().to(device)
    enc.load_state_dict(torch.load(COLES_CKPT, map_location=device))

    proj_s = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 128)).to(device)
    proj_t = nn.Sequential(nn.Linear(llm_train_t.shape[1], 256), nn.ReLU(), nn.Linear(256, 128)).to(device)
    classifier = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)).to(device)

    params = list(enc.parameters()) + list(proj_s.parameters()) + \
             list(proj_t.parameters()) + list(classifier.parameters())
    opt = torch.optim.Adam(params, lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)
    bce = nn.BCEWithLogitsLoss()

    baseline = eval_lgbm(extract(enc, train_rec), extract(enc, test_rec), seed)
    best = baseline
    g = torch.Generator().manual_seed(seed)

    for ep in range(N_EPOCHS):
        enc.train(); proj_s.train(); classifier.train()
        idx = torch.randperm(len(train_rec), generator=g)
        for s in range(0, len(train_rec), 32):
            bi = idx[s:s + 32].tolist()
            dl = inference_data_loader([train_rec[i] for i in bi], num_workers=0, batch_size=32)
            for batch in dl:
                seq_emb = enc(batch.to(device))
            z_s = F.normalize(proj_s(seq_emb), dim=1)
            z_t = F.normalize(proj_t(llm_train_t[bi]), dim=1)
            lo = z_s @ z_t.T / 0.07
            la = torch.arange(len(z_s), device=device)
            loss_c = (F.cross_entropy(lo, la) + F.cross_entropy(lo.T, la)) / 2
            y_b = torch.FloatTensor([train_rec[i]["target"] for i in bi]).to(device)
            loss_cls = bce(classifier(seq_emb).squeeze(-1), y_b)
            loss = (1 - ALPHA) * loss_cls + ALPHA * loss_c
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sch.step()

        if (ep + 1) % 5 == 0:
            emb_tr = extract(enc, train_rec)
            emb_te = extract(enc, test_rec)
            auc = eval_lgbm(emb_tr, emb_te, seed)
            if auc > best:
                best = auc
                torch.save(enc.state_dict(), OUT / f"gender_latte_a005_seed{seed}.pt")
            print(f"    seed={seed} ep={ep+1}/{N_EPOCHS} AUC={auc:.4f} best={best:.4f}", flush=True)

    del enc, proj_s, proj_t, classifier
    torch.cuda.empty_cache(); gc.collect()
    return {"seed": seed, "baseline": baseline, "best": best}


# ---- Run ----
print(f"\nTrue LATTE α={ALPHA}, Gender, 5 seeds\n{'='*50}", flush=True)
results = []
for seed in SEEDS:
    print(f"\n  [seed={seed}]", flush=True)
    r = train_one_seed(seed)
    results.append(r)
    print(f"  seed={seed}: baseline={r['baseline']:.4f} best={r['best']:.4f} "
          f"(Δ={r['best']-r['baseline']:+.4f})", flush=True)

bests = [r["best"] for r in results]
baselines = [r["baseline"] for r in results]
summary = {
    "dataset": "gender", "alpha": ALPHA, "n_epochs": N_EPOCHS,
    "seeds": SEEDS, "per_seed": results,
    "baseline_mean": float(np.mean(baselines)),
    "baseline_std": float(np.std(baselines)),
    "best_mean": float(np.mean(bests)),
    "best_std": float(np.std(bests)),
}
delta = summary["best_mean"] - summary["baseline_mean"]
print(f"\n{'='*50}")
print(f"Gender True LATTE α={ALPHA}:")
print(f"  baseline: {summary['baseline_mean']:.4f} ± {summary['baseline_std']:.4f}")
print(f"  best:     {summary['best_mean']:.4f} ± {summary['best_std']:.4f}")
print(f"  Δ = {delta:+.4f} pp")
print(f"  Compare: α=0.1 was 0.8674±0.0005, α=0.3 was 0.8664±0.0007")

with open(OUT / "gender_latte_a005.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {OUT / 'gender_latte_a005.json'}")
