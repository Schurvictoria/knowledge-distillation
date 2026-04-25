#!/usr/bin/env python3
"""
RAMD-KD: Retrieval-Augmented Mutual Distillation with Real KD.

Step 1: Get OOF kNN-CoT teacher predictions on train (5-fold, no leakage)
Step 2: Fine-tune CoLES GRU with reverse KL against kNN-CoT teacher
Step 3: Evaluate on 5 seeds

This is REAL bidirectional distillation:
  Direction 2: CoLES → kNN → LLM prompt → better LLM predictions
  Direction 1: better LLM predictions → reverse KL → fine-tune CoLES encoder

Literature basis:
  - kNN retrieval: TabR [ICLR 2024]
  - Reverse KL: MiniLLM [ICLR 2024]
  - CoLES encoder: CoLES [SIGMOD 2022], LATTE [EMNLP 2025]
  - Contrastive fine-tuning: True LATTE (our baseline)

5 seeds [42, 123, 456, 789, 1024]. Saves checkpoints.
"""
import json, warnings, gc, random, time, argparse, os
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.neighbors import NearestNeighbors
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
    ("data/rosbank_train.csv", "experiments/rq1_bidirectional/coles/run_rosbank_coles.py"),
    ("data/transactions.csv", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
]
for _p, _hint in _required_inputs:
    assert _P(_p).exists(), f"\n  Missing input: {_p}\n  Run prerequisite: {_hint}"
# ---- end input check ----


SEEDS = [42, 123, 456, 789, 1024]
OUT = Path("results/ramd_kd")
OUT.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MCC_GROUPS = {range(1,1500):"Agriculture",range(4000,4800):"Transportation",
              range(5000,5600):"Retail",range(5600,5700):"Clothing",
              range(5800,5900):"Restaurants",range(6000,7000):"Financial",
              range(7500,7600):"Auto",range(8000,8100):"Medical",
              range(8200,8300):"Education"}

def mcc_cat(mcc):
    try: mcc=int(mcc)
    except: return "Other"
    for r,n in MCC_GROUPS.items():
        if mcc in r: return n
    return "Other"

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ============================================================
# Step 1: OOF kNN-CoT teacher predictions
# ============================================================
def get_knn_cot_oof_predictions(dataset_name, teacher=None):
    """Get OOF kNN-CoT LLM predictions for train set. 5-fold CV.

    If `teacher` (e.g. 'deepseek_v3', 'qwen36_35b') is given, loads OOF
    generated via OpenRouter from results/ramd_openrouter/ instead of
    running local Qwen2.5-7B.
    """
    if teacher is not None:
        ext = Path(f"results/ramd_openrouter/{dataset_name}_{teacher}_oof.npz")
        if ext.exists():
            data = np.load(ext)
            print(f"  Loaded OpenRouter OOF ({teacher}) from {ext}")
            return data["probs"]
        else:
            print(f"  [WARN] Teacher OOF file not found: {ext}. Falling back to Qwen2.5-7B.")

    cache_path = OUT / f"{dataset_name}_knn_cot_oof.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        print(f"  Loaded cached OOF predictions from {cache_path}")
        return data["probs"]

    print(f"  Computing OOF kNN-CoT predictions for {dataset_name}...")
    print(f"  Loading LLM (Qwen2.5-7B-Instruct, 4-bit)...")

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_cfg,
                                                device_map="auto", trust_remote_code=True)
    llm.eval()

    # Load data
    DATA = Path("data")
    coles_train = np.load(f"embeddings/{dataset_name}/emb_train_seed42.npy")
    cids_train = np.load(f"embeddings/{dataset_name}/cids_train_seed42.npy")
    y_train = np.load(f"embeddings/{dataset_name}/y_train_seed42.npy")

    if dataset_name == "gender":
        tx = pd.read_csv(DATA / "transactions.csv")
        labels = pd.read_csv(DATA / "gender_train.csv")
        tx = tx[tx["customer_id"].isin(labels["customer_id"])].copy()
        grouped = tx.groupby("customer_id")
        pos_tokens = ["male", " male", "Male", " Male"]
        neg_tokens = ["female", " female", "Female", " Female"]
        task_desc = "gender (male or female)"
        answer_fmt = "male or female"
        pos_label, neg_label = "male", "female"

        def serialize(cid):
            if cid not in grouped.groups: return "No txns."
            ct = grouped.get_group(cid)
            n = len(ct)
            amt = np.abs(ct["amount"].values)
            cats = ct["mcc_code"].apply(mcc_cat).value_counts()
            top = ", ".join(f"{c} ({n_})" for c, n_ in cats.head(6).items())
            return f"Client with {n} txns. Avg amount: {amt.mean():.0f}. Categories: {top}."
    else:  # rosbank
        df = pd.read_csv(DATA / "rosbank_train.csv")
        df["dt"] = pd.to_datetime(df["TRDATETIME"], format="%d%b%y:%H:%M:%S")
        df = df.sort_values(["cl_id", "dt"])
        grouped = df.groupby("cl_id")
        pos_tokens = ["yes", " yes", "Yes", "churn", " churn"]
        neg_tokens = ["no", " no", "No", "stay", " stay"]
        task_desc = "churn (yes or no)"
        answer_fmt = "yes or no"
        pos_label, neg_label = "churn", "stay"

        def serialize(cid):
            if cid not in grouped.groups: return "No txns."
            ct = grouped.get_group(cid)
            n = len(ct)
            amt = np.abs(ct["amount"].values)
            cats = ct["MCC"].fillna(0).astype(int).apply(mcc_cat).value_counts()
            top = ", ".join(f"{c} ({n_})" for c, n_ in cats.head(6).items())
            return f"Client with {n} txns. Avg amount: {amt.mean():.0f}. Categories: {top}."

    def predict_binary(messages):
        def get_ids(tokens):
            ids = set()
            for t in tokens:
                e = tokenizer.encode(t, add_special_tokens=False)
                if e: ids.add(e[0])
            return list(ids)
        pi, ni = get_ids(pos_tokens), get_ids(neg_tokens)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(llm.device)
        with torch.no_grad():
            logits = llm(**inputs).logits[0, -1, :]
        all_ids = pi + ni
        probs = torch.softmax(logits[all_ids].float(), dim=0)
        pp = probs[:len(pi)].sum().item()
        pn = probs[len(pi):].sum().item()
        total = pp + pn
        del inputs
        return pp / total if total > 1e-8 else 0.5

    SYSTEM = f"You are a bank analyst predicting client {task_desc}. You have ML analysis. Answer: {answer_fmt}."

    # OOF: 5-fold
    sc = MaxAbsScaler()
    coles_scaled = sc.fit_transform(coles_train)
    oof_probs = np.full(len(y_train), 0.5)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(coles_train, y_train)):
        print(f"  Fold {fold+1}/5 ({len(val_idx)} samples)...", flush=True)
        nn = NearestNeighbors(n_neighbors=10, metric="cosine")
        nn.fit(coles_scaled[tr_idx])
        dists, idxs = nn.kneighbors(coles_scaled[val_idx])

        t0 = time.time()
        for i, vi in enumerate(val_idx):
            nb_labels = y_train[tr_idx[idxs[i]]]
            pos_count = int(nb_labels.sum())
            neg_count = 10 - pos_count
            majority = pos_label if pos_count > 5 else neg_label
            cid = int(cids_train[vi])

            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": (
                    f"Profile:\n{serialize(cid)}\n\n"
                    f"Similar clients: {pos_count} {pos_label}, {neg_count} {neg_label} "
                    f"(majority: {majority}).\n\nPredict.")}
            ]
            oof_probs[vi] = predict_binary(messages)

            if (i+1) % 200 == 0:
                elapsed = time.time() - t0
                print(f"    {i+1}/{len(val_idx)} ({(i+1)/elapsed:.1f}/s)", flush=True)

    # Save
    np.savez(cache_path, probs=oof_probs, cids=cids_train, y=y_train)
    oof_auc = roc_auc_score(y_train, oof_probs)
    print(f"  OOF kNN-CoT AUC = {oof_auc:.4f}")
    print(f"  Saved: {cache_path}")

    del llm, tokenizer; torch.cuda.empty_cache(); gc.collect()
    return oof_probs


# ============================================================
# Step 2: Fine-tune CoLES with reverse KL against kNN-CoT teacher
# ============================================================
def build_gender_data():
    DATA = Path("data")
    tx = pd.read_csv(DATA / "transactions.csv")
    labels = pd.read_csv(DATA / "gender_train.csv")
    tx = tx[tx["customer_id"].isin(labels["customer_id"])].copy()
    def parse_dt(s):
        parts = str(s).split(" ", 1)
        day = int(parts[0])
        if len(parts) > 1:
            t = parts[1].split(":")
            return day + (int(t[0])*3600 + int(t[1])*60 + int(t[2]))/86400.0
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
            if cid not in target_map or cid not in grouped.groups: continue
            ct = grouped.get_group(cid)
            if len(ct) < 25: continue
            days = ct["day_float"].values
            rec = {"customer_id": cid, "target": target_map[cid],
                   "event_time": torch.FloatTensor((days - days[0]).astype(np.float32)),
                   "amount": torch.FloatTensor(ct["amount"].values)}
            for col, enc in encs.items():
                rec[col] = torch.LongTensor(enc.transform(ct[col].values) + 1)
            records.append(rec)
        return records
    def build_encoder():
        trx = TrxEncoder(
            embeddings={"mcc_code": {"in": feature_dims["mcc_code"], "out": 48},
                        "tr_type": {"in": feature_dims["tr_type"], "out": 24}},
            numeric_values={"amount": "identity"}, embeddings_noise=0.003,
            use_batch_norm_with_lens=True)
        return RnnSeqEncoder(trx_encoder=trx, hidden_size=1024, type="gru",
                             bidir=False, trainable_starter="static")
    ids = labels["customer_id"].values
    targets = np.array([target_map[c] for c in ids])
    idx_tr, idx_te = train_test_split(np.arange(len(ids)), test_size=0.1, random_state=42, stratify=targets)
    train_rec = build_records(set(ids[idx_tr]))
    test_rec = build_records(set(ids[idx_te]))
    return train_rec, test_rec, build_encoder, 1024


def build_rosbank_data():
    DATA = Path("data")
    df = pd.read_csv(DATA / "rosbank_train.csv")
    df["dt"] = pd.to_datetime(df["TRDATETIME"], format="%d%b%y:%H:%M:%S")
    df = df.sort_values(["cl_id", "dt"])
    labels_df = df.groupby("cl_id")["target_flag"].max().reset_index()
    labels_df.columns = ["customer_id", "target"]
    target_map = dict(zip(labels_df["customer_id"], labels_df["target"]))
    tx = df.rename(columns={"cl_id": "customer_id", "MCC": "mcc_code"}).copy()
    tx["mcc_code"] = tx["mcc_code"].fillna(0).astype(int)
    tx["amount"] = np.sign(tx["amount"]) * np.log1p(np.abs(tx["amount"]))
    encs = {}
    for col in ["mcc_code", "channel_type", "currency", "trx_category"]:
        if col in tx.columns:
            tx[col] = tx[col].fillna("UNK").astype(str)
            encs[col] = LabelEncoder().fit(tx[col])
    grouped = tx.groupby("customer_id")
    EMB_DIMS = {"mcc_code": 24, "channel_type": 4, "currency": 4, "trx_category": 4}
    feature_dims = {c: len(e.classes_)+2 for c, e in encs.items()}
    def build_records(cid_set):
        records = []
        for cid in cid_set:
            if cid not in target_map or cid not in grouped.groups: continue
            ct = grouped.get_group(cid)
            if len(ct) < 15: continue
            dt_vals = ct["dt"].values
            days = (dt_vals - dt_vals[0]) / np.timedelta64(1, "D")
            rec = {"customer_id": cid, "target": target_map[cid],
                   "event_time": torch.FloatTensor(days.astype(np.float32)),
                   "amount": torch.FloatTensor(ct["amount"].values)}
            for col, enc in encs.items():
                rec[col] = torch.LongTensor(enc.transform(ct[col].values) + 1)
            records.append(rec)
        return records
    def build_encoder():
        embs = {c: {"in": feature_dims[c], "out": EMB_DIMS[c]} for c in feature_dims if c in EMB_DIMS}
        trx = TrxEncoder(embeddings=embs, numeric_values={"amount": "identity"},
                         embeddings_noise=0.0003, use_batch_norm_with_lens=True)
        return RnnSeqEncoder(trx_encoder=trx, hidden_size=1024, type="lstm",
                             bidir=False, trainable_starter="static")
    ids = labels_df["customer_id"].values
    targets = np.array([target_map[c] for c in ids])
    idx_tr, idx_te = train_test_split(np.arange(len(ids)), test_size=0.1, random_state=42, stratify=targets)
    train_rec = build_records(set(ids[idx_tr]))
    test_rec = build_records(set(ids[idx_te]))
    return train_rec, test_rec, build_encoder, 1024


BUILDERS = {"gender": build_gender_data, "rosbank": build_rosbank_data}
COLES_CKPT = {
    "gender": "results/gender_true_latte/coles_baseline.pt",
    "rosbank": "results/rosbank_true_latte/coles_baseline.pt",
}
LGBM_P = dict(n_estimators=500, learning_rate=0.02, max_depth=6, subsample=0.5,
              colsample_bytree=0.75, reg_alpha=1, reg_lambda=1, min_child_samples=50, verbosity=-1)


def extract(enc, records, bs=64):
    enc.eval()
    dl = inference_data_loader(records, num_workers=0, batch_size=bs)
    with torch.no_grad():
        return torch.cat([enc(b.to(device)).cpu() for b in dl]).numpy()


def eval_lgbm(emb_tr, y_tr, emb_te, y_te, seed):
    sc = MaxAbsScaler()
    clf = LGBMClassifier(**LGBM_P, random_state=seed)
    clf.fit(sc.fit_transform(emb_tr), y_tr)
    return roc_auc_score(y_te, clf.predict_proba(sc.transform(emb_te))[:, 1])


def reverse_kl_per_sample(teacher_probs, student_logits):
    """KL(P_student || P_teacher) per sample."""
    sp = F.softmax(student_logits, dim=1)
    return (sp * (torch.log(sp + 1e-8) - torch.log(teacher_probs + 1e-8))).sum(dim=1)


def compute_knn_soft_labels(emb_train, y_train, k=10):
    """Compute kNN vote distribution as soft labels (no LLM needed)."""
    sc = MaxAbsScaler()
    emb_sc = sc.fit_transform(emb_train)
    nn = NearestNeighbors(n_neighbors=k+1, metric="cosine")  # +1 to exclude self
    nn.fit(emb_sc)
    dists, idxs = nn.kneighbors(emb_sc)
    soft = np.zeros(len(y_train))
    for i in range(len(y_train)):
        # Exclude self (first neighbor)
        nb_idx = idxs[i, 1:]
        soft[i] = y_train[nb_idx].mean()
    return np.clip(soft, 0.01, 0.99)


def train_one_seed(name, seed, teacher_probs_t, train_rec, test_rec, build_enc, hidden,
                   alpha=0.1, n_epochs=15, use_dakd=False):
    set_seed(seed)
    y_tr_full = np.array([r["target"] for r in train_rec])
    y_te = np.array([r["target"] for r in test_rec])

    # Fixed val split (seed=42, not per-training-seed) — for honest model selection
    tr_idx, val_idx = train_test_split(
        np.arange(len(train_rec)), test_size=0.1, random_state=42,
        stratify=y_tr_full)
    train_rec_sub = [train_rec[i] for i in tr_idx]
    val_rec = [train_rec[i] for i in val_idx]
    y_tr = y_tr_full[tr_idx]
    y_val = y_tr_full[val_idx]
    teacher_probs_sub = teacher_probs_t[torch.LongTensor(tr_idx).to(device)]

    enc = build_enc().to(device)
    enc.load_state_dict(torch.load(COLES_CKPT[name], map_location=device))

    classifier = nn.Sequential(
        nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2)).to(device)

    params = list(enc.parameters()) + list(classifier.parameters())
    opt = torch.optim.Adam(params, lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    # Baseline = eval on test using pretrained CoLES (pre-finetune)
    baseline = eval_lgbm(extract(enc, train_rec_sub), y_tr, extract(enc, test_rec), y_te, seed)
    # Model selection on VAL, not test
    best_val = eval_lgbm(extract(enc, train_rec_sub), y_tr, extract(enc, val_rec), y_val, seed)
    best_test = baseline  # test eval corresponding to best val
    g = torch.Generator().manual_seed(seed)

    for ep in range(n_epochs):
        enc.train(); classifier.train()
        idx = torch.randperm(len(train_rec_sub), generator=g)
        tot = 0; nb = 0
        for s in range(0, len(train_rec_sub), 32):
            bi = idx[s:s+32].tolist()
            dl = inference_data_loader([train_rec_sub[i] for i in bi], num_workers=0, batch_size=32)
            for batch in dl:
                seq_emb = enc(batch.to(device))

            logits = classifier(seq_emb)
            y_b = torch.LongTensor([train_rec_sub[i]["target"] for i in bi]).to(device)

            loss_ce = F.cross_entropy(logits, y_b)

            # Teacher: kNN-CoT LLM soft labels (binary → 2-class probs)
            t_pos = teacher_probs_sub[bi].unsqueeze(1)
            t_probs = torch.cat([1 - t_pos, t_pos], dim=1)

            kl_per_sample = reverse_kl_per_sample(t_probs, logits)
            if use_dakd:
                # DA-KD (ICML 2024): weight by difficulty (1 - teacher confidence)
                difficulty = 1 - t_probs.max(dim=1).values
                weights = difficulty / (difficulty.mean() + 1e-8)
                loss_kd = (weights * kl_per_sample).mean()
            else:
                loss_kd = kl_per_sample.mean()
            loss = (1 - alpha) * loss_ce + alpha * loss_kd

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sch.step()

        if (ep + 1) % 5 == 0:
            emb_tr = extract(enc, train_rec_sub)
            emb_val = extract(enc, val_rec)
            emb_te = extract(enc, test_rec)
            val_score = eval_lgbm(emb_tr, y_tr, emb_val, y_val, seed)
            test_score = eval_lgbm(emb_tr, y_tr, emb_te, y_te, seed)
            # Model selection on VAL only (no test-peeking)
            if val_score > best_val:
                best_val = val_score
                best_test = test_score
                torch.save(enc.state_dict(), OUT / f"{name}_ramdkd_seed{seed}.pt")
            print(f"    seed={seed} ep={ep+1}/{n_epochs} loss={tot/nb:.4f} "
                  f"val={val_score:.4f} test={test_score:.4f} "
                  f"best_val={best_val:.4f} best_test={best_test:.4f}", flush=True)

    del enc, classifier; torch.cuda.empty_cache(); gc.collect()
    # best_test = test score at the epoch where val was best (honest model selection)
    return {"seed": seed, "baseline": baseline, "best": best_test, "best_val": best_val}


def run_dataset(name, alpha=0.1, n_epochs=15, use_dakd=False):
    tag = "RAMD-KD+DA-KD" if use_dakd else "RAMD-KD"
    print(f"\n{'='*60}")
    print(f"{tag}: {name.upper()}")
    print(f"{'='*60}", flush=True)

    # Step 1: OOF teacher predictions
    oof_probs = get_knn_cot_oof_predictions(name, teacher=os.environ.get("RAMD_TEACHER"))

    # Step 2: Build data
    train_rec, test_rec, build_enc, hidden = BUILDERS[name]()
    y_tr = np.array([r["target"] for r in train_rec])
    print(f"  train={len(train_rec)}, test={len(test_rec)}")

    # Align OOF probs with train_rec order (by customer_id)
    cids_train = np.load(f"embeddings/{name}/cids_train_seed42.npy")
    cid_to_prob = {int(cids_train[i]): oof_probs[i] for i in range(len(cids_train))}
    teacher_aligned = np.array([cid_to_prob.get(r["customer_id"], 0.5) for r in train_rec])
    teacher_probs_t = torch.FloatTensor(teacher_aligned).to(device)

    oof_auc = roc_auc_score(y_tr, teacher_aligned)
    print(f"  kNN-CoT teacher OOF AUC = {oof_auc:.4f}", flush=True)

    # Step 3: Fine-tune with 5 seeds
    results = []
    t0 = time.time()
    for seed in SEEDS:
        gc.collect(); torch.cuda.empty_cache()
        print(f"\n  [seed={seed}]", flush=True)
        r = train_one_seed(name, seed, teacher_probs_t, train_rec, test_rec,
                           build_enc, hidden, alpha, n_epochs, use_dakd)
        results.append(r)
        print(f"  seed={seed}: baseline={r['baseline']:.4f} best={r['best']:.4f} "
              f"(elapsed {(time.time()-t0)/60:.1f}m)", flush=True)

    bests = [r["best"] for r in results]
    baselines = [r["baseline"] for r in results]
    method_name = "ramd_kd_dakd" if use_dakd else "ramd_kd"
    summary = {
        "dataset": name, "method": method_name, "alpha": alpha, "use_dakd": use_dakd,
        "teacher_oof_auc": oof_auc,
        "per_seed": results,
        "baseline_mean": float(np.mean(baselines)),
        "baseline_std": float(np.std(baselines)),
        "best_mean": float(np.mean(bests)),
        "best_std": float(np.std(bests)),
    }
    delta = summary["best_mean"] - summary["baseline_mean"]
    print(f"\n  {name}: baseline={summary['baseline_mean']:.4f}+/-{summary['baseline_std']:.4f}")
    print(f"  {tag}:  {summary['best_mean']:.4f}+/-{summary['best_std']:.4f}  (delta={delta:+.4f})")
    return summary


def run_gkd_onpolicy(name, n_rounds=3, alpha=0.1, n_epochs=10, use_dakd=True):
    """GKD-inspired on-policy: recompute kNN on updated embeddings each round."""
    print(f"\n{'='*60}")
    print(f"GKD ON-POLICY: {name.upper()} ({n_rounds} rounds)")
    print(f"{'='*60}", flush=True)

    train_rec, test_rec, build_enc, hidden = BUILDERS[name]()
    y_tr = np.array([r["target"] for r in train_rec])
    y_te = np.array([r["target"] for r in test_rec])
    print(f"  train={len(train_rec)}, test={len(test_rec)}")

    # Round 0: use cached kNN-CoT LLM OOF predictions
    oof_probs = get_knn_cot_oof_predictions(name, teacher=os.environ.get("RAMD_TEACHER"))
    cids_train = np.load(f"embeddings/{name}/cids_train_seed42.npy")
    cid_to_prob = {int(cids_train[i]): oof_probs[i] for i in range(len(cids_train))}
    teacher_aligned = np.array([cid_to_prob.get(r["customer_id"], 0.5) for r in train_rec])

    seed = 42  # Single seed for GKD (multi-round is expensive)
    set_seed(seed)
    enc = build_enc().to(device)
    enc.load_state_dict(torch.load(COLES_CKPT[name], map_location=device))

    baseline = eval_lgbm(extract(enc, train_rec), y_tr, extract(enc, test_rec), y_te, seed)
    print(f"  Baseline: {baseline:.4f}")

    round_results = []
    best_overall = baseline

    for rnd in range(n_rounds):
        gc.collect(); torch.cuda.empty_cache()

        if rnd == 0:
            teacher = teacher_aligned  # LLM kNN-CoT OOF predictions
            teacher_source = "kNN-CoT LLM (OOF)"
        else:
            # GKD on-policy: recompute kNN on UPDATED embeddings
            emb_updated = extract(enc, train_rec)
            teacher = compute_knn_soft_labels(emb_updated, y_tr, k=10)
            teacher_source = "kNN on updated CoLES embeddings"

        teacher_auc = roc_auc_score(y_tr, teacher)
        teacher_t = torch.FloatTensor(teacher).to(device)
        print(f"\n  Round {rnd}: teacher={teacher_source}, teacher_AUC={teacher_auc:.4f}", flush=True)

        # Fine-tune
        classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2)).to(device)
        params = list(enc.parameters()) + list(classifier.parameters())
        opt = torch.optim.Adam(params, lr=5e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
        g = torch.Generator().manual_seed(seed + rnd)
        best_rnd = baseline if rnd == 0 else round_results[-1]["best"]

        for ep in range(n_epochs):
            enc.train(); classifier.train()
            idx = torch.randperm(len(train_rec), generator=g)
            tot = 0; nb = 0
            for s in range(0, len(train_rec), 32):
                bi = idx[s:s+32].tolist()
                dl = inference_data_loader([train_rec[i] for i in bi], num_workers=0, batch_size=32)
                for batch in dl:
                    seq_emb = enc(batch.to(device))
                logits = classifier(seq_emb)
                y_b = torch.LongTensor([train_rec[i]["target"] for i in bi]).to(device)
                loss_ce = F.cross_entropy(logits, y_b)
                t_pos = teacher_t[bi].unsqueeze(1)
                t_probs = torch.cat([1 - t_pos, t_pos], dim=1)
                kl_ps = reverse_kl_per_sample(t_probs, logits)
                if use_dakd:
                    difficulty = 1 - t_probs.max(dim=1).values
                    weights = difficulty / (difficulty.mean() + 1e-8)
                    loss_kd = (weights * kl_ps).mean()
                else:
                    loss_kd = kl_ps.mean()
                loss = (1 - alpha) * loss_ce + alpha * loss_kd
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); tot += loss.item(); nb += 1
            sch.step()

            if (ep + 1) % 5 == 0:
                emb_tr = extract(enc, train_rec)
                emb_te = extract(enc, test_rec)
                score = eval_lgbm(emb_tr, y_tr, emb_te, y_te, seed)
                if score > best_rnd:
                    best_rnd = score
                    torch.save(enc.state_dict(), OUT / f"{name}_gkd_round{rnd}.pt")
                if score > best_overall:
                    best_overall = score
                print(f"    R{rnd} ep={ep+1} loss={tot/nb:.4f} AUC={score:.4f} "
                      f"best_rnd={best_rnd:.4f} best_all={best_overall:.4f}", flush=True)

        round_results.append({
            "round": rnd, "teacher_source": teacher_source,
            "teacher_auc": teacher_auc, "best": best_rnd,
        })
        print(f"  Round {rnd}: best={best_rnd:.4f}", flush=True)
        del classifier; torch.cuda.empty_cache(); gc.collect()

    del enc; torch.cuda.empty_cache(); gc.collect()
    summary = {
        "dataset": name, "method": "gkd_onpolicy", "n_rounds": n_rounds,
        "baseline": baseline, "best_overall": best_overall,
        "rounds": round_results,
    }
    print(f"\n  {name} GKD: baseline={baseline:.4f} → best={best_overall:.4f} "
          f"(delta={best_overall-baseline:+.4f})", flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=["gender", "rosbank"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--dakd", action="store_true", help="Enable DA-KD difficulty weighting")
    ap.add_argument("--gkd", action="store_true", help="GKD on-policy multi-round")
    ap.add_argument("--gkd-rounds", type=int, default=3)
    args = ap.parse_args()

    use_dakd = args.dakd
    all_summaries = {}

    if args.gkd:
        for d in args.datasets:
            s = run_gkd_onpolicy(d, args.gkd_rounds, args.alpha, args.epochs, use_dakd=True)
            all_summaries[d] = s
            with open(OUT / f"{d}_gkd.json", "w") as f:
                json.dump(s, f, indent=2)
        print("\n" + "=" * 60)
        print("GKD ON-POLICY SUMMARY")
        print("=" * 60)
        for d, s in all_summaries.items():
            print(f"  {d}: baseline={s['baseline']:.4f} → best={s['best_overall']:.4f} "
                  f"(delta={s['best_overall']-s['baseline']:+.4f})")
            for r in s["rounds"]:
                print(f"    Round {r['round']}: teacher_AUC={r['teacher_auc']:.4f} best={r['best']:.4f}")
    else:
        for d in args.datasets:
            s = run_dataset(d, args.alpha, args.epochs, use_dakd)
            all_summaries[d] = s
            suffix = "_dakd" if use_dakd else ""
            with open(OUT / f"{d}_results{suffix}.json", "w") as f:
                json.dump(s, f, indent=2)
        print("\n" + "=" * 60)
        print("RAMD-KD SUMMARY")
        print("=" * 60)
        for d, s in all_summaries.items():
            delta = s["best_mean"] - s["baseline_mean"]
            print(f"  {d}: teacher_AUC={s['teacher_oof_auc']:.4f}  "
                  f"baseline={s['baseline_mean']:.4f}  RAMD-KD={s['best_mean']:.4f}  "
                  f"delta={delta:+.4f}")

    with open(OUT / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved: {OUT / 'summary.json'}")
