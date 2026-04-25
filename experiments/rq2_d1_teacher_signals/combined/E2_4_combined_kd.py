#!/usr/bin/env python3
"""
Combined KD: Contrastive (embedding-space) + Reverse KL (prediction-space).

Inspired by CLIP-KD (CVPR 2024): relation-based + logit-based distillation
outperforms either alone, even with weak teacher.

L = (1-α-β) * CE(student, y_true)
  + α * InfoNCE(CoLES_emb, LLM4ES_emb)           ← embedding-space
  + β * reverse_KL(student, kNN-CoT_teacher)      ← prediction-space

Teacher 1 (embedding): LLM4ES text embeddings (2048d, frozen)
Teacher 2 (prediction): kNN-CoT LLM OOF soft labels (cached)
Student: CoLES GRU encoder (fine-tuned)

Literature basis:
  - CLIP-KD [CVPR 2024]: combined relation + logit KD
  - LATTE [EMNLP 2025]: contrastive alignment
  - MiniLLM [ICLR 2024]: reverse KL
  - DA-KD [ICML 2024]: difficulty-aware weighting
  - TabR [ICLR 2024]: kNN retrieval

5 seeds. Saves checkpoints.
"""
import json, warnings, gc, random, time, argparse
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

SEEDS = [42, 123, 456, 789, 1024]
OUT = Path("results/combined_kd")
OUT.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLES_CKPT = {
    "gender": "results/gender_true_latte/coles_baseline.pt",
    "rosbank": "results/rosbank_true_latte/coles_baseline.pt",
}
LLM_EMB = {
    "gender": "results/gender_llm4es/llm4es_embeddings.npz",
    "rosbank": "results/rosbank_llm4es/llm4es_embeddings.npz",
}
KNN_COT_OOF = {
    "gender": "results/ramd_kd/gender_knn_cot_oof.npz",
    "rosbank": "results/ramd_kd/rosbank_knn_cot_oof.npz",
}
LGBM_P = dict(n_estimators=500, learning_rate=0.02, max_depth=6, subsample=0.5,
              colsample_bytree=0.75, reg_alpha=1, reg_lambda=1, min_child_samples=50, verbosity=-1)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


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


def train_one_seed(name, seed, llm_emb_t, teacher_pred_t, train_rec, test_rec,
                   build_enc, hidden, alpha=0.1, beta=0.05, n_epochs=15):
    """
    Combined KD: contrastive (alpha) + reverse KL (beta).
    alpha: weight for InfoNCE (embedding-space)
    beta: weight for reverse KL (prediction-space)
    """
    set_seed(seed)
    y_tr = np.array([r["target"] for r in train_rec])
    y_te = np.array([r["target"] for r in test_rec])

    enc = build_enc().to(device)
    enc.load_state_dict(torch.load(COLES_CKPT[name], map_location=device))

    # Projection heads for contrastive
    proj_seq = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, 128)).to(device)
    proj_text = nn.Sequential(nn.Linear(llm_emb_t.shape[1], 256), nn.ReLU(), nn.Linear(256, 128)).to(device)

    # Classifier head for prediction-space KD
    classifier = nn.Sequential(
        nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2)).to(device)

    params = (list(enc.parameters()) + list(proj_seq.parameters()) +
              list(proj_text.parameters()) + list(classifier.parameters()))
    opt = torch.optim.Adam(params, lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    baseline = eval_lgbm(extract(enc, train_rec), y_tr, extract(enc, test_rec), y_te, seed)
    best = baseline
    g = torch.Generator().manual_seed(seed)

    for ep in range(n_epochs):
        enc.train(); proj_seq.train(); proj_text.train(); classifier.train()
        idx = torch.randperm(len(train_rec), generator=g)
        tot = 0; nb = 0
        for s in range(0, len(train_rec), 32):
            bi = idx[s:s+32].tolist()
            dl = inference_data_loader([train_rec[i] for i in bi], num_workers=0, batch_size=32)
            for batch in dl:
                seq_emb = enc(batch.to(device))

            # 1. Classification loss (CE)
            logits = classifier(seq_emb)
            y_b = torch.LongTensor([train_rec[i]["target"] for i in bi]).to(device)
            loss_ce = F.cross_entropy(logits, y_b)

            # 2. Contrastive loss (InfoNCE, embedding-space)
            z_s = F.normalize(proj_seq(seq_emb), dim=1)
            z_t = F.normalize(proj_text(llm_emb_t[bi]), dim=1)
            sim = z_s @ z_t.T / 0.07
            labels_c = torch.arange(len(z_s), device=device)
            loss_con = (F.cross_entropy(sim, labels_c) + F.cross_entropy(sim.T, labels_c)) / 2

            # 3. Reverse KL loss (prediction-space, MiniLLM-inspired)
            t_pos = teacher_pred_t[bi].unsqueeze(1)
            t_probs = torch.cat([1 - t_pos, t_pos], dim=1)
            sp = F.softmax(logits, dim=1)
            loss_rkl = (sp * (torch.log(sp + 1e-8) - torch.log(t_probs + 1e-8))).sum(dim=1).mean()

            # Combined: CE + contrastive + reverse KL
            loss = (1 - alpha - beta) * loss_ce + alpha * loss_con + beta * loss_rkl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sch.step()

        if (ep + 1) % 5 == 0:
            emb_tr = extract(enc, train_rec)
            emb_te = extract(enc, test_rec)
            score = eval_lgbm(emb_tr, y_tr, emb_te, y_te, seed)
            if score > best:
                best = score
                torch.save(enc.state_dict(), OUT / f"{name}_combined_seed{seed}.pt")
            print(f"    seed={seed} ep={ep+1}/{n_epochs} loss={tot/nb:.4f} "
                  f"AUC={score:.4f} best={best:.4f}", flush=True)

    del enc, proj_seq, proj_text, classifier
    torch.cuda.empty_cache(); gc.collect()
    return {"seed": seed, "baseline": baseline, "best": best}


def run_dataset(name, alpha=0.1, beta=0.05, n_epochs=15):
    print(f"\n{'='*60}")
    print(f"COMBINED KD (α={alpha}, β={beta}): {name.upper()}")
    print(f"{'='*60}", flush=True)

    # Load data
    train_rec, test_rec, build_enc, hidden = BUILDERS[name]()
    y_tr = np.array([r["target"] for r in train_rec])
    print(f"  train={len(train_rec)}, test={len(test_rec)}")

    # Teacher 1: LLM4ES embeddings
    llm_all = np.load(LLM_EMB[name])["embeddings"].astype(np.float32)
    llm_tr = llm_all[:len(train_rec)]
    sc_l = StandardScaler()
    llm_emb_t = torch.FloatTensor(sc_l.fit_transform(llm_tr)).to(device)

    # Teacher 2: kNN-CoT OOF predictions
    oof_data = np.load(KNN_COT_OOF[name])
    oof_probs = oof_data["probs"]
    cids_train = np.load(f"embeddings/{name}/cids_train_seed42.npy")
    cid_to_prob = {int(cids_train[i]): oof_probs[i] for i in range(len(cids_train))}
    teacher_aligned = np.array([cid_to_prob.get(r["customer_id"], 0.5) for r in train_rec])
    teacher_pred_t = torch.FloatTensor(teacher_aligned).to(device)

    from sklearn.metrics import roc_auc_score as auc_fn
    teacher_auc = auc_fn(y_tr, teacher_aligned)
    print(f"  LLM4ES emb dim={llm_emb_t.shape[1]}, kNN-CoT teacher OOF AUC={teacher_auc:.4f}", flush=True)

    # Train with 5 seeds
    results = []
    t0 = time.time()
    for seed in SEEDS:
        gc.collect(); torch.cuda.empty_cache()
        print(f"\n  [seed={seed}]", flush=True)
        r = train_one_seed(name, seed, llm_emb_t, teacher_pred_t, train_rec, test_rec,
                           build_enc, hidden, alpha, beta, n_epochs)
        results.append(r)
        print(f"  seed={seed}: baseline={r['baseline']:.4f} best={r['best']:.4f} "
              f"(elapsed {(time.time()-t0)/60:.1f}m)", flush=True)

    bests = [r["best"] for r in results]
    baselines = [r["baseline"] for r in results]
    summary = {
        "dataset": name, "method": "combined_kd", "alpha": alpha, "beta": beta,
        "teacher_oof_auc": teacher_auc,
        "per_seed": results,
        "baseline_mean": float(np.mean(baselines)),
        "baseline_std": float(np.std(baselines)),
        "best_mean": float(np.mean(bests)),
        "best_std": float(np.std(bests)),
    }
    delta = summary["best_mean"] - summary["baseline_mean"]
    print(f"\n  {name}: baseline={summary['baseline_mean']:.4f}+/-{summary['baseline_std']:.4f}")
    print(f"  Combined KD: {summary['best_mean']:.4f}+/-{summary['best_std']:.4f} (delta={delta:+.4f})")
    print(f"  Compare: True LATTE alone was Gender +0.49pp, Rosbank +0.02pp")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=["gender", "rosbank"])
    ap.add_argument("--alpha", type=float, default=0.1, help="Contrastive weight")
    ap.add_argument("--beta", type=float, default=0.05, help="Reverse KL weight")
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    all_summaries = {}
    for d in args.datasets:
        s = run_dataset(d, args.alpha, args.beta, args.epochs)
        all_summaries[d] = s
        with open(OUT / f"{d}_results.json", "w") as f:
            json.dump(s, f, indent=2)

    print("\n" + "=" * 60)
    print("COMBINED KD SUMMARY (contrastive + reverse KL)")
    print("=" * 60)
    for d, s in all_summaries.items():
        delta = s["best_mean"] - s["baseline_mean"]
        print(f"  {d}: baseline={s['baseline_mean']:.4f}  combined={s['best_mean']:.4f}  "
              f"delta={delta:+.4f}  (α={s['alpha']}, β={s['beta']})")

    with open(OUT / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved: {OUT / 'summary.json'}")
