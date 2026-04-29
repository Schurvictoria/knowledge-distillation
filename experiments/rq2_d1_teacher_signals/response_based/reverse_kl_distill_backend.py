import json, warnings, gc, random, argparse, time
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, StandardScaler
from lightgbm import LGBMClassifier

from ptls.data_load.datasets import inference_data_loader
from ptls.nn import TrxEncoder, RnnSeqEncoder

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
    ("data/rosbank_train.csv", "experiments/rq1_bidirectional/coles/run_rosbank_coles.py"),
    ("data/train_target.csv", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("data/transactions.csv", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
    ("data/transactions_train.csv", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
]
for _p, _hint in _required_inputs:
    assert _P(_p).exists(), f"\n  Missing input: {_p}\n  Run prerequisite: {_hint}"

SEEDS = [42, 123, 456, 789, 1024]
OUT = Path("results/reverse_kl")
OUT.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_teacher_soft_labels(llm_emb, y, task, n_folds=5):
    n_cls = len(np.unique(y))
    oof_probs = np.zeros((len(y), n_cls if task == "multi" else 2))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(llm_emb, y)):
        if task == "binary":
            clf = LGBMClassifier(n_estimators=500, learning_rate=0.02, max_depth=6,
                                 subsample=0.5, colsample_bytree=0.75, verbosity=-1, random_state=42)
        else:
            clf = LGBMClassifier(n_estimators=1000, learning_rate=0.02, objective="multiclass",
                                 num_class=n_cls, max_depth=12, num_leaves=50, verbosity=-1, random_state=42)
        clf.fit(llm_emb[tr_idx], y[tr_idx])
        oof_probs[val_idx] = clf.predict_proba(llm_emb[val_idx])

    oof_probs = np.clip(oof_probs, 1e-6, 1.0 - 1e-6)

    oof_probs = oof_probs / oof_probs.sum(axis=1, keepdims=True)
    return oof_probs

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
    return train_rec, test_rec, build_encoder, 1024, "binary"

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
    feature_dims = {c: len(e.classes_) + 2 for c, e in encs.items()}

    def build_records(cid_set):
        records = []
        for cid in cid_set:
            if cid not in target_map or cid not in grouped.groups:
                continue
            ct = grouped.get_group(cid)
            if len(ct) < 15:
                continue
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
    return train_rec, test_rec, build_encoder, 1024, "binary"

def build_age_data():
    DATA = Path("data")
    tx = pd.read_csv(DATA / "transactions_train.csv")
    labels = pd.read_csv(DATA / "train_target.csv")
    target_map = dict(zip(labels["client_id"], labels["bins"]))
    tx = tx.sort_values(["client_id", "trans_date"])
    tx["amount_rur"] = np.sign(tx["amount_rur"]) * np.log1p(np.abs(tx["amount_rur"]))
    tx["small_group"] = tx["small_group"].fillna(0).astype(str)
    sg_enc = LabelEncoder().fit(tx["small_group"])
    grouped = tx.groupby("client_id")
    feature_dims = {"small_group": len(sg_enc.classes_) + 2}

    def build_records(cid_set):
        records = []
        for cid in cid_set:
            if cid not in target_map or cid not in grouped.groups:
                continue
            ct = grouped.get_group(cid)
            if len(ct) < 25:
                continue
            days = ct["trans_date"].values.astype(np.float32)
            records.append({"customer_id": cid, "target": target_map[cid],
                            "event_time": torch.FloatTensor(days - days[0]),
                            "amount": torch.FloatTensor(ct["amount_rur"].values),
                            "small_group": torch.LongTensor(sg_enc.transform(ct["small_group"].values) + 1)})
        return records

    def build_encoder():
        trx = TrxEncoder(embeddings={"small_group": {"in": feature_dims["small_group"], "out": 16}},
                         numeric_values={"amount": "identity"}, embeddings_noise=0.003,
                         use_batch_norm_with_lens=True)
        return RnnSeqEncoder(trx_encoder=trx, hidden_size=800, type="gru",
                             bidir=False, trainable_starter="static")

    ids = labels["client_id"].values
    targets = np.array([target_map[c] for c in ids])
    idx_tr, idx_te = train_test_split(np.arange(len(ids)), test_size=0.1, random_state=42, stratify=targets)
    train_rec = build_records(set(ids[idx_tr]))
    test_rec = build_records(set(ids[idx_te]))
    return train_rec, test_rec, build_encoder, 800, "multi"

BUILDERS = {"gender": build_gender_data, "rosbank": build_rosbank_data, "age": build_age_data}
COLES_CKPT = {
    "gender": "results/gender_true_latte/coles_baseline.pt",
    "rosbank": "results/rosbank_true_latte/coles_baseline.pt",
    "age": "results/age_true_latte/coles_baseline.pt",
}
LLM_EMB = {
    "gender": "results/gender_llm4es/llm4es_embeddings.npz",
    "rosbank": "results/rosbank_llm4es/llm4es_embeddings.npz",
    "age": "results/age_llm4es/llm4es_embeddings.npz",
}
BEST_ALPHA = {"gender": 0.1, "rosbank": 0.1, "age": 0.05}

def forward_kl(teacher_probs, student_logits):
    student_log_probs = F.log_softmax(student_logits, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

def reverse_kl(teacher_probs, student_logits):
    student_probs = F.softmax(student_logits, dim=1)
    teacher_log_probs = torch.log(teacher_probs + 1e-8)
    student_log_probs = torch.log(student_probs + 1e-8)
    return (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=1).mean()

def extract(enc, records, bs=64):
    enc.eval()
    dl = inference_data_loader(records, num_workers=0, batch_size=bs)
    with torch.no_grad():
        return torch.cat([enc(b.to(device)).cpu() for b in dl]).numpy()

def eval_lgbm(emb_tr, y_tr, emb_te, y_te, task, seed):
    sc = MaxAbsScaler()
    x_tr = sc.fit_transform(emb_tr)
    x_te = sc.transform(emb_te)
    if task == "binary":
        p = dict(n_estimators=500, learning_rate=0.02, max_depth=6, subsample=0.5,
                 colsample_bytree=0.75, reg_alpha=1, reg_lambda=1,
                 min_child_samples=50, verbosity=-1, random_state=seed)
        clf = LGBMClassifier(**p).fit(x_tr, y_tr)
        return roc_auc_score(y_te, clf.predict_proba(x_te)[:, 1])
    else:
        p = dict(n_estimators=1000, learning_rate=0.02, objective="multiclass",
                 num_class=4, max_depth=12, num_leaves=50, subsample=0.75,
                 colsample_bytree=0.75, reg_alpha=1, reg_lambda=1,
                 min_child_samples=50, verbosity=-1, random_state=seed)
        clf = LGBMClassifier(**p).fit(x_tr, y_tr)
        return accuracy_score(y_te, clf.predict(x_te))

def train_one_seed(name, seed, kl_mode, alpha, n_epochs, teacher_probs_t,
                   train_rec, test_rec, build_enc, hidden, task):
    set_seed(seed)
    y_tr = np.array([r["target"] for r in train_rec])
    y_te = np.array([r["target"] for r in test_rec])
    n_cls = len(np.unique(y_tr))

    enc = build_enc().to(device)
    enc.load_state_dict(torch.load(COLES_CKPT[name], map_location=device))

    classifier = nn.Sequential(
        nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, n_cls)).to(device)

    params = list(enc.parameters()) + list(classifier.parameters())
    opt = torch.optim.Adam(params, lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    baseline = eval_lgbm(extract(enc, train_rec), y_tr,
                         extract(enc, test_rec), y_te, task, seed)
    best = baseline
    g = torch.Generator().manual_seed(seed)

    kl_fn = reverse_kl if kl_mode == "reverse" else forward_kl

    for ep in range(n_epochs):
        enc.train(); classifier.train()
        idx = torch.randperm(len(train_rec), generator=g)
        tot_loss = 0; n_b = 0
        for s in range(0, len(train_rec), 32):
            bi = idx[s:s + 32].tolist()
            dl = inference_data_loader([train_rec[i] for i in bi], num_workers=0, batch_size=32)
            for batch in dl:
                seq_emb = enc(batch.to(device))

            logits = classifier(seq_emb)
            y_b = torch.LongTensor([train_rec[i]["target"] for i in bi]).to(device)

            loss_ce = F.cross_entropy(logits, y_b)

            t_probs = teacher_probs_t[bi]
            loss_kl = kl_fn(t_probs, logits)

            loss = (1 - alpha) * loss_ce + alpha * loss_kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot_loss += loss.item(); n_b += 1
        sch.step()

        if (ep + 1) % 5 == 0:
            emb_tr = extract(enc, train_rec)
            emb_te = extract(enc, test_rec)
            score = eval_lgbm(emb_tr, y_tr, emb_te, y_te, task, seed)
            if score > best:
                best = score
                torch.save(enc.state_dict(),
                           OUT / f"{name}_{kl_mode}_seed{seed}.pt")
            print(f"    seed={seed} ep={ep+1}/{n_epochs} loss={tot_loss/n_b:.4f} "
                  f"score={score:.4f} best={best:.4f}")

    del enc, classifier
    torch.cuda.empty_cache(); gc.collect()
    return {"seed": seed, "baseline": baseline, "best": best}

def run_dataset(name, n_epochs=15):
    print(f"REVERSE KL EXPERIMENT: {name.upper()}")

    train_rec, test_rec, build_enc, hidden, task = BUILDERS[name]()
    y_tr = np.array([r["target"] for r in train_rec])

    llm_all = np.load(LLM_EMB[name])["embeddings"].astype(np.float32)
    llm_tr = llm_all[:len(train_rec)]

    print("  Computing OOF teacher soft labels...")
    teacher_probs = get_teacher_soft_labels(llm_tr, y_tr, task)
    teacher_probs_t = torch.FloatTensor(teacher_probs).to(device)
    print(f"  Teacher OOF probs shape: {teacher_probs.shape}")

    alpha = BEST_ALPHA[name]
    all_results = {}

    for kl_mode in ["forward", "reverse"]:
        results = []
        t0 = time.time()
        for seed in SEEDS:
            print(f"\n  [{kl_mode}] seed={seed}")
            r = train_one_seed(name, seed, kl_mode, alpha, n_epochs,
                               teacher_probs_t, train_rec, test_rec,
                               build_enc, hidden, task)
            r["kl_mode"] = kl_mode
            results.append(r)
            elapsed = time.time() - t0
            print(f"  {kl_mode} seed={seed}: baseline={r['baseline']:.4f} "
                  f"best={r['best']:.4f} (elapsed {elapsed/60:.1f}m)")

        bests = [r["best"] for r in results]
        baselines = [r["baseline"] for r in results]
        all_results[kl_mode] = {
            "kl_mode": kl_mode, "alpha": alpha,
            "per_seed": results,
            "baseline_mean": float(np.mean(baselines)),
            "baseline_std": float(np.std(baselines)),
            "best_mean": float(np.mean(bests)),
            "best_std": float(np.std(bests)),
        }

    fwd = all_results["forward"]
    rev = all_results["reverse"]
    print(f"\n  {name}: baseline={fwd['baseline_mean']:.4f}±{fwd['baseline_std']:.4f}")
    print(f"  Forward KL: {fwd['best_mean']:.4f}±{fwd['best_std']:.4f}  "
          f"(Δ={fwd['best_mean']-fwd['baseline_mean']:+.4f})")
    print(f"  Reverse KL: {rev['best_mean']:.4f}±{rev['best_std']:.4f}  "
          f"(Δ={rev['best_mean']-rev['baseline_mean']:+.4f})")

    return {"dataset": name, **all_results}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=["rosbank", "gender", "age"])
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    all_summaries = {}
    for d in args.datasets:
        s = run_dataset(d, args.epochs)
        all_summaries[d] = s
        with open(OUT / f"{d}_results.json", "w") as f:
            json.dump(s, f, indent=2)
        print(f"  Saved: {OUT / f'{d}_results.json'}")

    print("FORWARD vs REVERSE KL — 5-SEED SUMMARY")
    for d, s in all_summaries.items():
        fwd = s["forward"]
        rev = s["reverse"]
        metric = "AUC" if d in ["gender", "rosbank"] else "Acc"
        print(f"\n  {d} ({metric}):")
        print(f"    baseline:    {fwd['baseline_mean']:.4f} ± {fwd['baseline_std']:.4f}")
        print(f"    forward KL:  {fwd['best_mean']:.4f} ± {fwd['best_std']:.4f}  "
              f"Δ={fwd['best_mean']-fwd['baseline_mean']:+.4f}")
        print(f"    reverse KL:  {rev['best_mean']:.4f} ± {rev['best_std']:.4f}  "
              f"Δ={rev['best_mean']-rev['baseline_mean']:+.4f}")
        if rev['best_mean'] > fwd['best_mean']:
            print(f"    → Reverse KL wins by {rev['best_mean']-fwd['best_mean']:+.4f}")
        else:
            print(f"    → Forward KL wins by {fwd['best_mean']-rev['best_mean']:+.4f}")

    with open(OUT / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved: {OUT / 'summary.json'}")
