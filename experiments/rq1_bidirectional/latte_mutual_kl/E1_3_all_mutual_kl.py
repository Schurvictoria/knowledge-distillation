#!/usr/bin/env python3
"""
True Bidirectional Distillation on Rosbank + Age.
Same approach as Gender (both CoLES GRU + LLM LoRA fine-tuned simultaneously).
Uses best config from Gender: cls=0.5, contrast=0.3, mutual=0.2.
Loads checkpoints, saves results.
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
    ("data/rosbank_train.csv", "experiments/rq1_bidirectional/coles/run_rosbank_coles.py"),
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
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from lightgbm import LGBMClassifier
from ptls.data_load.datasets import MemoryMapDataset, inference_data_loader
from ptls.nn import TrxEncoder, RnnSeqEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LGBM_BIN = dict(n_estimators=500, learning_rate=0.02, max_depth=6, subsample=0.5,
                colsample_bytree=0.75, reg_alpha=1, reg_lambda=1, min_child_samples=50, verbosity=-1)
LGBM_MULTI = dict(n_estimators=1000, learning_rate=0.02, objective="multiclass", num_class=4,
                  max_depth=12, num_leaves=50, subsample=0.75, colsample_bytree=0.75,
                  reg_alpha=1, reg_lambda=1, min_child_samples=50, verbosity=-1)

MCC_GROUPS = {range(1,1500):"Agriculture",range(4000,4800):"Transportation",range(5000,5600):"Retail",
              range(5600,5700):"Clothing",range(5800,5900):"Restaurants",range(6000,7000):"Financial",
              range(7500,7600):"Auto",range(8000,8100):"Medical",range(8200,8300):"Education"}

def mcc_cat(mcc):
    try: mcc=int(mcc)
    except: return "Other"
    for r,n in MCC_GROUPS.items():
        if mcc in r: return n
    return "Other"


def run_bidirectional(dataset_name):
    """Run true bidirectional on one dataset."""
    print(f"\n{'='*60}")
    print(f"TRUE BIDIRECTIONAL: {dataset_name.upper()}")
    print(f"{'='*60}")

    OUT = Path(f"results/{dataset_name}_true_bidirectional")
    OUT.mkdir(parents=True, exist_ok=True)
    DATA = Path("data")

    # ---- Load data ----
    if dataset_name == "rosbank":
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
        EMB_DIMS = {"mcc_code":24,"channel_type":4,"currency":4,"trx_category":4}
        feature_dims = {c: len(e.classes_)+2 for c,e in encs.items()}
        hidden, rnn_type = 1024, "lstm"
        lgbm_p, metric_fn, metric_name = LGBM_BIN, roc_auc_score, "AUC"

        def build_rec(cid):
            ct = grouped.get_group(cid)
            if len(ct) < 15: return None
            dt_vals = ct["dt"].values
            days = (dt_vals - dt_vals[0]) / np.timedelta64(1, "D")
            rec = {"customer_id": cid, "target": target_map[cid],
                   "event_time": torch.FloatTensor(days.astype(np.float32)),
                   "amount": torch.FloatTensor(ct["amount"].values)}
            for col, enc in encs.items():
                rec[col] = torch.LongTensor(enc.transform(ct[col].values) + 1)
            return rec

        def serialize(cid):
            if cid not in grouped.groups: return "No txns."
            ct = grouped.get_group(cid).tail(30)
            lines = [f"Client ({len(ct)} txns):"]
            for _, r in ct.iterrows():
                d = "spent" if r["amount"]<0 else "received"
                lines.append(f"{r['dt'].strftime('%Y-%m-%d')}: {d} {abs(r['amount']):.0f} at {mcc_cat(r['mcc_code'])}")
            return "\n".join(lines)

        def build_encoder():
            embs = {c:{"in":feature_dims[c],"out":EMB_DIMS[c]} for c in feature_dims if c in EMB_DIMS}
            trx = TrxEncoder(embeddings=embs, numeric_values={"amount":"identity"},
                              embeddings_noise=0.0003, use_batch_norm_with_lens=True)
            return RnnSeqEncoder(trx_encoder=trx, hidden_size=hidden, type=rnn_type, bidir=False, trainable_starter="static")

        ids = labels_df["customer_id"].values

    else:  # age
        tx = pd.read_csv(DATA / "transactions_train.csv")
        labels = pd.read_csv(DATA / "train_target.csv")
        target_map = dict(zip(labels["client_id"], labels["bins"]))
        tx = tx.sort_values(["client_id", "trans_date"])
        tx["amount_rur"] = np.sign(tx["amount_rur"]) * np.log1p(np.abs(tx["amount_rur"]))
        tx["small_group"] = tx["small_group"].fillna(0).astype(str)
        sg_enc = LabelEncoder().fit(tx["small_group"])
        grouped = tx.groupby("client_id")
        feature_dims = {"small_group": len(sg_enc.classes_) + 2}
        hidden, rnn_type = 800, "gru"
        lgbm_p, metric_name = LGBM_MULTI, "acc"
        metric_fn = accuracy_score

        def build_rec(cid):
            ct = grouped.get_group(cid)
            if len(ct) < 25: return None
            days = ct["trans_date"].values.astype(np.float32)
            return {"customer_id": cid, "target": target_map[cid],
                    "event_time": torch.FloatTensor(days - days[0]),
                    "amount": torch.FloatTensor(ct["amount_rur"].values),
                    "small_group": torch.LongTensor(sg_enc.transform(ct["small_group"].values) + 1)}

        def serialize(cid):
            if cid not in grouped.groups: return "No txns."
            ct = grouped.get_group(cid).tail(30)
            lines = [f"Client ({len(ct)} txns):"]
            for _, r in ct.iterrows():
                d = "spent" if r["amount_rur"]<0 else "received"
                lines.append(f"Day {int(r['trans_date'])}: {d} {abs(r['amount_rur']):.0f} at {mcc_cat(r['small_group'])}")
            return "\n".join(lines)

        def build_encoder():
            trx = TrxEncoder(embeddings={"small_group":{"in":feature_dims["small_group"],"out":16}},
                              numeric_values={"amount":"identity"}, embeddings_noise=0.003, use_batch_norm_with_lens=True)
            return RnnSeqEncoder(trx_encoder=trx, hidden_size=hidden, type=rnn_type, bidir=False, trainable_starter="static")

        ids = labels["client_id"].values

    targets = np.array([target_map[c] for c in ids])
    idx_tr, idx_te = train_test_split(np.arange(len(ids)), test_size=0.1, random_state=42, stratify=targets)
    train_ids, test_ids = set(ids[idx_tr]), set(ids[idx_te])

    train_rec_full = [r for r in (build_rec(c) for c in train_ids if c in grouped.groups) if r is not None]
    test_rec = [r for r in (build_rec(c) for c in test_ids if c in grouped.groups) if r is not None]
    # Honest val split (10% from train, seed=42, stratified)
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

    # Load CoLES checkpoint
    COLES_CKPT = Path(f"results/{dataset_name}_true_latte/coles_baseline.pt")
    seq_encoder = build_encoder().to(device)
    seq_encoder.load_state_dict(torch.load(COLES_CKPT, map_location=device))
    print(f"  CoLES loaded from {COLES_CKPT}")

    # Load LLM
    LLM_CKPT = Path(f"results/{dataset_name}_llm4es/checkpoints/llm4es_lora")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", quantization_config=bnb, device_map="auto")
    llm = PeftModel.from_pretrained(llm, str(LLM_CKPT))
    hidden_llm = llm.config.hidden_size
    print(f"  LLM loaded, VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    # Heads
    proj_s = nn.Sequential(nn.Linear(hidden,256),nn.ReLU(),nn.Linear(256,128)).to(device)
    proj_t = nn.Sequential(nn.Linear(hidden_llm,256),nn.ReLU(),nn.Linear(256,128)).to(device)
    n_cls = 4 if dataset_name == "age" else 1
    cls_s = nn.Sequential(nn.Linear(hidden,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,n_cls)).to(device)
    cls_t = nn.Sequential(nn.Linear(hidden_llm,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,n_cls)).to(device)

    def get_llm_emb(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.set_grad_enabled(llm.training):
            out = llm(**inputs, output_hidden_states=True)
            h = torch.stack(out.hidden_states[-4:]).mean(0)
            mask = inputs["attention_mask"][0].unsqueeze(-1).float()
            return (h[0]*mask).sum(0)/mask.sum(0)

    def _extract(records):
        seq_encoder.eval()
        dl = inference_data_loader(records, num_workers=0, batch_size=64)
        with torch.no_grad():
            return torch.cat([seq_encoder(b.to(device)).cpu() for b in dl]).numpy()

    def eval_model():
        """Return (val_score, test_score) — selection on val only."""
        sc = MaxAbsScaler()
        lgbm = LGBMClassifier(**lgbm_p, random_state=42)
        etr = _extract(train_rec); evl = _extract(val_rec); ete = _extract(test_rec)
        lgbm.fit(sc.fit_transform(etr), y_train)
        if dataset_name == "age":
            v = accuracy_score(y_val, lgbm.predict(sc.transform(evl)))
            t = accuracy_score(y_test, lgbm.predict(sc.transform(ete)))
        else:
            v = roc_auc_score(y_val, lgbm.predict_proba(sc.transform(evl))[:, 1])
            t = roc_auc_score(y_test, lgbm.predict_proba(sc.transform(ete))[:, 1])
        return v, t

    baseline_val, baseline_test = eval_model()
    print(f"  Baseline: val={baseline_val:.4f} test={baseline_test:.4f}")

    # Best config from Gender
    alpha_cls, alpha_con, alpha_mut = 0.5, 0.3, 0.2

    trainable = list(seq_encoder.parameters()) + list(proj_s.parameters()) + list(proj_t.parameters()) + \
                list(cls_s.parameters()) + list(cls_t.parameters())
    for _, p in llm.named_parameters():
        if p.requires_grad: trainable.append(p)

    opt = torch.optim.Adam(trainable, lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss() if dataset_name == "age" else nn.BCEWithLogitsLoss()
    best_val = baseline_val
    best_test = baseline_test

    for ep in range(10):
        seq_encoder.train(); llm.train()
        for m in [proj_s,proj_t,cls_s,cls_t]: m.train()
        idx = torch.randperm(len(train_rec))
        tot, nb = 0, 0

        for s in range(0, len(train_rec), 16):
            bi = idx[s:s+16].tolist()
            br = [train_rec[i] for i in bi]
            bt = [train_texts[i] for i in bi]
            if dataset_name == "age":
                yb = torch.LongTensor([r["target"] for r in br]).to(device)
            else:
                yb = torch.FloatTensor([r["target"] for r in br]).to(device)

            dl = inference_data_loader(br, num_workers=0, batch_size=32)
            for batch in dl:
                se = seq_encoder(batch.to(device))
            te = torch.stack([get_llm_emb(t) for t in bt])

            zs = F.normalize(proj_s(se),dim=1)
            zt = F.normalize(proj_t(te),dim=1)
            ls = cls_s(se).squeeze(-1) if n_cls==1 else cls_s(se)
            lt = cls_t(te).squeeze(-1) if n_cls==1 else cls_t(te)

            lc_s = loss_fn(ls, yb)
            lc_t = loss_fn(lt, yb)

            lo = zs @ zt.T / 0.07
            la = torch.arange(len(zs), device=device)
            l_con = (F.cross_entropy(lo,la)+F.cross_entropy(lo.T,la))/2

            if n_cls == 1:
                ps, pt_ = torch.sigmoid(ls), torch.sigmoid(lt)
                l_mut = (F.binary_cross_entropy(ps,pt_.detach())+F.binary_cross_entropy(pt_,ps.detach()))/2
            else:
                ps, pt_ = F.softmax(ls,dim=1), F.softmax(lt,dim=1)
                l_mut = (F.kl_div(ps.log(),pt_.detach(),reduction='batchmean')+
                         F.kl_div(pt_.log(),ps.detach(),reduction='batchmean'))/2

            loss = alpha_cls*(lc_s+lc_t)/2 + alpha_con*l_con + alpha_mut*l_mut
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1

        val_score, test_score = eval_model()
        if val_score > best_val:
            best_val = val_score
            best_test = test_score
            torch.save(seq_encoder.state_dict(), OUT / "coles_bidir_best.pt")
        print(f"  ep {ep+1}: loss={tot/nb:.4f}, "
              f"val={val_score:.4f} test={test_score:.4f} "
              f"best_val={best_val:.4f} best_test={best_test:.4f}")

    results = {"baseline": baseline_test, "bidirectional_best": best_test,
               "baseline_val": baseline_val, "best_val": best_val}
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  {dataset_name}: baseline={baseline_test:.4f}, bidir={best_test:.4f} "
          f"({'+' if best_test>baseline_test else ''}{best_test-baseline_test:.4f})")

    del llm, tokenizer; torch.cuda.empty_cache(); gc.collect()
    return results


# ---- Run both ----
all_results = {}
for ds in ["rosbank", "age"]:
    all_results[ds] = run_bidirectional(ds)

print("\n" + "=" * 60)
print("ALL RESULTS")
for ds, r in all_results.items():
    d = r["bidirectional_best"] - r["baseline"]
    print(f"  {ds}: baseline={r['baseline']:.4f}, bidir={r['bidirectional_best']:.4f} ({'+' if d>=0 else ''}{d:.4f})")
