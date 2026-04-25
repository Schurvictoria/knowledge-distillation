#!/usr/bin/env python3
"""
Fill the strategy × enrichment matrix for RQ2 Direction 2.

Matrix:
              | None   | + SHAP | + kNN  | + Both |
  Zero-shot   | 0.498  |   ?    |   ?    |   ?    |
  Few-shot    | 0.578  |   ?    |   ?    |   ?    |
  CoT         |   ?    | 0.606  | 0.762  | 0.745  |

Fills the 7 missing cells. Uses Qwen2.5-7B-Instruct locally.
Runs on Gender dataset (test set, 818 clients).
"""
import json, warnings, time
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier
import shap

OUTPUT_DIR = Path("results/strategy_matrix")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")

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

# ---- Load LLM ----
print("Loading LLM...", flush=True)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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
    ("embeddings/gender/emb_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
    ("embeddings/gender/emb_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
    ("embeddings/gender/y_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
    ("embeddings/gender/y_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_gender_coles.py"),
]
for _p, _hint in _required_inputs:
    assert _P(_p).exists(), f"\n  Missing input: {_p}\n  Run prerequisite: {_hint}"
# ---- end input check ----



MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                              device_map="auto", trust_remote_code=True)
model.eval()


def predict_binary(messages, pos_tokens, neg_tokens):
    def get_ids(tokens):
        ids = set()
        for t in tokens:
            e = tokenizer.encode(t, add_special_tokens=False)
            if e: ids.add(e[0])
        return list(ids)
    pi, ni = get_ids(pos_tokens), get_ids(neg_tokens)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    all_ids = pi + ni
    probs = torch.softmax(logits[all_ids].float(), dim=0)
    pp = probs[:len(pi)].sum().item()
    pn = probs[len(pi):].sum().item()
    total = pp + pn
    del inputs
    return pp / total if total > 1e-8 else 0.5


def run_gender():
    print("\n" + "="*60)
    print("STRATEGY × ENRICHMENT MATRIX: GENDER")
    print("="*60, flush=True)

    # Load data
    cids_train = np.load("embeddings/gender/cids_train_seed42.npy")
    cids_test = np.load("embeddings/gender/cids_test_seed42.npy")
    y_train = np.load("embeddings/gender/y_train_seed42.npy")
    y_test = np.load("embeddings/gender/y_test_seed42.npy")
    coles_train = np.load("embeddings/gender/emb_train_seed42.npy")
    coles_test = np.load("embeddings/gender/emb_test_seed42.npy")

    tx = pd.read_csv(DATA_DIR / "transactions.csv")
    labels = pd.read_csv(DATA_DIR / "gender_train.csv")
    tx = tx[tx["customer_id"].isin(labels["customer_id"])].copy()
    target_map = dict(zip(labels["customer_id"], labels["gender"]))
    grouped = tx.groupby("customer_id")
    pos_tokens = ["male", " male", "Male", " Male"]
    neg_tokens = ["female", " female", "Female", " Female"]
    pos_label, neg_label = "male", "female"

    def serialize(cid):
        if cid not in grouped.groups: return "No txns."
        ct = grouped.get_group(cid)
        n = len(ct)
        amt = np.abs(ct["amount"].values)
        cats = ct["mcc_code"].apply(mcc_cat).value_counts()
        top = ", ".join(f"{c} ({n_})" for c, n_ in cats.head(6).items())
        return f"Client with {n} txns. Avg amount: {amt.mean():.0f}. Categories: {top}."

    # Agg features for SHAP
    def agg_feat(cids):
        recs = []
        for cid in cids:
            if cid not in grouped.groups:
                recs.append({"n":0,"mean":0,"std":0,"med":0,"nmcc":0})
                continue
            ct = grouped.get_group(cid)
            a = ct["amount"].values
            recs.append({"n":len(ct),"mean":np.abs(a).mean(),"std":np.abs(a).std(),
                         "med":np.median(np.abs(a)),"nmcc":ct["mcc_code"].nunique()})
        return np.array(pd.DataFrame(recs).values, dtype=np.float32)

    feat_names = ["n_tx", "mean_amt", "std_amt", "median_amt", "n_mcc"]

    # SHAP context
    print("  Computing SHAP...", flush=True)
    X_tr = agg_feat(cids_train)
    X_te = agg_feat(cids_test)
    xgb = XGBClassifier(n_estimators=300, max_depth=6, random_state=42, verbosity=0)
    xgb.fit(X_tr, y_train)
    test_preds_prob = xgb.predict_proba(X_te)[:, 1]
    explainer = shap.TreeExplainer(xgb)
    sv = explainer.shap_values(X_te)

    shap_ctx = {}
    for i, cid in enumerate(cids_test):
        pred = pos_label if test_preds_prob[i] > 0.5 else neg_label
        conf = test_preds_prob[i] if test_preds_prob[i] > 0.5 else 1 - test_preds_prob[i]
        svi = sv[i] if sv.ndim == 2 else sv[1][i]
        top_idx = np.argsort(np.abs(svi))[::-1][:3]
        factors = [f"{feat_names[int(j)]}={X_te[i,int(j)]:.0f} ({'supports' if svi[int(j)]>0 else 'against'} {pos_label})"
                   for j in top_idx]
        shap_ctx[cid] = {"pred": pred, "conf": f"{conf*100:.0f}%", "factors": "; ".join(factors)}

    # kNN context
    print("  Building kNN...", flush=True)
    sc = MaxAbsScaler()
    nn = NearestNeighbors(n_neighbors=10, metric="cosine")
    nn.fit(sc.fit_transform(coles_train))
    dists, idxs = nn.kneighbors(sc.transform(coles_test))

    knn_ctx = {}
    for i, cid in enumerate(cids_test):
        nb_labels = y_train[idxs[i]]
        pos_count = int(nb_labels.sum())
        neg_count = 10 - pos_count
        majority = pos_label if pos_count > 5 else neg_label
        knn_ctx[cid] = {"pos": pos_count, "neg": neg_count, "majority": majority}

    # Few-shot examples (2 fixed examples from train)
    male_cid = int(cids_train[y_train == 1][0])
    female_cid = int(cids_train[y_train == 0][0])
    few_shot_examples = (
        f"Example 1:\n{serialize(male_cid)}\nAnswer: male\n\n"
        f"Example 2:\n{serialize(female_cid)}\nAnswer: female\n\n"
    )

    # Build enrichment strings
    def enrichment_none(cid):
        return ""

    def enrichment_shap(cid):
        s = shap_ctx[cid]
        return f"\nML model: predicts {s['pred']} ({s['conf']} confidence).\nKey factors: {s['factors']}.\n"

    def enrichment_knn(cid):
        k = knn_ctx[cid]
        return f"\nSimilar clients: {k['pos']} {pos_label}, {k['neg']} {neg_label} (majority: {k['majority']}).\n"

    def enrichment_both(cid):
        s = shap_ctx[cid]
        k = knn_ctx[cid]
        return (f"\nML model: {s['pred']} ({s['conf']}). Factors: {s['factors']}.\n"
                f"Similar clients: {k['pos']} {pos_label}, {k['neg']} {neg_label}.\n")

    enrichments = {
        "none": enrichment_none,
        "shap": enrichment_shap,
        "knn": enrichment_knn,
        "both": enrichment_both,
    }

    # Build strategy prompts
    SYSTEM_PLAIN = "You are a bank analyst predicting client gender (male or female). Answer: male or female."
    SYSTEM_ENRICHED = "You are a bank analyst predicting client gender (male or female). You have ML analysis. Answer: male or female."

    def make_messages(cid, strategy, enrich_fn):
        profile = serialize(cid)
        enrichment = enrich_fn(cid)
        has_enrichment = enrichment.strip() != ""
        system = SYSTEM_ENRICHED if has_enrichment else SYSTEM_PLAIN

        if strategy == "zero_shot":
            return [{"role": "system", "content": system},
                    {"role": "user", "content": f"Profile:\n{profile}{enrichment}\nPredict."}]
        elif strategy == "few_shot":
            return [{"role": "system", "content": system},
                    {"role": "user", "content": f"{few_shot_examples}Now predict:\nProfile:\n{profile}{enrichment}\nPredict."}]
        elif strategy == "cot":
            return [{"role": "system", "content": system},
                    {"role": "user", "content": f"Profile:\n{profile}{enrichment}\nThink step by step, then predict."}]

    strategies = ["zero_shot", "few_shot", "cot"]

    # Already have results for these (skip if you want)
    known = {
        ("zero_shot", "none"): 0.498,
        ("few_shot", "none"): 0.578,
        ("cot", "shap"): 0.606,
        ("cot", "knn"): 0.762,
        ("cot", "both"): 0.745,
    }

    results = {}
    for strat in strategies:
        for enrich_name, enrich_fn in enrichments.items():
            key = (strat, enrich_name)
            if key in known:
                results[f"{strat}_{enrich_name}"] = known[key]
                print(f"  {strat:10s} × {enrich_name:5s} = {known[key]:.4f} (cached)", flush=True)
                continue

            print(f"\n  --- {strat} × {enrich_name} ---", flush=True)
            t0 = time.time()
            preds = []
            for i, cid in enumerate(cids_test):
                msgs = make_messages(cid, strat, enrich_fn)
                preds.append(predict_binary(msgs, pos_tokens, neg_tokens))
                if (i+1) % 200 == 0:
                    auc = roc_auc_score(y_test[:i+1], preds)
                    print(f"    {i+1}/{len(cids_test)} ({(i+1)/(time.time()-t0):.1f}/s, AUC={auc:.4f})", flush=True)
            auc = roc_auc_score(y_test, preds)
            results[f"{strat}_{enrich_name}"] = auc
            print(f"  {strat} × {enrich_name}: AUC={auc:.4f}", flush=True)

    # Print matrix
    print("\n" + "="*60)
    print("STRATEGY × ENRICHMENT MATRIX (Gender, Qwen2.5-7B)")
    print("="*60)
    print(f"{'Strategy':<12} {'None':>8} {'+ SHAP':>8} {'+ kNN':>8} {'+ Both':>8}")
    for strat in strategies:
        row = []
        for enrich_name in ["none", "shap", "knn", "both"]:
            val = results.get(f"{strat}_{enrich_name}", None)
            row.append(f"{val:.4f}" if val else "?")
        print(f"{strat:<12} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8}")

    with open(OUTPUT_DIR / "gender_matrix.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR / 'gender_matrix.json'}")
    return results


if __name__ == "__main__":
    results = run_gender()
    del model, tokenizer; torch.cuda.empty_cache()
