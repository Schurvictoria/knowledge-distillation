#!/usr/bin/env python3
"""
Structured CoT for Gender + Rosbank LLM inference.
Variants: baseline, SHAP CoT, kNN CoT (from CoLES embeddings), full (both).
Runs both datasets sequentially. Auto-pushes.
"""

import time, json, warnings, gc
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier
import shap

OUTPUT_DIR = Path("results/gender_rosbank_cot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")

MCC_GROUPS = {range(1,1500):"Agriculture",range(4000,4800):"Transportation",range(5000,5600):"Retail",
              range(5600,5700):"Clothing",range(5800,5900):"Restaurants",range(6000,7000):"Financial",
              range(7500,7600):"Auto Services",range(8000,8100):"Medical",range(8200,8300):"Education"}
def mcc_cat(mcc):
    try: mcc=int(mcc)
    except: return "Other"
    for r,n in MCC_GROUPS.items():
        if mcc in r: return n
    return "Other"

# ---- Load LLM ----
print("Loading LLM...")
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto", trust_remote_code=True)
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


def run_dataset(dataset_name):
    print(f"\n{'='*60}")
    print(f"STRUCTURED CoT: {dataset_name.upper()}")
    print(f"{'='*60}")

    # Load data
    cids_train = np.load(f"embeddings/{dataset_name}/cids_train_seed42.npy")
    cids_test = np.load(f"embeddings/{dataset_name}/cids_test_seed42.npy")
    y_train = np.load(f"embeddings/{dataset_name}/y_train_seed42.npy")
    y_test = np.load(f"embeddings/{dataset_name}/y_test_seed42.npy")
    coles_train = np.load(f"embeddings/{dataset_name}/emb_train_seed42.npy")
    coles_test = np.load(f"embeddings/{dataset_name}/emb_test_seed42.npy")

    if dataset_name == "gender":
        tx = pd.read_csv(DATA_DIR / "transactions.csv")
        labels = pd.read_csv(DATA_DIR / "gender_train.csv")
        tx = tx[tx["customer_id"].isin(labels["customer_id"])].copy()
        target_map = dict(zip(labels["customer_id"], labels["gender"]))
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

    else:  # rosbank
        df = pd.read_csv(DATA_DIR / "rosbank_train.csv")
        df["dt"] = pd.to_datetime(df["TRDATETIME"], format="%d%b%y:%H:%M:%S")
        df = df.sort_values(["cl_id", "dt"])
        labels_df = df.groupby("cl_id")["target_flag"].max().reset_index()
        labels_df.columns = ["customer_id", "target"]
        target_map = dict(zip(labels_df["customer_id"], labels_df["target"]))
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

        def agg_feat(cids):
            recs = []
            for cid in cids:
                if cid not in grouped.groups:
                    recs.append({"n":0,"mean":0,"std":0,"med":0,"nmcc":0})
                    continue
                ct = grouped.get_group(cid)
                a = ct["amount"].values
                recs.append({"n":len(ct),"mean":np.abs(a).mean(),"std":np.abs(a).std(),
                             "med":np.median(np.abs(a)),"nmcc":ct["MCC"].nunique()})
            return np.array(pd.DataFrame(recs).values, dtype=np.float32)

    feat_names = ["n_tx", "mean_amt", "std_amt", "median_amt", "n_mcc"]
    print(f"  train={len(cids_train)}, test={len(cids_test)}")

    # SHAP
    print("  Computing SHAP...")
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

    # kNN
    print("  Building kNN...")
    sc = MaxAbsScaler()
    nn = NearestNeighbors(n_neighbors=10, metric="cosine")
    nn.fit(sc.fit_transform(coles_train))
    dists, idxs = nn.kneighbors(sc.transform(coles_test))

    knn_ctx = {}
    for i, cid in enumerate(cids_test):
        nb_labels = y_train[idxs[i]]
        pos_count = nb_labels.sum()
        neg_count = 10 - pos_count
        majority = pos_label if pos_count > 5 else neg_label
        knn_ctx[cid] = {"pos": int(pos_count), "neg": int(neg_count), "majority": majority}

    # Run variants
    SYSTEM_BASE = f"You are a bank analyst predicting client {task_desc}. Answer: {answer_fmt}."
    SYSTEM_COT = f"You are a bank analyst predicting client {task_desc}. You have ML analysis. Answer: {answer_fmt}."

    variants = {
        "baseline": lambda cid: [
            {"role": "system", "content": SYSTEM_BASE},
            {"role": "user", "content": f"Profile:\n{serialize(cid)}\n\nPredict."}],
        "shap_cot": lambda cid: [
            {"role": "system", "content": SYSTEM_COT},
            {"role": "user", "content": (
                f"Profile:\n{serialize(cid)}\n\n"
                f"ML model: predicts {shap_ctx[cid]['pred']} ({shap_ctx[cid]['conf']} confidence).\n"
                f"Key factors: {shap_ctx[cid]['factors']}.\n\nPredict.")}],
        "knn_cot": lambda cid: [
            {"role": "system", "content": SYSTEM_COT},
            {"role": "user", "content": (
                f"Profile:\n{serialize(cid)}\n\n"
                f"Similar clients: {knn_ctx[cid]['pos']} {pos_label}, {knn_ctx[cid]['neg']} {neg_label} "
                f"(majority: {knn_ctx[cid]['majority']}).\n\nPredict.")}],
        "full_cot": lambda cid: [
            {"role": "system", "content": SYSTEM_COT},
            {"role": "user", "content": (
                f"Profile:\n{serialize(cid)}\n\n"
                f"ML model: {shap_ctx[cid]['pred']} ({shap_ctx[cid]['conf']}). "
                f"Factors: {shap_ctx[cid]['factors']}.\n"
                f"Similar clients: {knn_ctx[cid]['pos']} {pos_label}, {knn_ctx[cid]['neg']} {neg_label}.\n\nPredict.")}],
    }

    ds_results = {}
    for vname, msg_fn in variants.items():
        print(f"\n  --- {vname} ---")
        ts = time.time()
        preds = []
        for i, cid in enumerate(cids_test):
            preds.append(predict_binary(msg_fn(cid), pos_tokens, neg_tokens))
            if (i+1) % 200 == 0:
                auc = roc_auc_score(y_test[:i+1], preds)
                print(f"    {i+1}/{len(cids_test)} ({(i+1)/(time.time()-ts):.1f}/s, AUC={auc:.4f})")
        auc = roc_auc_score(y_test, preds)
        ds_results[vname] = auc
        print(f"  {vname}: AUC={auc:.4f}")

    # Summary
    print(f"\n  {dataset_name.upper()} CoT RESULTS:")
    base = ds_results["baseline"]
    for n, v in sorted(ds_results.items(), key=lambda x: -x[1]):
        print(f"    {n:<15} AUC={v:.4f} ({'+' if v-base>=0 else ''}{v-base:.4f})")

    with open(OUTPUT_DIR / f"{dataset_name}_cot_results.json", "w") as f:
        json.dump(ds_results, f, indent=2)

    return ds_results


# ---- Run both ----
all_results = {}
for ds in ["gender", "rosbank"]:
    all_results[ds] = run_dataset(ds)

print("\n" + "=" * 60)
print("ALL STRUCTURED CoT RESULTS")
print("=" * 60)
for ds, r in all_results.items():
    base = r["baseline"]
    best_name = max(r, key=r.get)
    print(f"  {ds}: baseline={base:.4f}, best={r[best_name]:.4f} ({best_name}, {'+' if r[best_name]-base>=0 else ''}{r[best_name]-base:.4f})")

del model, tokenizer; torch.cuda.empty_cache()
