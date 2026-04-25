#!/usr/bin/env python3
"""
Structured CoT for Age LLM inference.
Direction: Structured model → LLM (enriches LLM prompts with ML signals).

Three CoT variants:
A) SHAP CoT: XGBoost reasoning in prompt
B) kNN CoT: similar clients from CoLES embeddings + their labels
C) Combined: SHAP + kNN

Hypothesis: structured CoT improves LLM Age accuracy from 0.284 → 0.4+.
If yes → better LLM embeddings → distillation works better on Age.
"""

import time, json, warnings, gc
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier
import shap

OUTPUT_DIR = Path("results/age_structured_cot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")

AGE_LABELS = {0: "young (under 25)", 1: "adult (25-35)", 2: "middle-aged (35-55)", 3: "senior (over 55)"}

MCC_GROUPS = {range(1,1500):"Agriculture",range(4000,4800):"Transportation",range(5000,5600):"Retail",
              range(5600,5700):"Clothing",range(5800,5900):"Restaurants",range(6000,7000):"Financial",
              range(7500,7600):"Auto Services",range(8000,8100):"Medical",range(8200,8300):"Education"}
def mcc_cat(mcc):
    try: mcc=int(mcc)
    except: return "Other"
    for r,n in MCC_GROUPS.items():
        if mcc in r: return n
    return "Other"

# ---- Load data ----
print("Loading data...")
tx = pd.read_csv(DATA_DIR / "transactions_train.csv")
labels = pd.read_csv(DATA_DIR / "train_target.csv")
target_map = dict(zip(labels["client_id"], labels["bins"]))
tx = tx.sort_values(["client_id", "trans_date"])
grouped = tx.groupby("client_id")

# Use same split as other experiments
cids_train = np.load("embeddings/age/cids_train_seed42.npy")
cids_test = np.load("embeddings/age/cids_test_seed42.npy")
y_train = np.load("embeddings/age/y_train_seed42.npy")
y_test = np.load("embeddings/age/y_test_seed42.npy")

# Sample 3k from test for speed
np.random.seed(42)
sample_idx = np.random.choice(len(cids_test), min(3000, len(cids_test)), replace=False)
eval_cids = cids_test[sample_idx]
eval_targets = y_test[sample_idx]
print(f"  eval: {len(eval_cids)} clients, classes: {np.bincount(eval_targets)}")

# ---- Build structured signals ----
print("\nBuilding structured signals...")

# 1. Aggregate features for XGBoost
def agg_features(cids):
    records = []
    for cid in cids:
        if cid not in grouped.groups:
            records.append({"cid": cid, "n_tx": 0, "mean_amt": 0, "std_amt": 0, "median_amt": 0, "n_mcc": 0})
            continue
        ct = grouped.get_group(cid)
        a = ct["amount_rur"].values
        records.append({"cid": cid, "n_tx": len(ct), "mean_amt": np.abs(a).mean(),
                        "std_amt": np.abs(a).std(), "median_amt": np.median(np.abs(a)),
                        "n_mcc": ct["small_group"].nunique()})
    return pd.DataFrame(records).set_index("cid")

feat_train = agg_features(cids_train)
feat_test = agg_features(eval_cids)

# OOF XGBoost predictions + SHAP
print("  Computing OOF SHAP...")
xgb = XGBClassifier(n_estimators=300, max_depth=6, objective="multi:softmax", num_class=4,
                     random_state=42, verbosity=0)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = cross_val_predict(xgb, feat_train.values, y_train, cv=cv)

xgb_full = XGBClassifier(n_estimators=300, max_depth=6, objective="multi:softmax", num_class=4,
                          random_state=42, verbosity=0)
xgb_full.fit(feat_train.values, y_train)

# SHAP for test clients
test_preds = xgb_full.predict(feat_test.values)
explainer = shap.TreeExplainer(xgb_full)
shap_values = explainer.shap_values(feat_test.values)
feat_names = list(feat_train.columns)

# Build SHAP context per test client
shap_contexts = {}
for i, cid in enumerate(eval_cids):
    pred_class = int(test_preds[i])
    if isinstance(shap_values, list):
        sv = shap_values[pred_class][i]
    elif shap_values.ndim == 3:
        sv = shap_values[i, :, pred_class]
    else:
        sv = shap_values[i]
    top_idx = np.argsort(np.abs(sv))[::-1][:3]
    factors = []
    for j in top_idx:
        val = feat_test.values[i, int(j)]
        direction = "high" if sv[int(j)] > 0 else "low"
        factors.append(f"{feat_names[int(j)]}={val:.0f} ({direction} → {AGE_LABELS[pred_class]})")
    shap_contexts[cid] = {
        "pred": AGE_LABELS[pred_class],
        "factors": "; ".join(factors),
    }
print(f"  SHAP: {len(shap_contexts)} contexts")

# 2. kNN from CoLES embeddings
print("  Building kNN index...")
coles_train = np.load("embeddings/age/emb_train_seed42.npy")
coles_test = np.load("embeddings/age/emb_test_seed42.npy")

sc = MaxAbsScaler()
coles_train_s = sc.fit_transform(coles_train)
coles_test_s = sc.transform(coles_test)

# Only use eval subset of test
coles_eval = coles_test_s[sample_idx]

nn = NearestNeighbors(n_neighbors=10, metric="cosine")
nn.fit(coles_train_s)
distances, indices = nn.kneighbors(coles_eval)

knn_contexts = {}
for i, cid in enumerate(eval_cids):
    neighbor_labels = y_train[indices[i]]
    label_counts = np.bincount(neighbor_labels, minlength=4)
    top_label = np.argmax(label_counts)
    knn_contexts[cid] = {
        "neighbors": f"young={label_counts[0]}, adult={label_counts[1]}, middle={label_counts[2]}, senior={label_counts[3]}",
        "majority": AGE_LABELS[top_label],
        "confidence": label_counts[top_label] / 10,
    }
print(f"  kNN: {len(knn_contexts)} contexts")

# ---- Serialization ----
def serialize_client(cid, max_txns=30):
    if cid not in grouped.groups: return "No transactions."
    ct = grouped.get_group(cid).tail(max_txns)
    n = len(ct)
    amt = np.abs(ct["amount_rur"].values)
    cats = ct["small_group"].fillna(0).astype(int).apply(mcc_cat).value_counts()
    top = ", ".join(f"{c} ({n_})" for c, n_ in cats.head(6).items())
    return (f"Client with {n} transactions across {cats.nunique()} categories. "
            f"Avg amount: {amt.mean():.0f}, median: {np.median(amt):.0f}. "
            f"Top categories: {top}.")

# ---- LLM inference ----
print("\nLoading LLM...")
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



MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto", trust_remote_code=True)
model.eval()

AGE_TOKENS = {0: ["young"," young","Young","0"], 1: ["adult"," adult","Adult","1"],
              2: ["middle"," middle","Middle","2"], 3: ["senior"," senior","Senior","3"]}

def get_ids(tokens):
    ids = set()
    for t in tokens:
        e = tokenizer.encode(t, add_special_tokens=False)
        if e: ids.add(e[0])
    return list(ids)

class_ids = {k: get_ids(v) for k, v in AGE_TOKENS.items()}

def predict_class(messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    all_ids = []
    ranges = {}
    s = 0
    for cls in range(4):
        ids = class_ids[cls]
        all_ids.extend(ids)
        ranges[cls] = (s, s+len(ids))
        s += len(ids)
    probs = torch.softmax(logits[all_ids].float(), dim=0)
    class_probs = [probs[a:b].sum().item() for a,b in ranges.values()]
    del inputs
    return int(np.argmax(class_probs))

SYSTEM_BASE = ("You are a bank analyst predicting client age group from transaction patterns. "
               "Categories: young (under 25), adult (25-35), middle-aged (35-55), senior (over 55). "
               "Answer with one word: young, adult, middle, or senior.")

SYSTEM_COT = ("You are a bank analyst predicting client age group. You have the client's transactions "
              "AND analysis from machine learning models. Use all available evidence. "
              "Think step by step, then answer: young, adult, middle, or senior.")

# ---- Run all variants ----
variants = {
    "baseline": lambda cid: [
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user", "content": f"Profile:\n{serialize_client(cid)}\n\nPredict age group."}
    ],
    "shap_cot": lambda cid: [
        {"role": "system", "content": SYSTEM_COT},
        {"role": "user", "content": (
            f"Profile:\n{serialize_client(cid)}\n\n"
            f"ML model analysis:\n"
            f"XGBoost predicts: {shap_contexts[cid]['pred']}.\n"
            f"Key factors: {shap_contexts[cid]['factors']}.\n\n"
            f"Based on all evidence, predict age group."
        )}
    ],
    "knn_cot": lambda cid: [
        {"role": "system", "content": SYSTEM_COT},
        {"role": "user", "content": (
            f"Profile:\n{serialize_client(cid)}\n\n"
            f"Similar clients analysis:\n"
            f"Among 10 most similar clients: {knn_contexts[cid]['neighbors']}.\n"
            f"Most common age group: {knn_contexts[cid]['majority']} "
            f"({knn_contexts[cid]['confidence']*100:.0f}% of neighbors).\n\n"
            f"Based on all evidence, predict age group."
        )}
    ],
    "full_cot": lambda cid: [
        {"role": "system", "content": SYSTEM_COT},
        {"role": "user", "content": (
            f"Profile:\n{serialize_client(cid)}\n\n"
            f"Evidence 1 - ML model:\n"
            f"XGBoost predicts: {shap_contexts[cid]['pred']}.\n"
            f"Key factors: {shap_contexts[cid]['factors']}.\n\n"
            f"Evidence 2 - Similar clients:\n"
            f"Among 10 similar clients: {knn_contexts[cid]['neighbors']}.\n"
            f"Majority: {knn_contexts[cid]['majority']}.\n\n"
            f"Consider both evidences and predict age group."
        )}
    ],
}

results = {}
for variant_name, msg_fn in variants.items():
    print(f"\n--- {variant_name} ---")
    ts = time.time()
    preds = []
    for i, cid in enumerate(eval_cids):
        msgs = msg_fn(cid)
        preds.append(predict_class(msgs))
        if (i+1) % 300 == 0:
            acc = accuracy_score(eval_targets[:i+1], preds)
            rate = (i+1)/(time.time()-ts)
            print(f"  {i+1}/{len(eval_cids)} ({rate:.1f}/s, acc={acc:.4f})")

    acc = accuracy_score(eval_targets, preds)
    results[variant_name] = acc
    print(f"  {variant_name}: acc={acc:.4f}")

    pd.DataFrame({"customer_id": eval_cids, "target": eval_targets,
                   "pred": preds}).to_csv(OUTPUT_DIR / f"age_{variant_name}_preds.csv", index=False)

# Summary
print("\n" + "=" * 60)
print("STRUCTURED CoT RESULTS")
print("=" * 60)
for n, v in sorted(results.items(), key=lambda x: -x[1]):
    d = v - results["baseline"]
    print(f"  {n:<15} acc={v:.4f} ({'+' if d>=0 else ''}{d:.4f} vs baseline)")

with open(OUTPUT_DIR / "cot_results.json", "w") as f:
    json.dump(results, f, indent=2)

del model, tokenizer; torch.cuda.empty_cache()
