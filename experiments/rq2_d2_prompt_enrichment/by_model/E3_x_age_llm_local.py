#!/usr/bin/env python3
"""
Phase 2: LLM inference on Age (4-class age group prediction).
Qwen2.5-7B-Instruct, 4-bit NF4. Sample 5k from 30k for inference speed.
"""

import time, json, warnings, gc
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import shap

OUTPUT_DIR = Path("results/age_llm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

MCC_GROUPS = {
    range(1, 1500): "Agriculture", range(4000, 4800): "Transportation",
    range(5000, 5600): "Retail", range(5600, 5700): "Clothing",
    range(5800, 5900): "Restaurants", range(6000, 7000): "Financial",
    range(7500, 7600): "Auto", range(8000, 8100): "Medical",
    range(8200, 8300): "Education",
}

def mcc_cat(mcc):
    try:
        mcc = int(mcc)
    except: return "Other"
    for r, n in MCC_GROUPS.items():
        if mcc in r: return n
    return "Other"

# ---- Load data ----
print("Loading Age data...")
tx = pd.read_csv(DATA_DIR / "transactions_train.csv")
labels = pd.read_csv(DATA_DIR / "train_target.csv")
target_map = dict(zip(labels["client_id"], labels["bins"]))
tx = tx.sort_values(["client_id", "trans_date"])
grouped = tx.groupby("client_id")

# Sample 5k clients stratified
np.random.seed(42)
all_cids = labels["client_id"].values
all_targets = np.array([target_map[c] for c in all_cids])
from sklearn.model_selection import train_test_split
sample_idx, _ = train_test_split(np.arange(len(all_cids)), train_size=5000, random_state=42, stratify=all_targets)
customer_ids = all_cids[sample_idx]
targets = all_targets[sample_idx]
print(f"  Sampled {len(customer_ids)} from {len(all_cids)}, class dist: {np.bincount(targets)}")

AGE_LABELS = {0: "young (under 25)", 1: "adult (25-35)", 2: "middle-aged (35-55)", 3: "senior (over 55)"}

def serialize_client(cid):
    if cid not in grouped.groups:
        return "No transactions."
    ct = grouped.get_group(cid)
    n = len(ct)
    amt = np.abs(ct["amount_rur"].values)
    cats = ct["small_group"].fillna(0).astype(int).apply(mcc_cat).value_counts()
    top = ", ".join(f"{c} ({n_})" for c, n_ in cats.head(6).items())
    return (f"Client with {n} transactions across {cats.nunique()} categories. "
            f"Avg amount: {amt.mean():.0f}, median: {np.median(amt):.0f}. "
            f"Categories: {top}.")

def compute_oof_shap():
    records = []
    for cid in customer_ids:
        if cid not in grouped.groups: continue
        ct = grouped.get_group(cid)
        a = ct["amount_rur"].values
        records.append({"cid": cid, "n_tx": len(ct), "mean_amt": np.abs(a).mean(),
                        "std_amt": np.abs(a).std(), "median_amt": np.median(np.abs(a)),
                        "n_mcc": ct["small_group"].nunique()})
    feat_df = pd.DataFrame(records).set_index("cid")
    X, y = feat_df.values, np.array([target_map[c] for c in feat_df.index])
    xgb_cv = XGBClassifier(n_estimators=300, max_depth=6, objective="multi:softmax", num_class=4,
                            random_state=42, verbosity=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(xgb_cv, X, y, cv=cv)
    xgb_full = XGBClassifier(n_estimators=300, max_depth=6, objective="multi:softmax", num_class=4,
                              random_state=42, verbosity=0)
    xgb_full.fit(X, y)
    sv = shap.TreeExplainer(xgb_full).shap_values(X)
    fn = list(feat_df.columns)
    contexts = {}
    for i, cid in enumerate(feat_df.index):
        pred_class = int(oof[i])
        # For multiclass, sv is list of arrays — take the predicted class
        if isinstance(sv, list):
            sv_i = sv[pred_class][i]
        elif sv.ndim == 3:
            sv_i = sv[i, :, pred_class]
        else:
            sv_i = sv[i]
        top_idx = np.argsort(np.abs(sv_i))[::-1][:3]
        factors = [f"{fn[int(j)]} ({abs(sv_i[int(j)]):.3f})" for j in top_idx]
        contexts[cid] = f"ML predicts '{AGE_LABELS[pred_class]}'. Key factors: {'; '.join(factors)}."
    return contexts

print("Serializing & SHAP...")
client_texts = {cid: serialize_client(cid) for cid in customer_ids}
shap_contexts = compute_oof_shap()

# ---- Load model ----
print("Loading model...")
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


bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto", trust_remote_code=True)
model.eval()

# Tokens for 4 age classes
AGE_TOKENS = {
    0: ["young", " young", "Young", "0"],
    1: ["adult", " adult", "Adult", "1"],
    2: ["middle", " middle", "Middle", "2"],
    3: ["senior", " senior", "Senior", "3"],
}

def get_ids(tokens):
    ids = set()
    for t in tokens:
        e = tokenizer.encode(t, add_special_tokens=False)
        if e: ids.add(e[0])
    return list(ids)

class_ids = {k: get_ids(v) for k, v in AGE_TOKENS.items()}

SYSTEM = ("You are a bank analyst predicting client age group from transaction patterns. "
          "Categories: young (under 25), adult (25-35), middle-aged (35-55), senior (over 55). "
          "Answer with one word: young, adult, middle, or senior.")

SYSTEM_SHAP = ("You are a bank analyst predicting client age group. You have transaction profile "
               "AND ML model predictions. Answer: young, adult, middle, or senior.")

def predict_one(messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    all_ids = []
    class_ranges = {}
    start = 0
    for cls in range(4):
        ids = class_ids[cls]
        all_ids.extend(ids)
        class_ranges[cls] = (start, start + len(ids))
        start += len(ids)
    probs = torch.softmax(logits[all_ids].float(), dim=0)
    class_probs = [probs[s:e].sum().item() for s, e in class_ranges.values()]
    total = sum(class_probs)
    if total < 1e-8:
        return 0
    return int(np.argmax(class_probs))

# Few-shot examples
few_shot = []
for cls in range(4):
    cls_cids = [c for c in customer_ids[:1000] if target_map[c] == cls]
    if cls_cids:
        few_shot.append((client_texts[cls_cids[0]], cls))

# ---- Run ----
strategies = ["zero_shot", "few_shot", "shap_enriched"]
all_preds = {}
t0 = time.time()

for strat in strategies:
    print(f"\n{'='*40}\nStrategy: {strat}")
    ts = time.time()
    preds = {}
    for i, cid in enumerate(customer_ids):
        text = client_texts.get(cid, "Unknown.")
        if strat == "zero_shot":
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Profile:\n{text}\n\nPredict age group."}]
        elif strat == "few_shot":
            ex = "".join(f"\nProfile:\n{t}\nAnswer: {list(AGE_LABELS.values())[l].split('(')[0].strip()}\n"
                        for t, l in few_shot[:4])
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Examples:{ex}\nPredict:\n{text}"}]
        else:
            ctx = shap_contexts.get(cid, "")
            msgs = [{"role": "system", "content": SYSTEM_SHAP},
                    {"role": "user", "content": f"Profile:\n{text}\n\nML analysis:\n{ctx}\n\nPredict age group."}]

        preds[cid] = predict_one(msgs)
        if (i+1) % 200 == 0:
            rate = (i+1)/(time.time()-ts)
            y_so_far = [target_map[c] for c in list(preds.keys())]
            acc = accuracy_score(y_so_far, list(preds.values()))
            print(f"  {i+1}/{len(customer_ids)} ({rate:.1f}/s, ETA {(len(customer_ids)-i-1)/rate/60:.0f}m, acc={acc:.4f})")

    all_preds[strat] = preds
    y_pred = np.array([preds.get(c, 0) for c in customer_ids])
    acc = accuracy_score(targets, y_pred)
    print(f"  {strat}: acc={acc:.4f}, time={time.time()-ts:.0f}s")

    pd.DataFrame({"customer_id": customer_ids, "target": targets,
                   "pred_class": y_pred}).to_csv(OUTPUT_DIR / f"age_{strat}_predictions.csv", index=False)

print("\n" + "=" * 60)
print("AGE Phase 2 RESULTS")
for strat in strategies:
    y_pred = np.array([all_preds[strat].get(c, 0) for c in customer_ids])
    print(f"  {strat:<15} acc={accuracy_score(targets, y_pred):.4f}")
print(f"Total: {time.time()-t0:.0f}s")

with open(OUTPUT_DIR / "age_llm_summary.json", "w") as f:
    json.dump({"model": MODEL_ID, "strategies": strategies, "n_customers": len(customer_ids),
               "time": time.time()-t0, "date": time.strftime("%Y-%m-%d %H:%M")}, f, indent=2)
