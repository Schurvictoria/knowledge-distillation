#!/usr/bin/env python3
"""
Phase 2: LLM inference on Rosbank (churn prediction).
Qwen2.5-7B-Instruct, 4-bit NF4, RTX 3090.
Strategies: zero-shot, few-shot, shap-enriched (OOF).
Batched inference for better GPU utilization.
"""

import time, json, warnings, gc
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import shap

OUTPUT_DIR = Path("results/rosbank_llm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

MCC_GROUPS = {
    range(1, 1500): "Agriculture", range(4000, 4800): "Transportation",
    range(4800, 5000): "Utilities", range(5000, 5600): "Retail",
    range(5600, 5700): "Clothing", range(5800, 5900): "Restaurants",
    range(6000, 7000): "Financial Services", range(7500, 7600): "Auto Services",
    range(7700, 7800): "Entertainment", range(8000, 8100): "Medical",
}

def mcc_cat(mcc):
    try:
        mcc = int(mcc)
    except (ValueError, TypeError):
        return "Other"
    for r, name in MCC_GROUPS.items():
        if mcc in r:
            return name
    return "Other"


# ---- Load data ----
print("Loading Rosbank data...")
df = pd.read_csv(DATA_DIR / "rosbank_train.csv")
df["dt"] = pd.to_datetime(df["TRDATETIME"], format="%d%b%y:%H:%M:%S")
df = df.sort_values(["cl_id", "dt"])

labels_df = df.groupby("cl_id")["target_flag"].max().reset_index()
labels_df.columns = ["customer_id", "target"]
target_map = dict(zip(labels_df["customer_id"], labels_df["target"]))
customer_ids = labels_df["customer_id"].values
targets = np.array([target_map[c] for c in customer_ids])
grouped = df.groupby("cl_id")
print(f"  {len(customer_ids)} customers, churn rate: {targets.mean():.3f}")


def serialize_client(cid):
    if cid not in grouped.groups:
        return "No transactions."
    ct = grouped.get_group(cid)
    n = len(ct)
    amounts = ct["amount"].values
    abs_amt = np.abs(amounts)
    cats = ct["MCC"].fillna(0).astype(int).apply(mcc_cat).value_counts()
    top = ", ".join(f"{c} ({n_})" for c, n_ in cats.head(6).items())
    spend = amounts[amounts < 0]
    income = amounts[amounts > 0]

    return (
        f"Bank client with {n} transactions across {cats.nunique()} categories. "
        f"Average amount: {abs_amt.mean():.0f}, median: {np.median(abs_amt):.0f}. "
        f"Top categories: {top}. "
        f"Spending: {len(spend)} txns (total {abs(spend.sum()) if len(spend)>0 else 0:.0f}), "
        f"Income: {len(income)} txns (total {income.sum() if len(income)>0 else 0:.0f})."
    )


def compute_oof_shap():
    records = []
    for cid in customer_ids:
        if cid not in grouped.groups:
            continue
        ct = grouped.get_group(cid)
        a = ct["amount"].values
        records.append({"cid": cid, "n_tx": len(ct), "mean_amt": np.abs(a).mean(),
                        "std_amt": np.abs(a).std(), "median_amt": np.median(np.abs(a)),
                        "n_mcc": ct["MCC"].nunique()})
    feat_df = pd.DataFrame(records).set_index("cid")
    X, y = feat_df.values, np.array([target_map[c] for c in feat_df.index])

    xgb_cv = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(xgb_cv, X, y, cv=cv, method="predict_proba")[:, 1]

    xgb_full = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
    xgb_full.fit(X, y)
    sv = shap.TreeExplainer(xgb_full).shap_values(X)
    fn = list(feat_df.columns)

    contexts = {}
    for i, cid in enumerate(feat_df.index):
        pred = "churn" if oof[i] > 0.5 else "stay"
        top_idx = np.argsort(np.abs(sv[i]))[::-1][:3]
        factors = [f"{fn[j]} ({'increases' if sv[i][j]>0 else 'decreases'}, {abs(sv[i][j]):.3f})" for j in top_idx]
        contexts[cid] = f"ML predicts '{pred}' ({oof[i]*100:.0f}% churn prob). Key: {'; '.join(factors)}."
    return contexts


# Serialize + SHAP
print("Serializing & computing SHAP...")
client_texts = {cid: serialize_client(cid) for cid in customer_ids}
shap_contexts = compute_oof_shap()

# ---- Load model ----
print("Loading model...")
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto", trust_remote_code=True)
model.eval()

POS_TOKENS = ["yes", " yes", "Yes", " Yes", "churn", " churn", "Churn"]
NEG_TOKENS = ["no", " no", "No", " No", "stay", " stay", "Stay"]

def get_ids(tokens):
    ids = set()
    for t in tokens:
        e = tokenizer.encode(t, add_special_tokens=False)
        if e: ids.add(e[0])
    return list(ids)

pos_ids, neg_ids = get_ids(POS_TOKENS), get_ids(NEG_TOKENS)

SYSTEM_ZERO = (
    "You are a bank churn prediction analyst. Based on transaction patterns, predict if the client "
    "will churn (leave the bank). Answer with exactly one word: yes or no."
)
SYSTEM_SHAP = (
    "You are an expert bank churn analyst. You have the client's transaction profile AND predictions "
    "from a machine learning model. Use both. Answer: yes or no."
)


def predict_one(messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    all_ids = pos_ids + neg_ids
    probs = torch.softmax(logits[all_ids].float(), dim=0)
    pp = probs[:len(pos_ids)].sum().item()
    pn = probs[len(pos_ids):].sum().item()
    total = pp + pn
    del inputs
    return pp / total if total > 1e-8 else 0.5


# Few-shot examples
def select_examples():
    records = []
    for cid in customer_ids[:500]:
        if cid not in grouped.groups: continue
        ct = grouped.get_group(cid)
        records.append({"cid": cid, "n_tx": len(ct), "mean": np.abs(ct["amount"]).mean(), "n_mcc": ct["MCC"].nunique()})
    feat_df = pd.DataFrame(records).set_index("cid")
    xgb = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, verbosity=0)
    xgb.fit(feat_df.values, [target_map[c] for c in feat_df.index])
    p = xgb.predict_proba(feat_df.values)[:, 1]
    pos_idx, neg_idx = np.argmax(p), np.argmin(p)
    cids = list(feat_df.index)
    return [(client_texts[cids[pos_idx]], target_map[cids[pos_idx]]),
            (client_texts[cids[neg_idx]], target_map[cids[neg_idx]])]

few_shot_examples = select_examples()

# ---- Run inference ----
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
            msgs = [{"role": "system", "content": SYSTEM_ZERO},
                    {"role": "user", "content": f"Client profile:\n{text}\n\nWill this client churn?"}]
        elif strat == "few_shot":
            ex = "".join(f"\nProfile:\n{t}\nAnswer: {'yes' if l else 'no'}\n" for t, l in few_shot_examples[:2])
            msgs = [{"role": "system", "content": SYSTEM_ZERO},
                    {"role": "user", "content": f"Examples:{ex}\nNow predict:\n{text}"}]
        else:
            ctx = shap_contexts.get(cid, "")
            msgs = [{"role": "system", "content": SYSTEM_SHAP},
                    {"role": "user", "content": f"Profile:\n{text}\n\nML analysis:\n{ctx}\n\nWill this client churn?"}]

        preds[cid] = predict_one(msgs)
        if (i+1) % 200 == 0:
            rate = (i+1) / (time.time()-ts)
            y_so_far = [target_map[c] for c in list(preds.keys())]
            try: auc = roc_auc_score(y_so_far, list(preds.values()))
            except: auc = 0
            print(f"  {i+1}/{len(customer_ids)} ({rate:.1f}/s, ETA {(len(customer_ids)-i-1)/rate/60:.0f}m, AUC={auc:.4f})")

    all_preds[strat] = preds
    y_pred = np.array([preds.get(c, 0.5) for c in customer_ids])
    auc = roc_auc_score(targets, y_pred)
    print(f"  {strat}: AUC={auc:.4f}, time={time.time()-ts:.0f}s")

    pd.DataFrame({"customer_id": customer_ids, "target": targets,
                   "pred_prob": y_pred, "pred_label": (y_pred>=0.5).astype(int)}).to_csv(
        OUTPUT_DIR / f"rosbank_{strat}_predictions.csv", index=False)

print("\n" + "=" * 60)
print("ROSBANK Phase 2 RESULTS")
for strat in strategies:
    y_pred = np.array([all_preds[strat].get(c, 0.5) for c in customer_ids])
    print(f"  {strat:<15} AUC={roc_auc_score(targets, y_pred):.4f}")
print(f"Total: {time.time()-t0:.0f}s")

with open(OUTPUT_DIR / "rosbank_llm_summary.json", "w") as f:
    json.dump({"model": MODEL_ID, "strategies": strategies, "n_customers": len(customer_ids),
               "time": time.time()-t0, "date": time.strftime("%Y-%m-%d %H:%M")}, f, indent=2)
