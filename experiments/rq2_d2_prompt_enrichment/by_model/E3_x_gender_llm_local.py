#!/usr/bin/env python3
"""
Phase 2: LLM inference on Gender dataset.
Qwen2.5-7B-Instruct, 4-bit NF4, RTX 3090.
Strategies: zero-shot, few-shot, shap-enriched (OOF).
Saves predictions + probabilities for Phase 3.
"""

import time, json, warnings, gc, re
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
import shap

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


print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

OUTPUT_DIR = Path("results/gender_llm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# ---- MCC code to human-readable categories ----
MCC_GROUPS = {
    range(1, 1500): "Agriculture",
    range(1500, 3000): "Construction/Contractors",
    range(3000, 3300): "Airlines",
    range(3300, 3500): "Car Rental",
    range(3500, 4000): "Hotels/Lodging",
    range(4000, 4800): "Transportation",
    range(4800, 5000): "Utilities/Telecom",
    range(5000, 5600): "Retail/Stores",
    range(5600, 5700): "Clothing",
    range(5700, 5800): "Home Furnishing",
    range(5800, 5900): "Restaurants/Food",
    range(5900, 6000): "Drug Stores/Pharmacies",
    range(6000, 7000): "Financial Services",
    range(7000, 7300): "Personal Services",
    range(7300, 7500): "Business Services",
    range(7500, 7600): "Auto Services",
    range(7600, 7700): "Repair Services",
    range(7700, 7800): "Entertainment",
    range(7800, 8000): "Recreation",
    range(8000, 8100): "Medical Services",
    range(8100, 8200): "Legal Services",
    range(8200, 8300): "Education",
    range(8300, 8700): "Membership/Organizations",
    range(8700, 8800): "Professional Services",
    range(8800, 9000): "Government",
    range(9000, 10000): "Government/Other",
}

def mcc_to_category(mcc):
    try:
        mcc = int(mcc)
    except (ValueError, TypeError):
        return "Unknown"
    for r, name in MCC_GROUPS.items():
        if mcc in r:
            return name
    return "Other"


def load_gender_data():
    """Load Gender dataset, return per-client transaction data + labels."""
    tx = pd.read_csv(DATA_DIR / "transactions.csv")
    labels = pd.read_csv(DATA_DIR / "gender_train.csv")
    tx = tx[tx["customer_id"].isin(labels["customer_id"])].copy()

    # Parse datetime
    def parse_dt(s):
        parts = str(s).split(" ", 1)
        day = int(parts[0])
        if len(parts) > 1:
            t = parts[1].split(":")
            return day + (int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])) / 86400.0
        return float(day)

    tx["day_float"] = tx["tr_datetime"].apply(parse_dt)
    tx = tx.sort_values(["customer_id", "day_float"])

    target_map = dict(zip(labels["customer_id"], labels["gender"]))
    return tx, target_map, labels


def serialize_client(client_tx, dataset_name="gender"):
    """Convert client transactions to natural language summary."""
    n_tx = len(client_tx)
    amounts = client_tx["amount"].values
    abs_amounts = np.abs(amounts)

    parts = []

    # Basic stats
    mcc_cats = client_tx["mcc_code"].apply(mcc_to_category)
    n_unique_cats = mcc_cats.nunique()
    parts.append(f"Bank client with {n_tx} transactions across {n_unique_cats} merchant categories.")

    # Amount stats
    parts.append(f"Amount stats: mean={abs_amounts.mean():.0f}, median={np.median(abs_amounts):.0f}, "
                 f"std={abs_amounts.std():.0f}, min={abs_amounts.min():.0f}, max={abs_amounts.max():.0f}.")

    # Top MCC categories
    cat_counts = mcc_cats.value_counts()
    top_cats = [f"{cat} ({count}, {count/n_tx*100:.0f}%)" for cat, count in cat_counts.head(8).items()]
    parts.append(f"Top categories: {', '.join(top_cats)}.")

    # Temporal patterns
    if "tr_datetime" in client_tx.columns:
        hours = client_tx["tr_datetime"].astype(str).str.extract(r' (\d+):')[0]
        if hours.notna().any():
            hours = hours.dropna().astype(int)
            morning = ((hours >= 6) & (hours < 12)).sum()
            afternoon = ((hours >= 12) & (hours < 18)).sum()
            evening = ((hours >= 18) & (hours < 23)).sum()
            night = ((hours >= 23) | (hours < 6)).sum()
            parts.append(f"Time of day: morning={morning}, afternoon={afternoon}, "
                         f"evening={evening}, night={night}.")

    # Spending trend
    half = n_tx // 2
    if half > 0:
        first_half_avg = abs_amounts[:half].mean()
        second_half_avg = abs_amounts[half:].mean()
        if first_half_avg > 0:
            change = (second_half_avg - first_half_avg) / first_half_avg
            if abs(change) > 0.1:
                direction = "increasing" if change > 0 else "declining"
                parts.append(f"Spending trend: {direction} ({change:+.0%}).")

    # Gender-specific: income vs spending
    spend = amounts[amounts < 0]
    income = amounts[amounts > 0]
    total_spend = abs(spend.sum()) if len(spend) > 0 else 0
    total_income = income.sum() if len(income) > 0 else 0
    parts.append(f"Financial flow: {len(spend)} spending txns (total: {total_spend:.0f}), "
                 f"{len(income)} income txns (total: {total_income:.0f}).")
    if total_spend > 0 and total_income > 0:
        parts.append(f"Spend-to-income ratio: {total_spend / total_income:.2f}.")

    # Transaction types
    if "tr_type" in client_tx.columns:
        tr_types = client_tx["tr_type"].value_counts()
        top_types = [f"type_{t}: {c}" for t, c in tr_types.head(5).items()]
        parts.append(f"Uses {len(tr_types)} transaction types. Top: {', '.join(top_types)}.")

    return " ".join(parts)


def compute_oof_shap(tx, target_map, customer_ids):
    """Compute OOF predictions + SHAP values for gender."""
    # Aggregate features per customer
    records = []
    grouped = tx.groupby("customer_id")
    for cid in customer_ids:
        if cid not in grouped.groups:
            continue
        ct = grouped.get_group(cid)
        amounts = ct["amount"].values
        records.append({
            "customer_id": cid,
            "n_tx": len(ct),
            "mean_amt": np.abs(amounts).mean(),
            "std_amt": np.abs(amounts).std(),
            "median_amt": np.median(np.abs(amounts)),
            "n_mcc": ct["mcc_code"].nunique(),
        })

    feat_df = pd.DataFrame(records).set_index("customer_id")
    X = feat_df.values
    y = np.array([target_map[cid] for cid in feat_df.index])

    # OOF predictions (no leakage)
    xgb_cv = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                           random_state=42, verbosity=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = cross_val_predict(xgb_cv, X, y, cv=cv, method="predict_proba")[:, 1]

    # SHAP values (full model, for feature attributions only)
    xgb_full = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                              random_state=42, verbosity=0)
    xgb_full.fit(X, y)
    explainer = shap.TreeExplainer(xgb_full)
    shap_values = explainer.shap_values(X)

    feature_names = list(feat_df.columns)
    shap_contexts = {}
    for i, cid in enumerate(feat_df.index):
        prob = oof_probs[i]
        pred_label = "male" if prob > 0.5 else "female"

        # Top features by |SHAP|
        sv = shap_values[i]
        top_idx = np.argsort(np.abs(sv))[::-1][:5]
        factors = []
        for idx in top_idx:
            direction = "increases" if sv[idx] > 0 else "decreases"
            factors.append(f"{feature_names[idx]} ({direction}, impact={abs(sv[idx]):.3f})")

        shap_contexts[cid] = (
            f"XGBoost predicts '{pred_label}' ({prob*100:.0f}% male confidence). "
            f"Key factors: {'; '.join(factors)}."
        )

    return shap_contexts


def load_model():
    """Load Qwen2.5-7B-Instruct in 4-bit."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading {MODEL_ID} (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    return model, tokenizer


# Target tokens for logits extraction
POS_TOKENS = ["male", " male", "Male", " Male"]
NEG_TOKENS = ["female", " female", "Female", " Female"]


def get_token_ids(tokenizer, token_strings):
    ids = set()
    for t in token_strings:
        encoded = tokenizer.encode(t, add_special_tokens=False)
        if encoded:
            ids.add(encoded[0])
    return list(ids)


SYSTEM_ZERO = (
    "You are a bank transaction analyst who predicts customer gender from spending patterns. "
    "Key patterns: men typically spend more on transportation, auto services, electronics, "
    "restaurants, and entertainment. Women typically spend more on clothing stores, cosmetics, "
    "home furnishing, and medical services. Higher transaction amounts and fewer categories "
    "often indicate male clients. More diverse small purchases across many categories often "
    "indicate female clients. You MUST answer with exactly one word: male or female. No explanations."
)

SYSTEM_SHAP = (
    "You are an expert bank transaction analyst predicting customer gender. "
    "You have the customer's transaction profile AND predictions from a machine learning model "
    "with feature importance analysis. Use both sources. Answer with exactly one word: male or female."
)


def build_messages(strategy, client_text, shap_context=None, few_shot_examples=None):
    if strategy == "zero_shot":
        return [
            {"role": "system", "content": SYSTEM_ZERO},
            {"role": "user", "content": f"Customer transaction profile:\n{client_text}\n\nPredict the customer's gender."},
        ]
    elif strategy == "few_shot":
        examples_text = ""
        for i, (ex_text, ex_label) in enumerate(few_shot_examples[:2]):
            label_str = "male" if ex_label == 1 else "female"
            examples_text += f"\nProfile {i+1}:\n{ex_text}\nAnswer: {label_str}\n"
        return [
            {"role": "system", "content": SYSTEM_ZERO},
            {"role": "user", "content": f"Examples:{examples_text}\nNow predict the gender for this customer:\n{client_text}"},
        ]
    elif strategy == "shap_enriched":
        return [
            {"role": "system", "content": SYSTEM_SHAP},
            {"role": "user", "content": f"Customer transaction profile:\n{client_text}\n\nMachine learning analysis:\n{shap_context}\n\nPredict the customer's gender."},
        ]


def predict_batch(model, tokenizer, messages_list, pos_ids, neg_ids):
    """Get P(male) from logits for a batch of message lists."""
    results = []
    for messages in messages_list:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        last_logits = outputs.logits[0, -1, :]
        all_ids = pos_ids + neg_ids
        target_logits = last_logits[all_ids]
        probs = torch.softmax(target_logits.float(), dim=0)

        pos_prob = probs[:len(pos_ids)].sum().item()
        neg_prob = probs[len(pos_ids):].sum().item()
        total = pos_prob + neg_prob
        results.append(pos_prob / total if total > 1e-8 else 0.5)

        del inputs, outputs
    return results


def select_few_shot_examples(tx, target_map, customer_ids):
    """Select informative few-shot examples using a quick XGBoost."""
    grouped = tx.groupby("customer_id")
    records = []
    for cid in customer_ids:
        if cid not in grouped.groups:
            continue
        ct = grouped.get_group(cid)
        records.append({
            "customer_id": cid,
            "n_tx": len(ct),
            "mean_amt": np.abs(ct["amount"]).mean(),
            "n_mcc": ct["mcc_code"].nunique(),
        })
    feat_df = pd.DataFrame(records).set_index("customer_id")
    X = feat_df.values
    y = np.array([target_map[cid] for cid in feat_df.index])
    cids = list(feat_df.index)

    xgb = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, verbosity=0)
    xgb.fit(X, y)
    probs = xgb.predict_proba(X)[:, 1]

    # Most confident male, most confident female
    male_idx = np.argmax(probs)
    female_idx = np.argmin(probs)

    examples = []
    for idx in [male_idx, female_idx]:
        cid = cids[idx]
        ct = grouped.get_group(cid)
        text = serialize_client(ct)
        examples.append((text, target_map[cid]))
    return examples


# =====================================================================
print("=" * 60)
print("GENDER Phase 2: LLM inference (Qwen2.5-7B, 4-bit)")
print("=" * 60)

# Load data
print("\nLoading data...")
tx, target_map, labels = load_gender_data()
customer_ids = labels["customer_id"].values
targets = np.array([target_map[c] for c in customer_ids])
print(f"  {len(customer_ids)} customers")

# Serialize all clients
print("Serializing transactions...")
grouped = tx.groupby("customer_id")
client_texts = {}
for cid in customer_ids:
    if cid in grouped.groups:
        client_texts[cid] = serialize_client(grouped.get_group(cid))
print(f"  {len(client_texts)} serialized")

# Compute OOF SHAP
print("Computing OOF SHAP...")
shap_contexts = compute_oof_shap(tx, target_map, customer_ids)
print(f"  {len(shap_contexts)} SHAP contexts")

# Select few-shot examples
print("Selecting few-shot examples...")
few_shot_examples = select_few_shot_examples(tx, target_map, customer_ids)

# Load model
model, tokenizer = load_model()
pos_ids = get_token_ids(tokenizer, POS_TOKENS)
neg_ids = get_token_ids(tokenizer, NEG_TOKENS)
print(f"  pos_ids={pos_ids}, neg_ids={neg_ids}")

# Run inference
strategies = ["zero_shot", "few_shot", "shap_enriched"]
all_predictions = {}
t0 = time.time()

for strategy in strategies:
    print(f"\n{'='*40}")
    print(f"Strategy: {strategy}")
    ts = time.time()

    preds = {}
    for i, cid in enumerate(customer_ids):
        if cid not in client_texts:
            preds[cid] = 0.5
            continue

        messages = build_messages(
            strategy,
            client_texts[cid],
            shap_context=shap_contexts.get(cid),
            few_shot_examples=few_shot_examples,
        )

        prob = predict_batch(model, tokenizer, [messages], pos_ids, neg_ids)[0]
        preds[cid] = prob

        if (i + 1) % 200 == 0:
            elapsed = time.time() - ts
            rate = (i + 1) / elapsed
            eta = (len(customer_ids) - i - 1) / rate
            # Compute running AUC
            y_so_far = [target_map[c] for c in list(preds.keys())]
            p_so_far = list(preds.values())
            try:
                running_auc = roc_auc_score(y_so_far, p_so_far)
            except:
                running_auc = 0.0
            print(f"  {i+1}/{len(customer_ids)} ({rate:.1f} it/s, ETA {eta/60:.0f}min, AUC={running_auc:.4f})")

    all_predictions[strategy] = preds

    # Evaluate
    y_true = np.array([target_map[c] for c in customer_ids])
    y_pred = np.array([preds.get(c, 0.5) for c in customer_ids])
    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
    print(f"  {strategy}: AUC={auc:.4f}, Accuracy={acc:.4f}, time={time.time()-ts:.0f}s")

    # Save per-strategy
    pred_df = pd.DataFrame({
        "customer_id": customer_ids,
        "target": y_true,
        "pred_prob": y_pred,
        "pred_label": (y_pred >= 0.5).astype(int),
    })
    pred_df.to_csv(OUTPUT_DIR / f"gender_{strategy}_predictions.csv", index=False)

elapsed = time.time() - t0

# Summary
print("\n" + "=" * 60)
print("GENDER Phase 2 RESULTS")
print("=" * 60)
for strategy in strategies:
    y_true = np.array([target_map[c] for c in customer_ids])
    y_pred = np.array([all_predictions[strategy].get(c, 0.5) for c in customer_ids])
    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
    print(f"  {strategy:<15} AUC={auc:.4f}  acc={acc:.4f}")

print(f"\nTotal time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")

with open(OUTPUT_DIR / "gender_llm_summary.json", "w") as f:
    json.dump({
        "experiment": "Gender Phase 2 LLM",
        "model": MODEL_ID,
        "quantization": "4-bit NF4",
        "strategies": strategies,
        "n_customers": len(customer_ids),
        "time": elapsed,
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }, f, indent=2)
