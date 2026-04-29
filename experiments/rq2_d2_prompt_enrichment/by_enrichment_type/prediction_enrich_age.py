import os, json, time, re, requests, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

import sys

import random as _random, os as _os
_SEED = 42
_random.seed(_SEED); np.random.seed(_SEED)
_os.environ["PYTHONHASHSEED"] = str(_SEED)

from pathlib import Path as _P
_required_inputs = [
    ("data/transactions_train.csv", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/cids_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/cids_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/y_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/y_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
]
for _p, _hint in _required_inputs:
    assert _P(_p).exists(), f"\n  Missing input: {_p}\n  Run prerequisite: {_hint}"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "qwen/qwen-2.5-7b-instruct"
OUT = Path("results/openrouter")
OUT.mkdir(parents=True, exist_ok=True)

class Budget:
    def __init__(self, cap=2.0):
        self.spent = 0; self.calls = 0; self.cap = cap
    def add(self, pt, ct):
        self.spent += pt * 0.20e-6 + ct * 0.40e-6
        self.calls += 1
        if self.calls % 50 == 0:
            print(f"    [BUDGET] ${self.spent:.3f}/${self.cap:.2f} ({self.calls})")
    def check(self):
        if self.spent > self.cap:
            print(f"    [BUDGET EXCEEDED]"); return False
        return True

budget = Budget()

def call_qwen25(messages, api_key, seed=42):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "label": {"type": "integer", "enum": [0, 1, 2, 3]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["label", "confidence"],
        "additionalProperties": False,
    }
    payload = {
        "model": MODEL_ID, "messages": messages,
        "max_tokens": 500, "temperature": 0, "seed": seed,
        "response_format": {"type": "json_schema",
            "json_schema": {"name": "cls", "strict": True, "schema": schema}},
    }
    for attempt in range(3):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429: time.sleep(5*(attempt+1)); continue
            if resp.status_code == 402:
                print(f"    [402 CREDITS EXHAUSTED] Halting.")
                import os; os._exit(2)

            if resp.status_code != 200:
                if attempt == 2: return -1
                time.sleep(2); continue
            data = resp.json()
            content = data["choices"][0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            budget.add(usage.get("prompt_tokens", 500), usage.get("completion_tokens", 50))
            if not budget.check(): return -1
            try:
                parsed = json.loads(content); return int(parsed["label"])
            except:
                m = re.search(r'"label"\s*:\s*(\d)', content)
                if m: return int(m.group(1))
                digits = [int(c) for c in content if c.isdigit() and int(c)<4]
                return digits[-1] if digits else -1
        except:
            if attempt == 2: return -1
            time.sleep(2)
    return -1

def stratified_subset(y, n=500, seed=42):
    rng = np.random.RandomState(seed)
    per = n // 4
    idxs = []
    for c in range(4):
        ci = np.where(y == c)[0]
        idxs.extend(rng.choice(ci, per, replace=False).tolist())
    rng.shuffle(idxs)
    return np.array(idxs)

def run(api_key, n=500):
    print(f"=== Prediction-CoT Age (Qwen2.5-7B via OpenRouter) ===")

    DATA = Path("data")
    tx = pd.read_csv(DATA / "transactions_train.csv")
    grouped = tx.groupby("client_id")
    cids_train = np.load("embeddings/age/cids_train_seed42.npy")
    cids_test = np.load("embeddings/age/cids_test_seed42.npy")
    y_train = np.load("embeddings/age/y_train_seed42.npy")
    y_test = np.load("embeddings/age/y_test_seed42.npy")

    def agg(cids):
        recs = []
        for cid in cids:
            if cid not in grouped.groups: recs.append([0]*5); continue
            ct = grouped.get_group(cid)
            a = np.abs(ct["amount_rur"].values)
            recs.append([len(ct), float(a.mean()), float(a.std()), float(np.median(a)),
                         int(ct["small_group"].nunique())])
        return np.array(recs, dtype=np.float32)

    X_tr = agg(cids_train); X_te = agg(cids_test)

    xgb = XGBClassifier(n_estimators=300, max_depth=6, random_state=42, verbosity=0,
                        objective="multi:softprob", num_class=4)
    xgb.fit(X_tr, y_train)
    xgb_test = xgb.predict_proba(X_te)

    subset = stratified_subset(y_test, n=n, seed=42)
    print(f"  subset: {len(subset)} samples, class dist: {np.bincount(y_test[subset]).tolist()}")

    cache = OUT / "age_qwen25_7b_prediction_cot.json"
    if cache.exists():
        c = json.load(open(cache))
        print(f"  Cached acc={c['accuracy']:.4f}"); return

    system = ("You are a bank analyst predicting age group from transaction patterns. "
              "Categories: 0=youngest, 1=young-adult, 2=middle-age, 3=senior. "
              "You also have prediction from ML model. Think step by step.")

    def serialize(cid):
        if cid not in grouped.groups: return "No txns."
        ct = grouped.get_group(cid)
        a = np.abs(ct["amount_rur"].values)
        mcc = ct["small_group"].value_counts().head(5).to_dict()
        return (f"Transactions: {len(ct)}\n"
                f"Spending: avg {a.mean():.0f} RUB, median {np.median(a):.0f}, max {a.max():.0f}\n"
                f"Top 5 categories: {list(mcc.items())[:5]}")

    def process_one(pos):
        cid = int(cids_test[pos])
        profile = serialize(cid)
        probs = xgb_test[pos]
        pred_class = int(np.argmax(probs))
        conf = probs[pred_class]
        enrich = f"ML model predicts: class {pred_class} ({conf*100:.0f}% confidence)."

        user = (f"{profile}\n\n{enrich}\n\nClassify age group. Output JSON: "
                f'{{"reasoning": "analysis", "label": 0|1|2|3, "confidence": 0-1}}')
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        return pos, call_qwen25(messages, api_key, seed=42)

    ckpt = OUT / "age_qwen25_7b_prediction_cot_ckpt.npz"
    if ckpt.exists():
        preds = list(np.load(ckpt)["preds"])
        start = len(preds); print(f"  Resuming from {start}")
    else:
        preds = []; start = 0

    if start == 0:
        print("  Sanity check...")
        for i in range(2):
            pos = int(subset[i])
            _, p = process_one(pos)
            print(f"    {i}: true={y_test[pos]}, xgb={np.argmax(xgb_test[pos])}, pred={p}")

    BATCH = 5; t0 = time.time()
    for bs in range(start, len(subset), BATCH):
        be = min(bs + BATCH, len(subset))
        with ThreadPoolExecutor(max_workers=BATCH) as ex:
            futs = [ex.submit(process_one, int(subset[i])) for i in range(bs, be)]
            br = {}
            for f in as_completed(futs):
                pos, pred = f.result(); br[pos] = pred
        for i in range(bs, be):
            preds.append(br[int(subset[i])])

        if len(preds) % 50 < BATCH:
            np.savez(ckpt, preds=np.array(preds))
            y_sub = y_test[subset[:len(preds)]]
            p_arr = np.array(preds); mask = p_arr >= 0
            if mask.sum() > 10:
                acc = accuracy_score(y_sub[mask], p_arr[mask])
                rate = (len(preds)-start)/max(time.time()-t0, 0.1)
                print(f"    {len(preds)}/{len(subset)} ({rate:.1f}/s, acc={acc:.4f})")
        if not budget.check():
            np.savez(ckpt, preds=np.array(preds)); return

    p_arr = np.array(preds); y_sub = y_test[subset[:len(preds)]]
    mask = p_arr >= 0
    acc = accuracy_score(y_sub[mask], p_arr[mask])
    f1 = f1_score(y_sub[mask], p_arr[mask], average="macro")
    print(f"\nFINAL: acc={acc:.4f}, macro-f1={f1:.4f} (valid {mask.sum()}/{len(preds)})")
    with open(cache, "w") as f:
        json.dump({"accuracy": acc, "macro_f1": f1, "method": "prediction_cot",
                   "model": "qwen2.5-7b", "dataset": "age", "n_test": int(mask.sum()), "seed": 42}, f, indent=2)
    np.savez(OUT / "age_qwen25_7b_prediction_cot_preds.npz",
             preds=p_arr, y_test=y_sub, subset_idx=subset)
    if ckpt.exists(): ckpt.unlink()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    budget.cap = args.budget
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key: exit("Set OPENROUTER_API_KEY")
    run(api_key, n=args.n)
