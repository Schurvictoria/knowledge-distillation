#!/usr/bin/env python3
"""
GLM-4.7 on Age dataset (4-class: 0,1,2,3).
Subset of 500 stratified test samples to control budget.
Strategies: zero_shot_knn, few_shot_knn (k=8 balanced).
Metric: Accuracy + macro-F1.
"""
import os, json, time, re, random, requests, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import NearestNeighbors

# ---- Reproducibility (seed=42) ----
import random as _random, os as _os
_SEED = 42
_random.seed(_SEED); np.random.seed(_SEED)
_os.environ["PYTHONHASHSEED"] = str(_SEED)

# ---- Required input files ----
from pathlib import Path as _P
_required_inputs = [
    ("data/transactions_train.csv", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/cids_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/cids_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/emb_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/emb_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/y_test_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
    ("embeddings/age/y_train_seed42.npy", "experiments/rq1_bidirectional/coles/run_age_coles.py"),
]
for _p, _hint in _required_inputs:
    assert _P(_p).exists(), f"\n  Missing input: {_p}\n  Run prerequisite: {_hint}"
# ---- end input check ----


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OUT = Path("results/openrouter")
OUT.mkdir(parents=True, exist_ok=True)
MODEL_ID = os.environ.get("AGE_MODEL_ID", "z-ai/glm-4.7")
MODEL_SHORT = os.environ.get("AGE_MODEL_SHORT", "glm47")


class Budget:
    def __init__(self, cap=2.0):
        self.spent = 0.0
        self.calls = 0
        self.cap = cap
    def add(self, prompt_tok, completion_tok):
        # GLM-4.7 approx: $0.20/1M input, $1.10/1M output
        cost = prompt_tok * 0.20e-6 + completion_tok * 1.10e-6
        self.spent += cost
        self.calls += 1
        if self.calls % 50 == 0:
            print(f"    [BUDGET] ${self.spent:.3f}/${self.cap:.2f} ({self.calls} calls)", flush=True)
    def check(self):
        if self.spent > self.cap:
            print(f"    [BUDGET EXCEEDED] ${self.spent:.3f}>${self.cap}. Stop.", flush=True)
            return False
        return True

budget = Budget()


def load_age_data():
    DATA_DIR = Path("data")
    tx = pd.read_csv(DATA_DIR / "transactions_train.csv")
    grouped = tx.groupby("client_id")

    cids_train = np.load("embeddings/age/cids_train_seed42.npy")
    cids_test = np.load("embeddings/age/cids_test_seed42.npy")
    y_train = np.load("embeddings/age/y_train_seed42.npy")
    y_test = np.load("embeddings/age/y_test_seed42.npy")
    emb_train = np.load("embeddings/age/emb_train_seed42.npy")
    emb_test = np.load("embeddings/age/emb_test_seed42.npy")

    def serialize(cid):
        if cid not in grouped.groups:
            return "No transaction data."
        ct = grouped.get_group(cid)
        amt = np.abs(ct["amount_rur"].values)
        mcc = ct["small_group"].value_counts().head(5).to_dict() if "small_group" in ct.columns else {}
        return (f"Client profile:\n"
                f"- Transactions: {len(ct)}\n"
                f"- Spending: avg {amt.mean():.0f} RUB, median {np.median(amt):.0f}, "
                f"max {amt.max():.0f}\n"
                f"- Top 5 categories: {list(mcc.items())[:5]}")

    system = ("You are an expert bank analyst. You predict age group (0=youngest, 1=young-adult, "
              "2=middle-age, 3=senior) from transaction patterns.")
    return {
        "cids_train": cids_train, "cids_test": cids_test,
        "y_train": y_train, "y_test": y_test,
        "emb_train": emb_train, "emb_test": emb_test,
        "serialize": serialize, "system": system,
    }


def stratified_subset(y, n=500, seed=42):
    """Return indices for balanced 4-class subset."""
    rng = np.random.RandomState(seed)
    per_class = n // 4
    idxs = []
    for c in range(4):
        ci = np.where(y == c)[0]
        idxs.extend(rng.choice(ci, per_class, replace=False).tolist())
    rng.shuffle(idxs)
    return np.array(idxs)


def call_glm(messages, api_key, seed=42):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    schema = {
        "type": "object",
        "properties": {
            "label": {"type": "integer", "enum": [0, 1, 2, 3]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["label", "confidence"],
        "additionalProperties": False,
    }
    if "qwen" in MODEL_ID.lower():
        max_tok, reasoning_cfg, temp = 4096, {"max_tokens": 3000}, 0.6
    elif "deepseek" in MODEL_ID.lower():
        max_tok, reasoning_cfg, temp = 8192, {"max_tokens": 6000}, 0.6
    else:  # glm and others
        max_tok, reasoning_cfg, temp = 400, {"enabled": False}, 0
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tok,
        "temperature": temp,
        "seed": seed,
        "reasoning": reasoning_cfg,
        "response_format": {"type": "json_schema",
            "json_schema": {"name": "cls", "strict": True, "schema": schema}},
    }

    for attempt in range(3):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429: time.sleep(5 * (attempt+1)); continue
            if resp.status_code == 402:
                print(f"    [402 CREDITS EXHAUSTED] Halting.", flush=True)
                import os; os._exit(2)

            if resp.status_code != 200:
                if attempt == 2: return -1
                time.sleep(2); continue
            data = resp.json()
            msg = data["choices"][0].get("message", {})
            content = msg.get("content") or ""
            usage = data.get("usage", {})
            budget.add(usage.get("prompt_tokens", 500), usage.get("completion_tokens", 30))
            if not budget.check(): return -1

            try:
                parsed = json.loads(content)
                return int(parsed["label"])
            except:
                combined = content + " " + (msg.get("reasoning") or "")
                m = re.search(r'"label"\s*:\s*(\d)', combined)
                if m: return int(m.group(1))
                # fallback: last digit in content
                digits = [int(c) for c in content if c.isdigit() and int(c) < 4]
                if digits: return digits[-1]
                return -1
        except:
            if attempt == 2: return -1
            time.sleep(2)
    return -1


def build_knn_demos(query_emb, train_emb, train_cids, y_train, serialize, k_per_class=2, rng=None):
    sc = MaxAbsScaler()
    tr_scaled = sc.fit_transform(train_emb)
    q_scaled = sc.transform(query_emb.reshape(1, -1))
    demos = []
    for c in range(4):
        ci = np.where(y_train == c)[0]
        nn = NearestNeighbors(n_neighbors=k_per_class, metric="cosine").fit(tr_scaled[ci])
        _, idxs = nn.kneighbors(q_scaled)
        for idx in ci[idxs[0]]:
            demos.append({"profile": serialize(int(train_cids[idx])), "label": int(y_train[idx])})
    if rng: rng.shuffle(demos)
    return demos


def run_strategy(strategy, data, subset_idx, api_key):
    print(f"\n=== GLM-4.7 {strategy} on age (n={len(subset_idx)}) ===", flush=True)
    cache = OUT / f"age_{MODEL_SHORT}_{strategy}.json"
    if cache.exists():
        c = json.load(open(cache))
        print(f"  Cached: acc={c['accuracy']:.4f}", flush=True)
        return c["accuracy"]

    def process_one(i):
        cid = int(data["cids_test"][i])
        query_emb = data["emb_test"][i]
        messages = [{"role": "system", "content": data["system"]}]

        if strategy == "zero_shot_knn":
            demos = []
        else:  # few_shot_knn
            demos = build_knn_demos(query_emb, data["emb_train"], data["cids_train"],
                                     data["y_train"], data["serialize"],
                                     k_per_class=2, rng=random.Random(42+i))

        for d in demos:
            messages.append({"role": "user", "content": f"{d['profile']}\nClassify age group."})
            messages.append({"role": "assistant",
                "content": json.dumps({"label": d["label"], "confidence": 0.9})})

        profile = data["serialize"](cid)
        user_q = (f"{profile}\n\nClassify age group. Output JSON: "
                 f'{{"label": 0|1|2|3, "confidence": 0-1}}')
        messages.append({"role": "user", "content": user_q})

        return i, call_glm(messages, api_key, seed=42)

    ckpt = OUT / f"age_{MODEL_SHORT}_{strategy}_ckpt.npz"
    if ckpt.exists():
        d = np.load(ckpt)
        preds = list(d["preds"])
        start = len(preds)
        print(f"  Resuming from {start}", flush=True)
    else:
        preds = []; start = 0

    print("  Sanity check...", flush=True)
    for i in range(3):
        pos = int(subset_idx[i])
        _, pred = process_one(pos)
        print(f"    Client {i}: true={data['y_test'][pos]}, pred={pred}", flush=True)

    BATCH = 5
    t0 = time.time()
    for bs in range(start, len(subset_idx), BATCH):
        be = min(bs + BATCH, len(subset_idx))
        with ThreadPoolExecutor(max_workers=BATCH) as ex:
            futs = [ex.submit(process_one, int(subset_idx[i])) for i in range(bs, be)]
            br = {}
            for f in as_completed(futs):
                pos, pred = f.result()
                br[pos] = pred
        for i in range(bs, be):
            preds.append(br[int(subset_idx[i])])

        if len(preds) % 50 < BATCH:
            np.savez(ckpt, preds=np.array(preds))
            y_sub = data["y_test"][subset_idx[:len(preds)]]
            p_arr = np.array(preds)
            mask = p_arr >= 0
            if mask.sum() > 10:
                acc = accuracy_score(y_sub[mask], p_arr[mask])
                rate = (len(preds)-start)/max(time.time()-t0, 0.1)
                print(f"    {len(preds)}/{len(subset_idx)} ({rate:.1f}/s, acc={acc:.4f}, valid={mask.sum()}/{len(preds)})", flush=True)

        if not budget.check():
            np.savez(ckpt, preds=np.array(preds))
            return 0.25

    p_arr = np.array(preds)
    y_sub = data["y_test"][subset_idx[:len(preds)]]
    mask = p_arr >= 0
    acc = accuracy_score(y_sub[mask], p_arr[mask])
    f1 = f1_score(y_sub[mask], p_arr[mask], average="macro")
    print(f"\n  {strategy}: acc={acc:.4f}, macro-f1={f1:.4f} (valid={mask.sum()}/{len(preds)})", flush=True)
    with open(cache, "w") as f:
        json.dump({"accuracy": acc, "macro_f1": f1, "method": strategy, "dataset": "age",
                   "n_test": int(mask.sum()), "n_invalid": int((~mask).sum())}, f, indent=2)
    np.savez(OUT / f"age_{MODEL_SHORT}_{strategy}_preds.npz",
             preds=p_arr, y_test=y_sub, subset_idx=subset_idx)
    if ckpt.exists(): ckpt.unlink()
    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    budget.cap = args.budget
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key: exit("Set OPENROUTER_API_KEY")

    data = load_age_data()
    subset_idx = stratified_subset(data["y_test"], n=args.n, seed=42)
    print(f"Subset: {len(subset_idx)} samples, "
          f"class dist: {np.bincount(data['y_test'][subset_idx]).tolist()}")

    for strat in ["zero_shot_knn", "few_shot_knn"]:
        if not budget.check(): break
        run_strategy(strat, data, subset_idx, api_key)

    print("\n" + "="*60)
    print("FINAL SUMMARY (GLM-4.7 Age 4-class)")
    print("="*60)
