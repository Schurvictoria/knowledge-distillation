import os, json, time, re, random, requests, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import NearestNeighbors

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from run_openrouter_experiments import load_dataset, budget, OUT
from distil.llm import build_cot_reasoning

import random as _random, os as _os
_SEED = 42
_random.seed(_SEED); np.random.seed(_SEED)
_os.environ["PYTHONHASHSEED"] = str(_SEED)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "qwen/qwen3.6-plus"
MODEL_SHORT = "qwen36"

def call_model(messages, api_key, pos_label, neg_label, cot=False, seed=42):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    props = {
        "label": {"type": "string", "enum": [pos_label, neg_label]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    if cot:
        props = {"reasoning": {"type": "string"}, **props}
    schema = {
        "type": "object",
        "properties": props,
        "required": ["label", "confidence"],
        "additionalProperties": False,
    }
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.6,
        "seed": seed,
        "reasoning": {"max_tokens": 3000},
        "response_format": {"type": "json_schema",
            "json_schema": {"name": "cls", "strict": True, "schema": schema}},
    }

    for attempt in range(3):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=180)
            if resp.status_code == 429: time.sleep(5 * (attempt+1)); continue
            if resp.status_code == 402:
                print(f"    [402 CREDITS EXHAUSTED] Halting to avoid contamination.")
                import os; os._exit(2)

            if resp.status_code != 200:
                if attempt == 2: return 0.5
                time.sleep(2); continue
            data = resp.json()
            msg = data["choices"][0].get("message", {})
            content = msg.get("content") or ""
            usage = data.get("usage", {})
            budget.add(usage.get("prompt_tokens", 500),
                      usage.get("completion_tokens", 100), MODEL_ID)
            if not budget.check(): return 0.5

            try:
                parsed = json.loads(content)
                label = parsed["label"].lower()
                conf = max(0.05, min(0.95, float(parsed.get("confidence", 0.85))))
                return conf if label == pos_label.lower() else 1 - conf
            except:
                combined = content + " " + (msg.get("reasoning") or "")
                m = re.search(r'\{[^{}]*"label"[^{}]*\}', combined)
                if m:
                    try:
                        parsed = json.loads(m.group())
                        label = parsed["label"].lower()
                        conf = max(0.05, min(0.95, float(parsed.get("confidence", 0.85))))
                        return conf if label == pos_label.lower() else 1 - conf
                    except: pass
                c = combined.lower()
                lp, ln = c.rfind(pos_label.lower()), c.rfind(neg_label.lower())
                if lp > ln: return 0.85
                elif ln > lp: return 0.15
                return 0.5
        except:
            if attempt == 2: return 0.5
            time.sleep(2)
    return 0.5

def build_random_demos(pos_cids, neg_cids, serialize, pos_label, neg_label, k_per_class=1, rng=None):
    if rng is None: rng = random.Random(42)
    pos_chosen = rng.sample(list(pos_cids), k_per_class)
    neg_chosen = rng.sample(list(neg_cids), k_per_class)
    demos = []
    for cid in pos_chosen:
        demos.append({"profile": serialize(int(cid)), "label": pos_label})
    for cid in neg_chosen:
        demos.append({"profile": serialize(int(cid)), "label": neg_label})
    rng.shuffle(demos)
    return demos

def build_knn_demos(query_emb, train_emb, train_cids, y_train, serialize,
                     pos_label, neg_label, k_per_class=2, rng=None):
    sc = MaxAbsScaler()
    tr_scaled = sc.fit_transform(train_emb)
    q_scaled = sc.transform(query_emb.reshape(1, -1))
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    def top_k(indices, k):
        nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(tr_scaled[indices])
        _, idxs = nn.kneighbors(q_scaled)
        return indices[idxs[0]]
    demos = []
    for idx in top_k(pos_idx, k_per_class):
        demos.append({"profile": serialize(int(train_cids[idx])), "label": pos_label})
    for idx in top_k(neg_idx, k_per_class):
        demos.append({"profile": serialize(int(train_cids[idx])), "label": neg_label})
    if rng: rng.shuffle(demos)
    return demos

def run_strategy(strategy, dataset_name, api_key):
    data = load_dataset(dataset_name)
    train_emb = np.load(f"embeddings/{dataset_name}/emb_train_seed42.npy")
    test_emb = np.load(f"embeddings/{dataset_name}/emb_test_seed42.npy")
    train_cids = np.load(f"embeddings/{dataset_name}/cids_train_seed42.npy")
    y_train = np.load(f"embeddings/{dataset_name}/y_train_seed42.npy")
    pos_cids = train_cids[y_train == 1]
    neg_cids = train_cids[y_train == 0]

    cache = OUT / f"{dataset_name}_{MODEL_SHORT}_{strategy}.json"
    if cache.exists():
        c = json.load(open(cache))
        print(f"  Cached: AUC={c['auc']:.4f}")
        return c["auc"]

    def process_one(i):
        cid = int(data["cids_test"][i])
        query_emb = test_emb[i]

        messages = [{"role": "system", "content": data["system_expert"]}]

        if strategy == "zero_shot_knn":
            demos = []
        elif strategy == "few_shot_random":
            demos = build_random_demos(pos_cids, neg_cids, data["serialize"],
                                        data["pos_label"], data["neg_label"],
                                        k_per_class=1, rng=random.Random(42+i))
        else:
            demos = build_knn_demos(query_emb, train_emb, train_cids, y_train,
                                     data["serialize"], data["pos_label"], data["neg_label"],
                                     k_per_class=2, rng=random.Random(42+i))

        is_cot = (strategy == "few_shot_cot_knn")
        for d in demos:
            messages.append({"role": "user", "content": f"{d['profile']}\nClassify this client."})
            if is_cot:
                reasoning = build_cot_reasoning(d["profile"], d["label"])
                messages.append({"role": "assistant", "content": json.dumps({
                    "reasoning": reasoning, "label": d["label"], "confidence": 0.9})})
            else:
                messages.append({"role": "assistant",
                    "content": json.dumps({"label": d["label"], "confidence": 0.9})})

        profile = data["serialize"](cid)
        k = data["knn_ctx"][cid]
        enrich = (f"\nSimilar clients: {k['pos']} {data['pos_label']}, "
                 f"{k['neg']} {data['neg_label']}.")
        if is_cot:
            user_q = (f"{profile}{enrich}\n\nClassify this client. Output JSON: "
                     f'{{"reasoning": "analysis (≤80 words)", "label": "{data["pos_label"]}" or "{data["neg_label"]}", "confidence": 0-1}}')
        else:
            user_q = (f"{profile}{enrich}\n\nClassify this client. Output JSON: "
                     f'{{"label": "{data["pos_label"]}" or "{data["neg_label"]}", "confidence": 0-1}}')
        messages.append({"role": "user", "content": user_q})

        return i, call_model(messages, api_key, data["pos_label"], data["neg_label"], cot=is_cot, seed=42)

    ckpt = OUT / f"{dataset_name}_{MODEL_SHORT}_{strategy}_ckpt.npz"
    if ckpt.exists():
        preds = list(np.load(ckpt)["preds"])
        start = len(preds)
        print(f"  Resuming from {start}")
    else:
        preds = []; start = 0

    if start == 0:
        print("  Sanity check...")
        for i in range(3):
            _, prob = process_one(i)
            true = data["pos_label"] if data["y_test"][i] == 1 else data["neg_label"]
            print(f"    Client {i}: true={true}, prob={prob:.3f}")

    BATCH = 5
    t0 = time.time()
    for bs in range(start, len(data["cids_test"]), BATCH):
        be = min(bs + BATCH, len(data["cids_test"]))
        with ThreadPoolExecutor(max_workers=BATCH) as ex:
            futs = [ex.submit(process_one, i) for i in range(bs, be)]
            br = {}
            for f in as_completed(futs):
                i, p = f.result()
                br[i] = p
        for i in range(bs, be):
            preds.append(br[i])

        if len(preds) % 50 < BATCH:
            np.savez(ckpt, preds=np.array(preds))
            auc = roc_auc_score(data["y_test"][:len(preds)], preds)
            rate = (len(preds)-start) / max(time.time()-t0, 0.1)
            print(f"    {len(preds)}/{len(data['cids_test'])} ({rate:.1f}/s, AUC={auc:.4f})")

        if not budget.check():
            np.savez(ckpt, preds=np.array(preds))
            return 0.5

    auc = roc_auc_score(data["y_test"], preds)
    print(f"\n  {strategy} on {dataset_name}: AUC={auc:.4f}")
    with open(cache, "w") as f:
        json.dump({"auc": auc, "method": strategy, "dataset": dataset_name, "n_test": len(preds)}, f, indent=2)
    np.savez(OUT / f"{dataset_name}_{MODEL_SHORT}_{strategy}_preds.npz",
             preds=np.array(preds), cids=data["cids_test"], y_test=data["y_test"])
    if ckpt.exists(): ckpt.unlink()
    return auc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["rosbank"])
    ap.add_argument("--budget", type=float, default=1.0)
    args = ap.parse_args()

    budget.max_budget = args.budget
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key: exit("Set OPENROUTER_API_KEY")

    for dataset in args.datasets:
        for strat in ["zero_shot_knn", "few_shot_random", "few_shot_knn", "few_shot_cot_knn"]:
            if not budget.check(): break
            run_strategy(strat, dataset, api_key)

    print(f"FINAL SUMMARY (Qwen3.6-Plus, max_tokens=4096)")
