import os, json, time, argparse, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

class BudgetTracker:
    def __init__(self, max_budget_usd=5.0):
        self.max_budget = max_budget_usd
        self.total_spent = 0.0
        self.calls = 0
        self.log_path = Path("results/openrouter/budget_log.json")

    def add(self, input_tokens, output_tokens, model_id):
        pricing = {
            "qwen/qwen-2.5-7b-instruct": (0.10, 0.15),
            "google/gemma-4-26b-a4b-it:free": (0.0, 0.0),
            "z-ai/glm-4.7": (0.10, 0.30),
            "qwen/qwen3.6-plus": (0.10, 0.30),
            "deepseek/deepseek-v3.2-speciale": (0.30, 0.90),
        }
        p_in, p_out = pricing.get(model_id, (1.0, 3.0))
        cost = (input_tokens * p_in + output_tokens * p_out) / 1_000_000
        self.total_spent += cost
        self.calls += 1
        if self.calls % 100 == 0:
            print(f"    [BUDGET] ${self.total_spent:.3f} / ${self.max_budget:.2f} "
                  f"({self.calls} calls)")
        return self.total_spent <= self.max_budget

    def check(self):
        if self.total_spent > self.max_budget:
            print(f"\n    [BUDGET EXCEEDED] ${self.total_spent:.3f} > ${self.max_budget:.2f}. Stopping.")
            return False
        return True

    def summary(self):
        print(f"\n    [BUDGET SUMMARY] Total: ${self.total_spent:.3f}, Calls: {self.calls}")

budget = BudgetTracker(max_budget_usd=5.0)

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier
import shap

OUT = Path("results/openrouter")
OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "qwen25_7b": {
        "id": "qwen/qwen-2.5-7b-instruct",
        "name": "Qwen2.5-7B-Instruct",
        "size": "7B",
        "supports_thinking": False,
    },
    "gemma4_26b": {
        "id": "google/gemma-4-26b-a4b-it:free",
        "name": "Gemma-4-26B-A4B",
        "size": "26B MoE (free)",
        "supports_thinking": False,
    },
    "glm47": {
        "id": "z-ai/glm-4.7",
        "name": "GLM-4.7",
        "size": "~9B",
        "supports_thinking": True,
    },
    "qwen36_35b": {
        "id": "qwen/qwen3.6-plus",
        "name": "Qwen3.6-35B-A3B",
        "size": "35B MoE",
        "supports_thinking": True,
    },
    "deepseek_v3": {
        "id": "deepseek/deepseek-v3.2-speciale",
        "name": "DeepSeek-V3.2-Speciale",
        "size": "671B MoE",
        "supports_thinking": True,
    },

}

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

def call_openrouter_logits(model_id, messages, api_key, pos_label, neg_label, thinking=None, seed=42):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_id,
        "messages": messages,

        "max_tokens": 500 if thinking else (300 if "deepseek" in model_id else 10),
        "temperature": 0,
        "seed": seed,
        **({"logprobs": True, "top_logprobs": 20} if not thinking else {}),
    }

    if "glm" in model_id and not thinking:
        payload["reasoning"] = {"enabled": False}

    for attempt in range(3):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"    HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt == 2:
                    return None
                time.sleep(2)
                continue
            data = resp.json()
            choice = data["choices"][0]
            msg = choice.get("message", {})
            content = msg.get("content") or ""

            if not content.strip() and msg.get("reasoning"):
                reasoning = msg["reasoning"]
                r_lower = reasoning.lower()
                pos_l = pos_label.lower()
                neg_l = neg_label.lower()

                if "answer:" in r_lower:
                    after_answer = r_lower.split("answer:")[-1].strip()
                    if pos_l in after_answer[:20]:
                        content = pos_label
                    elif neg_l in after_answer[:20]:
                        content = neg_label

                if not content.strip():
                    tail = r_lower[-50:]
                    last_pos = tail.rfind(pos_l)
                    last_neg = tail.rfind(neg_l)
                    if last_pos > last_neg:
                        content = pos_label
                    elif last_neg > last_pos:
                        content = neg_label

                if not content.strip():
                    last_pos = r_lower.rfind(pos_l)
                    last_neg = r_lower.rfind(neg_l)
                    if last_pos > last_neg:
                        content = pos_label
                    elif last_neg > last_pos:
                        content = neg_label
                    else:
                        content = ""
            usage = data.get("usage", {})
            budget.add(usage.get("prompt_tokens", 500),
                      usage.get("completion_tokens", 5), model_id)
            if not budget.check():
                return None

            logprobs_data = choice.get("logprobs")
            if logprobs_data and "content" in logprobs_data:
                top_lp = logprobs_data["content"][0].get("top_logprobs", [])
                if top_lp:
                    import math
                    pos_l = pos_label.lower()
                    neg_l = neg_label.lower()
                    pos_prob = 0.0
                    neg_prob = 0.0
                    for lp in top_lp:
                        token = lp["token"].lower().strip()
                        prob = math.exp(lp["logprob"])
                        if pos_l.startswith(token) or token.startswith(pos_l):
                            pos_prob += prob
                        elif neg_l.startswith(token) or token.startswith(neg_l):
                            neg_prob += prob
                    total = pos_prob + neg_prob
                    if total > 0.01:
                        return pos_prob / total

            content_clean = content.lower().strip()
            if "</think>" in content_clean:
                content_clean = content_clean.split("</think>")[-1].strip()

            pos_l = pos_label.lower()
            neg_l = neg_label.lower()
            import re
            conf_match = re.search(r'(\d{1,3})\s*%', content_clean)
            pos_found = pos_l in content_clean
            neg_found = neg_l in content_clean

            if conf_match:
                conf = max(0.05, min(0.95, int(conf_match.group(1)) / 100.0))
                if pos_found and not neg_found:
                    return conf
                elif neg_found and not pos_found:
                    return 1 - conf
                elif pos_found and neg_found:
                    return conf if content_clean.rfind(pos_l) > content_clean.rfind(neg_l) else 1 - conf
                return 0.5
            else:
                if pos_found and not neg_found:
                    return 0.85
                elif neg_found and not pos_found:
                    return 0.15
                elif pos_found and neg_found:
                    return 0.7 if content_clean.rfind(pos_l) > content_clean.rfind(neg_l) else 0.3
                return 0.5

        except Exception as e:
            if attempt == 2:
                print(f"    API error: {e}")
                return None
            time.sleep(2 * (attempt + 1))
    return None

def load_dataset(dataset_name):
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

        def parse_dt(s):
            parts = str(s).split(" ", 1)
            day = int(parts[0])
            if len(parts) > 1:
                t = parts[1].split(":")
                return day, int(t[0])
            return day, 12
        tx[["day", "hour"]] = tx["tr_datetime"].apply(lambda s: pd.Series(parse_dt(s)))

        grouped = tx.groupby("customer_id")
        pos_label, neg_label = "male", "female"
        task_desc = "gender (male or female)"
        answer_fmt = "male or female"
        system_expert = ("You are an expert bank analyst specializing in customer segmentation "
                        "by transaction behavior. You analyze spending patterns, merchant categories, "
                        "transaction frequency and amounts to predict customer demographics.")

        def serialize(cid):
            if cid not in grouped.groups: return "No transaction data available."
            ct = grouped.get_group(cid)
            n = len(ct)
            amt = np.abs(ct["amount"].values)
            cats = ct["mcc_code"].apply(mcc_cat).value_counts()
            top_cats = ", ".join(f"{c} ({cnt} txns, {cnt*100//n}%)" for c, cnt in cats.head(6).items())
            days_span = ct["day"].max() - ct["day"].min()
            months = max(1, days_span // 30)
            daytime = ((ct["hour"] >= 9) & (ct["hour"] < 18)).mean() * 100
            evening = ((ct["hour"] >= 18) & (ct["hour"] < 23)).mean() * 100
            return (f"Client profile:\n"
                    f"- Transactions: {n} over {months} months ({n//months}/month)\n"
                    f"- Spending: avg {amt.mean():.0f} RUB, median {np.median(amt):.0f}, "
                    f"max {amt.max():.0f}\n"
                    f"- Top categories: {top_cats}\n"
                    f"- Time pattern: {daytime:.0f}% daytime, {evening:.0f}% evening")

    elif dataset_name == "rosbank":
        df = pd.read_csv(DATA_DIR / "rosbank_train.csv")
        df["dt"] = pd.to_datetime(df["TRDATETIME"], format="%d%b%y:%H:%M:%S")
        df = df.sort_values(["cl_id", "dt"])
        grouped = df.groupby("cl_id")
        pos_label, neg_label = "churn", "stay"
        task_desc = "churn prediction (will the client leave the bank)"
        answer_fmt = "churn or stay"
        system_expert = ("You are an expert bank analyst specializing in customer retention. "
                        "You analyze transaction patterns, spending behavior and activity trends "
                        "to predict whether a client will churn (leave the bank).")

        def serialize(cid):
            if cid not in grouped.groups: return "No transaction data available."
            ct = grouped.get_group(cid)
            n = len(ct)
            amt = np.abs(ct["amount"].values)
            cats = ct["MCC"].fillna(0).astype(int).apply(mcc_cat).value_counts()
            top_cats = ", ".join(f"{c} ({cnt} txns, {cnt*100//n}%)" for c, cnt in cats.head(6).items())
            days_span = (ct["dt"].max() - ct["dt"].min()).days
            months = max(1, days_span // 30)

            mid = len(ct) // 2
            first_half_avg = np.abs(ct["amount"].values[:mid]).mean()
            second_half_avg = np.abs(ct["amount"].values[mid:]).mean()
            trend = "increasing" if second_half_avg > first_half_avg * 1.1 else \
                    "decreasing" if second_half_avg < first_half_avg * 0.9 else "stable"
            return (f"Client profile:\n"
                    f"- Transactions: {n} over {months} months ({n//months}/month)\n"
                    f"- Spending: avg {amt.mean():.0f} RUB, median {np.median(amt):.0f}\n"
                    f"- Top categories: {top_cats}\n"
                    f"- Activity trend: {trend}")

    else:
        tx = pd.read_csv(DATA_DIR / "transactions_train.csv")
        labels = pd.read_csv(DATA_DIR / "train_target.csv")
        grouped = tx.groupby("client_id")
        pos_label, neg_label = "young", "old"
        task_desc = "age group (0=youngest, 1, 2, 3=oldest)"
        answer_fmt = "0, 1, 2, or 3"
        system_expert = ("You are an expert bank analyst specializing in customer demographics. "
                        "You analyze spending patterns and transaction behavior to predict "
                        "the age group of bank clients.")

        def serialize(cid):
            if cid not in grouped.groups: return "No transaction data available."
            ct = grouped.get_group(cid)
            n = len(ct)
            amt = np.abs(ct["amount_rur"].values)
            return (f"Client profile:\n"
                    f"- Transactions: {n}\n"
                    f"- Spending: avg {amt.mean():.0f} RUB, median {np.median(amt):.0f}")

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

    return {
        "cids_test": cids_test, "y_test": y_test,
        "serialize": serialize, "knn_ctx": knn_ctx,
        "pos_label": pos_label, "neg_label": neg_label,
        "task_desc": task_desc, "answer_fmt": answer_fmt,
        "system_expert": system_expert,
    }

def run_rq3_dir2(dataset_name, model_keys=None, api_key=None, full_matrix=False):
    mode_str = "FULL MATRIX" if full_matrix else "ZERO-SHOT ± kNN"
    print(f"RQ3 DIR2 ({mode_str}): {dataset_name.upper()}")

    data = load_dataset(dataset_name)
    if model_keys is None:
        model_keys = list(MODELS.keys())

    shap_ctx = None
    if full_matrix:
        from xgboost import XGBClassifier
        import shap as shap_lib
        cids_train = np.load(f"embeddings/{dataset_name}/cids_train_seed42.npy")
        y_train = np.load(f"embeddings/{dataset_name}/y_train_seed42.npy")

        if dataset_name == "gender":
            tx = pd.read_csv(DATA_DIR / "transactions.csv")
            labels = pd.read_csv(DATA_DIR / "gender_train.csv")
            tx = tx[tx["customer_id"].isin(labels["customer_id"])].copy()
            grouped_shap = tx.groupby("customer_id")
        elif dataset_name == "rosbank":
            df = pd.read_csv(DATA_DIR / "rosbank_train.csv")
            grouped_shap = df.groupby("cl_id")
        feat_names = ["n_tx", "mean_amt", "std_amt", "median_amt", "n_mcc"]

        def agg_feat(cids, grp):
            recs = []
            for cid in cids:
                if cid not in grp.groups:
                    recs.append({"n":0,"mean":0,"std":0,"med":0,"nmcc":0})
                    continue
                ct = grp.get_group(cid)
                a = ct.iloc[:, -1].values if dataset_name == "rosbank" else ct["amount"].values
                a = np.abs(a)
                recs.append({"n":len(ct),"mean":a.mean(),"std":a.std(),"med":np.median(a),
                             "nmcc": ct.iloc[:, 2].nunique()})
            return np.array(pd.DataFrame(recs).values, dtype=np.float32)

        X_tr = agg_feat(cids_train, grouped_shap)
        X_te = agg_feat(data["cids_test"], grouped_shap)
        xgb = XGBClassifier(n_estimators=300, max_depth=6, random_state=42, verbosity=0)
        xgb.fit(X_tr, y_train)
        test_preds_prob = xgb.predict_proba(X_te)[:, 1]
        explainer = shap_lib.TreeExplainer(xgb)
        sv = explainer.shap_values(X_te)

        shap_ctx = {}
        for i, cid in enumerate(data["cids_test"]):
            pred = data["pos_label"] if test_preds_prob[i] > 0.5 else data["neg_label"]
            conf = test_preds_prob[i] if test_preds_prob[i] > 0.5 else 1 - test_preds_prob[i]
            svi = sv[i] if sv.ndim == 2 else sv[1][i]
            top_idx = np.argsort(np.abs(svi))[::-1][:3]
            factors = [f"{feat_names[int(j)]}={X_te[i,int(j)]:.0f} "
                      f"({'supports' if svi[int(j)]>0 else 'against'} {data['pos_label']})"
                      for j in top_idx]
            shap_ctx[cid] = {"pred": pred, "conf": f"{conf*100:.0f}%",
                            "factors": "; ".join(factors)}
        print("  SHAP context built.")

    few_shot_text = None
    if full_matrix:
        y_train = np.load(f"embeddings/{dataset_name}/y_train_seed42.npy")
        cids_train = np.load(f"embeddings/{dataset_name}/cids_train_seed42.npy")
        pos_cid = int(cids_train[y_train == 1][0])
        neg_cid = int(cids_train[y_train == 0][0])
        few_shot_text = (f"Example 1:\n{data['serialize'](pos_cid)}\n"
                        f"Answer: {data['pos_label']}\n\n"
                        f"Example 2:\n{data['serialize'](neg_cid)}\n"
                        f"Answer: {data['neg_label']}\n\n")

    if full_matrix:
        strategies = ["zero_shot", "few_shot", "cot"]
        enrichments = ["none", "shap", "knn", "both"]
    else:
        strategies = ["zero_shot"]
        enrichments = ["none", "knn"]

    results = {}
    for mk in model_keys:
        m = MODELS[mk]
        print(f"\n  --- {m['name']} ({m['size']}) ---")

        for strategy in strategies:
            for enrichment in enrichments:
                tag = f"{mk}_{strategy}_{enrichment}"
                cache_file = OUT / f"{dataset_name}_{tag}.json"
                if cache_file.exists():
                    cached = json.load(open(cache_file))
                    results[tag] = cached["auc"]
                    print(f"  {strategy}×{enrichment}: AUC={cached['auc']:.4f} (cached)")
                    continue

                has_enrichment = enrichment != "none"
                if has_enrichment:
                    system = (f"{data['system_expert']} "
                             f"You also have analysis from an ML model.")
                else:
                    system = data['system_expert']

                checkpoint_file = OUT / f"{dataset_name}_{tag}_checkpoint.npz"
                if checkpoint_file.exists():
                    ckpt = np.load(checkpoint_file)
                    preds = list(ckpt["preds"])
                    start_idx = len(preds)
                    print(f"  Resuming {strategy}×{enrichment}: {start_idx}/{len(data['cids_test'])}")
                else:
                    preds = []
                    start_idx = 0

                def build_messages(i):
                    cid = data["cids_test"][i]
                    profile = data["serialize"](int(cid))
                    enrich_text = ""
                    if enrichment in ["shap", "both"] and shap_ctx:
                        s = shap_ctx[cid]
                        enrich_text += (f"\nML model: predicts {s['pred']} ({s['conf']} confidence).\n"
                                       f"Key factors: {s['factors']}.\n")
                    if enrichment in ["knn", "both"]:
                        k = data["knn_ctx"][cid]
                        enrich_text += (f"\nSimilar clients: {k['pos']} {data['pos_label']}, "
                                       f"{k['neg']} {data['neg_label']} "
                                       f"(majority: {k['majority']}).\n")
                    answer_instruction = f"\n\nRespond with ONLY one word: {data['answer_fmt']}. Write your answer after 'ANSWER:'"
                    if strategy == "zero_shot":
                        user_msg = f"{profile}{enrich_text}{answer_instruction}"
                    elif strategy == "few_shot":
                        user_msg = f"{few_shot_text}Now predict:\n{profile}{enrich_text}{answer_instruction}"
                    elif strategy == "cot":
                        user_msg = (f"{profile}{enrich_text}\n"
                                   f"Analyze the transaction patterns step by step, "
                                   f"then give your final prediction: {data['answer_fmt']}.")
                    return [{"role": "system", "content": system},
                            {"role": "user", "content": user_msg}]

                thinking = (strategy == "cot")
                BATCH_SIZE = 5
                t0 = time.time()

                for batch_start in range(start_idx, len(data["cids_test"]), BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, len(data["cids_test"]))
                    batch_indices = list(range(batch_start, batch_end))

                    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
                        futures = {}
                        for idx in batch_indices:
                            msgs = build_messages(idx)
                            f = executor.submit(call_openrouter_logits, m["id"], msgs,
                                                api_key, data["pos_label"], data["neg_label"],
                                                thinking)
                            futures[f] = idx

                        batch_results = {}
                        for f in as_completed(futures):
                            batch_results[futures[f]] = f.result()

                    for idx in batch_indices:
                        preds.append(batch_results[idx])

                    if len(preds) % 100 < BATCH_SIZE:
                        np.savez(checkpoint_file, preds=np.array(preds))
                        elapsed = time.time() - t0
                        partial_auc = roc_auc_score(data["y_test"][:len(preds)], preds)
                        print(f"    {len(preds)}/{len(data['cids_test'])} "
                              f"({(len(preds)-start_idx)/max(elapsed,0.1):.1f}/s, AUC={partial_auc:.4f}) "
                              f"[checkpoint]")

                    if not budget.check():
                        np.savez(checkpoint_file, preds=np.array(preds))
                        print(f"  Budget exceeded at {len(preds)}/{len(data['cids_test'])}")
                        break

                auc = roc_auc_score(data["y_test"], preds)
                results[tag] = auc
                print(f"  {m['name']} {strategy}×{enrichment}: AUC={auc:.4f}")

                with open(cache_file, "w") as f:
                    json.dump({"model": m["name"], "size": m["size"],
                              "strategy": strategy, "enrichment": enrichment,
                              "auc": auc, "dataset": dataset_name,
                              "n_test": len(preds)}, f, indent=2)
                np.savez(OUT / f"{dataset_name}_{tag}_preds.npz",
                         preds=np.array(preds),
                         cids=data["cids_test"],
                         y_test=data["y_test"])
                if checkpoint_file.exists():
                    checkpoint_file.unlink()

    print(f"RQ3 DIR2 SUMMARY ({dataset_name})")
    if full_matrix:
        for mk in model_keys:
            m = MODELS[mk]
            print(f"\n  {m['name']} ({m['size']}):")
            print(f"  {'Strategy':<12} {'None':>8} {'+ SHAP':>8} {'+ kNN':>8} {'+ Both':>8}")
            for strat in strategies:
                row = []
                for enr in enrichments:
                    val = results.get(f"{mk}_{strat}_{enr}")
                    row.append(f"{val:.4f}" if val else "?")
                print(f"  {strat:<12} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8}")
    else:
        print(f"{'Model':<25} {'Size':<10} {'No enrich':>10} {'+ kNN CoT':>10} {'Δ':>8}")
        for mk in model_keys:
            m = MODELS[mk]
            none_auc = results.get(f"{mk}_zero_shot_none")
            knn_auc = results.get(f"{mk}_zero_shot_knn")
            none_str = f"{none_auc:.4f}" if none_auc else "?"
            knn_str = f"{knn_auc:.4f}" if knn_auc else "?"
            delta = f"+{(knn_auc-none_auc)*100:.0f}pp" if none_auc and knn_auc else "?"
            print(f"  {m['name']:<25} {m['size']:<10} {none_str:>10} {knn_str:>10} {delta:>8}")

    with open(OUT / f"{dataset_name}_rq3_dir2.json", "w") as f:
        json.dump(results, f, indent=2)
    return results

def run_rq3_cot_effect(dataset_name, api_key=None):
    print(f"RQ3 COT EFFECT: {dataset_name.upper()}")

    data = load_dataset(dataset_name)
    thinking_models = ["glm47", "qwen36_35b", "deepseek_v3"]
    results = {}

    for mk in thinking_models:
        m = MODELS[mk]
        for thinking in [False, True]:
            mode = "thinking_on" if thinking else "thinking_off"
            tag = f"{mk}_{mode}"
            cache_file = OUT / f"{dataset_name}_{tag}_knn.json"
            if cache_file.exists():
                cached = json.load(open(cache_file))
                results[tag] = cached["auc"]
                print(f"  {m['name']} {mode}: AUC={cached['auc']:.4f} (cached)")
                continue

            print(f"\n  --- {m['name']} {mode} + kNN CoT ---")

            if thinking:
                system = (f"You are a bank analyst predicting client {data['task_desc']}. "
                         f"Think step by step about the transaction patterns, then {data['answer_fmt']}.")
            else:
                system = (f"You are a bank analyst predicting client {data['task_desc']}. "
                         f"{data['answer_fmt']}. Do not explain, just answer.")

            checkpoint_file = OUT / f"{dataset_name}_{tag}_knn_checkpoint.npz"

            if checkpoint_file.exists():
                ckpt = np.load(checkpoint_file)
                preds = list(ckpt["preds"])
                start_idx = len(preds)
                print(f"  Resuming from checkpoint: {start_idx}/{len(data['cids_test'])}")
            else:
                preds = []
                start_idx = 0

            t0 = time.time()
            for i in range(start_idx, len(data["cids_test"])):
                cid = data["cids_test"][i]
                profile = data["serialize"](int(cid))
                k = data["knn_ctx"][cid]
                user_msg = (f"Profile:\n{profile}\n\n"
                           f"Similar clients: {k['pos']} {data['pos_label']}, "
                           f"{k['neg']} {data['neg_label']} "
                           f"(majority: {k['majority']}).\n\nPredict.")

                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ]

                prob = call_openrouter_logits(m["id"], messages, api_key,
                                              data["pos_label"], data["neg_label"],
                                              thinking=thinking)
                preds.append(prob)

                if (i+1) % 100 == 0:
                    np.savez(checkpoint_file, preds=np.array(preds))
                    elapsed = time.time() - t0
                    partial_auc = roc_auc_score(data["y_test"][:len(preds)], preds)
                    print(f"    {len(preds)}/{len(data['cids_test'])} "
                          f"({(len(preds)-start_idx)/elapsed:.1f}/s, AUC={partial_auc:.4f}) "
                          f"[checkpoint saved]")

                if not budget.check():
                    np.savez(checkpoint_file, preds=np.array(preds))
                    print(f"  Budget exceeded. Checkpoint saved at {len(preds)}/{len(data['cids_test'])}")
                    break

            auc = roc_auc_score(data["y_test"], preds)
            results[tag] = auc
            print(f"  {m['name']} {mode}: AUC={auc:.4f}")

            with open(cache_file, "w") as f:
                json.dump({"model": m["name"], "mode": mode, "auc": auc,
                          "dataset": dataset_name}, f, indent=2)
            np.savez(OUT / f"{dataset_name}_{tag}_knn_preds.npz",
                     preds=np.array(preds),
                     cids=data["cids_test"],
                     y_test=data["y_test"])
            if checkpoint_file.exists():
                checkpoint_file.unlink()

    print(f"COT EFFECT SUMMARY ({dataset_name})")
    for mk in thinking_models:
        m = MODELS[mk]
        off = results.get(f"{mk}_thinking_off", None)
        on = results.get(f"{mk}_thinking_on", None)
        off_str = f"{off:.4f}" if off else "?"
        on_str = f"{on:.4f}" if on else "?"
        delta = f"{(on-off)*100:+.1f}pp" if off and on else "?"
        print(f"  {m['name']:<25} off={off_str}  on={on_str}  Δ={delta}")

    with open(OUT / f"{dataset_name}_rq3_cot.json", "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True,
                    choices=["rq3_dir2", "rq3_cot", "all"])
    ap.add_argument("--dataset", default="gender")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Model keys: qwen25_7b, gemma4_26b, glm47, qwen36_35b, deepseek_v3")
    ap.add_argument("--budget", type=float, default=12.0,
                    help="Max budget in USD (default: $12.00)")
    ap.add_argument("--full-matrix", action="store_true",
                    help="Run full 3×4 matrix (zero-shot/few-shot/CoT × none/shap/knn/both)")
    args = ap.parse_args()

    budget.max_budget = args.budget
    print(f"Budget limit: ${args.budget:.2f}")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        print("  export OPENROUTER_API_KEY=sk-or-...")
        exit(1)

    if args.experiment == "rq3_dir2":
        run_rq3_dir2(args.dataset, args.models, api_key, full_matrix=args.full_matrix)
    elif args.experiment == "rq3_cot":
        run_rq3_cot_effect(args.dataset, api_key)
    elif args.experiment == "all":
        run_rq3_dir2(args.dataset, args.models, api_key, full_matrix=args.full_matrix)
        run_rq3_cot_effect(args.dataset, api_key)
