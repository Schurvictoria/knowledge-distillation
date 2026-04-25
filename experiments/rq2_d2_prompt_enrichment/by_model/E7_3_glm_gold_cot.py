#!/usr/bin/env python3
"""
Gold-standard CoT for GLM-4.7.

Full academic pipeline:
1. Few-shot CoT (Wei et al. 2022) — 4 reasoning demonstrations
2. Self-consistency (Wang et al. 2022) — k=5 samples, temperature=0.7
3. Majority vote → label; count/k → confidence
4. JSON structured output for reliable parsing

Cost: ~5x more than single-shot (k=5 samples per client).
"""
import os, json, time, re, requests, argparse
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from run_openrouter_experiments import load_dataset, MODELS, budget, OUT

# ---- Reproducibility (seed=42) ----
import random as _random, os as _os
_SEED = 42
_random.seed(_SEED); np.random.seed(_SEED)
_os.environ["PYTHONHASHSEED"] = str(_SEED)



OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_glm_sample(messages, api_key, reasoning_on, pos_label, neg_label, temperature=0.7):
    """Single sample from GLM-4.7. Returns label or None."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "label": {"type": "string", "enum": [pos_label, neg_label]},
        },
        "required": ["label"],
        "additionalProperties": False,
    }

    payload = {
        "model": "z-ai/glm-4.7",
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_schema",
            "json_schema": {"name": "cls", "strict": True, "schema": schema}},
    }
    if reasoning_on:
        payload["max_tokens"] = 2500
        payload["reasoning"] = {"max_tokens": 1500}
    else:
        payload["max_tokens"] = 400
        payload["reasoning"] = {"enabled": False}

    for attempt in range(3):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1)); continue
            if resp.status_code != 200:
                if attempt == 2: return None
                time.sleep(2); continue

            data = resp.json()
            msg = data["choices"][0].get("message", {})
            content = msg.get("content") or ""
            usage = data.get("usage", {})
            budget.add(usage.get("prompt_tokens", 500),
                      usage.get("completion_tokens", 20), "z-ai/glm-4.7")
            if not budget.check(): return None

            # Parse JSON from content or reasoning
            for source in [content, msg.get("reasoning", "") or ""]:
                try:
                    parsed = json.loads(source)
                    if "label" in parsed:
                        return parsed["label"].lower()
                except (json.JSONDecodeError, ValueError, TypeError):
                    m = re.search(r'\{[^{}]*"label"[^{}]*\}', source)
                    if m:
                        try:
                            return json.loads(m.group())["label"].lower()
                        except: pass

            # Fallback regex
            combined = (content + " " + (msg.get("reasoning") or "")).lower()
            m = re.search(rf'"label"[:\s]*"({pos_label.lower()}|{neg_label.lower()})"', combined)
            if m: return m.group(1)

            # Last resort
            if pos_label.lower() in combined[-100:] and neg_label.lower() not in combined[-100:]:
                return pos_label.lower()
            if neg_label.lower() in combined[-100:]:
                return neg_label.lower()
            return None

        except Exception as e:
            if attempt == 2: return None
            time.sleep(2)
    return None


def build_few_shot_demos(data, n_examples=4):
    """Build few-shot CoT demonstrations from train set."""
    cids_train = np.load(f"embeddings/gender/cids_train_seed42.npy")
    y_train = np.load(f"embeddings/gender/y_train_seed42.npy")

    # Pick 2 positive + 2 negative examples
    pos_cids = cids_train[y_train == 1][:n_examples//2]
    neg_cids = cids_train[y_train == 0][:n_examples//2]

    demos = []
    for cid, label_int in list(zip(pos_cids, [1]*len(pos_cids))) + list(zip(neg_cids, [0]*len(neg_cids))):
        profile = data["serialize"](int(cid))
        label = data["pos_label"] if label_int == 1 else data["neg_label"]

        # Simple reasoning template
        if label_int == 1:
            reasoning = f"Transaction patterns (categories, amounts) suggest {data['pos_label']} behavior."
        else:
            reasoning = f"Transaction patterns (categories, amounts) suggest {data['neg_label']} behavior."

        demo = {
            "user": profile + "\n\nClassify this client.",
            "assistant": json.dumps({"reasoning": reasoning, "label": label})
        }
        demos.append(demo)

    return demos


def classify_with_sc(cid_idx, data, api_key, reasoning_on, k_samples, few_shot_demos):
    """Self-consistency: k samples, majority vote."""
    cid = data["cids_test"][cid_idx]
    profile = data["serialize"](int(cid))
    knn = data["knn_ctx"][cid]
    enrich = (f"\nSimilar clients: {knn['pos']} {data['pos_label']}, "
             f"{knn['neg']} {data['neg_label']} (majority: {knn['majority']}).")

    messages = []
    system = f"{data['system_expert']} You also have analysis from an ML model."
    messages.append({"role": "system", "content": system})

    # Few-shot demonstrations
    for demo in few_shot_demos:
        messages.append({"role": "user", "content": demo["user"]})
        messages.append({"role": "assistant", "content": demo["assistant"]})

    # Final test query
    user = (f"{profile}{enrich}\n\n"
           f"Classify this client. Output JSON: "
           f'{{"reasoning": "analysis", "label": "{data["pos_label"]}" or "{data["neg_label"]}"}}')
    messages.append({"role": "user", "content": user})

    # k samples in parallel
    with ThreadPoolExecutor(max_workers=k_samples) as ex:
        futs = [ex.submit(call_glm_sample, messages, api_key, reasoning_on,
                          data["pos_label"], data["neg_label"], 0.7)
                for _ in range(k_samples)]
        labels = [f.result() for f in as_completed(futs)]

    # Majority vote + confidence
    labels = [l for l in labels if l is not None]
    if not labels:
        return 0.5

    counter = Counter(labels)
    pos_count = counter.get(data["pos_label"].lower(), 0)
    total = len(labels)
    # Soft probability = count(pos) / total
    return pos_count / total


def run_sc_experiment(dataset_name, reasoning_on, api_key, k=5, n_shots=4):
    mode = "on" if reasoning_on else "off"
    print(f"\n=== GOLD CoT: GLM-4.7 reasoning={mode}, SC k={k}, {n_shots}-shot ===", flush=True)

    data = load_dataset(dataset_name)
    demos = build_few_shot_demos(data, n_shots)
    print(f"  Built {len(demos)} few-shot demos", flush=True)

    cache = OUT / f"{dataset_name}_glm47_gold_cot_{mode}_k{k}.json"
    if cache.exists():
        c = json.load(open(cache))
        print(f"  Cached: AUC={c['auc']:.4f}", flush=True)
        return c["auc"]

    ckpt = OUT / f"{dataset_name}_glm47_gold_cot_{mode}_k{k}_ckpt.npz"
    if ckpt.exists():
        preds = list(np.load(ckpt)["preds"])
        start = len(preds)
        print(f"  Resuming from {start}/{len(data['cids_test'])}", flush=True)
    else:
        preds = []; start = 0

    t0 = time.time()
    for i in range(start, len(data["cids_test"])):
        prob = classify_with_sc(i, data, api_key, reasoning_on, k, demos)
        preds.append(prob)

        if (i + 1) % 25 == 0:
            np.savez(ckpt, preds=np.array(preds))
            auc = roc_auc_score(data["y_test"][:len(preds)], preds)
            rate = (len(preds) - start) / max(time.time() - t0, 0.1)
            unique = len(set(round(p, 3) for p in preds))
            print(f"    {len(preds)}/{len(data['cids_test'])} "
                  f"({rate:.2f}/s, AUC={auc:.4f}, {unique} unique)", flush=True)

        if not budget.check():
            np.savez(ckpt, preds=np.array(preds))
            return 0.5

    auc = roc_auc_score(data["y_test"], preds)
    print(f"  GLM-4.7 reasoning={mode} SC k={k}: AUC={auc:.4f}", flush=True)

    with open(cache, "w") as f:
        json.dump({"auc": auc, "n_test": len(preds), "reasoning": mode,
                  "method": "gold_cot_sc", "k": k, "n_shots": n_shots}, f, indent=2)
    np.savez(OUT / f"{dataset_name}_glm47_gold_cot_{mode}_k{k}_preds.npz",
             preds=np.array(preds), cids=data["cids_test"], y_test=data["y_test"])
    if ckpt.exists(): ckpt.unlink()
    return auc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gender")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    budget.max_budget = args.budget
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key: exit("Set OPENROUTER_API_KEY")

    # Only reasoning=off first (cheaper)
    auc_off = run_sc_experiment(args.dataset, False, api_key, k=args.k)
    print(f"\n{'='*50}")
    print(f"GOLD CoT GLM-4.7 reasoning=off: AUC={auc_off:.4f}")
    print(f"(vs single-shot: 0.7712)")
