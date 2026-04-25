#!/usr/bin/env python3
"""
Proper CoT for GLM-4.7 using JSON schema + correct reasoning config.

Fixes from previous approach:
1. JSON schema response_format (guaranteed parsing)
2. max_tokens=2500 for reasoning=True (not 500)
3. reasoning: {max_tokens: 1500} — explicit reasoning budget
4. Don't say "step by step" when reasoning=True
5. Fallback parsing: JSON → regex → rfind

Tests on Gender: zero-shot + kNN, with reasoning off/on.
"""
import os, json, time, re, requests, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from run_openrouter_experiments import load_dataset, MODELS, budget, OUT

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_glm_proper(messages, api_key, reasoning_on, pos_label, neg_label):
    """Call GLM-4.7 with proper CoT pipeline."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # JSON schema for guaranteed parsing
    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "description": "Your analysis of the client"},
            "label": {"type": "string", "enum": [pos_label, neg_label]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["label", "confidence"],
        "additionalProperties": False,
    }

    payload = {
        "model": "z-ai/glm-4.7",
        "messages": messages,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "classification", "strict": True, "schema": schema},
        },
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
                print(f"    HTTP {resp.status_code}: {resp.text[:200]}", flush=True)
                if attempt == 2: return 0.5
                time.sleep(2); continue

            data = resp.json()
            choice = data["choices"][0]
            msg = choice.get("message", {})
            content = msg.get("content") or ""
            usage = data.get("usage", {})
            budget.add(usage.get("prompt_tokens", 500),
                      usage.get("completion_tokens", 20), "z-ai/glm-4.7")
            if not budget.check(): return 0.5

            # Tier 1: JSON parse (primary)
            for source in [content, msg.get("reasoning", "") or ""]:
                try:
                    # Try direct JSON
                    parsed = json.loads(source)
                    if "label" in parsed:
                        label = parsed["label"].lower()
                        conf = float(parsed.get("confidence", 0.85))
                        conf = max(0.05, min(0.95, conf))
                        if label == pos_label.lower(): return conf
                        elif label == neg_label.lower(): return 1 - conf
                except (json.JSONDecodeError, ValueError, TypeError):
                    # Tier 2: extract JSON from text
                    m = re.search(r'\{[^{}]*"label"[^{}]*\}', source)
                    if m:
                        try:
                            parsed = json.loads(m.group())
                            label = parsed["label"].lower()
                            conf = max(0.05, min(0.95, float(parsed.get("confidence", 0.85))))
                            if label == pos_label.lower(): return conf
                            elif label == neg_label.lower(): return 1 - conf
                        except: pass

            # Tier 3: regex on answer marker
            combined = (content + " " + (msg.get("reasoning") or "")).lower()
            m = re.search(rf'(?:label|answer)["\s:]+({pos_label.lower()}|{neg_label.lower()})', combined)
            if m:
                return 0.85 if m.group(1) == pos_label.lower() else 0.15

            # Tier 4: last word in content
            if content.strip():
                c = content.lower().strip()
                if pos_label.lower() in c and neg_label.lower() not in c: return 0.85
                if neg_label.lower() in c and pos_label.lower() not in c: return 0.15

            # Tier 5: rfind fallback
            if combined:
                lp, ln = combined.rfind(pos_label.lower()), combined.rfind(neg_label.lower())
                if lp > ln: return 0.85
                elif ln > lp: return 0.15

            return 0.5

        except Exception as e:
            if attempt == 2:
                print(f"    Error: {e}", flush=True)
                return 0.5
            time.sleep(2)
    return 0.5


def run_glm_kncot(dataset_name, reasoning_on, api_key):
    """Run GLM-4.7 with kNN CoT enrichment, proper config."""
    mode = "on" if reasoning_on else "off"
    print(f"\n=== GLM-4.7 reasoning={mode} + kNN (PROPER CoT) ===", flush=True)

    data = load_dataset(dataset_name)
    cache = OUT / f"{dataset_name}_glm47_proper_cot_{mode}.json"
    if cache.exists():
        c = json.load(open(cache))
        print(f"  Cached: AUC={c['auc']:.4f}", flush=True)
        return c["auc"]

    def build_msg(i):
        cid = data["cids_test"][i]
        profile = data["serialize"](int(cid))
        k = data["knn_ctx"][cid]
        enrich = (f"\nSimilar clients: {k['pos']} {data['pos_label']}, "
                 f"{k['neg']} {data['neg_label']} (majority: {k['majority']}).")

        if reasoning_on:
            # Don't say "step by step" — GLM reasons naturally
            system = f"{data['system_expert']} You also have analysis from an ML model."
            user = (f"{profile}{enrich}\n\n"
                   f"Classify this client. Output JSON: "
                   f"{{\"label\": \"{data['pos_label']}\" or \"{data['neg_label']}\", "
                   f"\"confidence\": 0-1}}")
        else:
            # With reasoning=off, can use explicit CoT in prompt
            system = f"{data['system_expert']} You also have analysis from an ML model."
            user = (f"{profile}{enrich}\n\n"
                   f"Analyze the transaction patterns, then classify. "
                   f"Output JSON: {{\"reasoning\": \"your analysis\", "
                   f"\"label\": \"{data['pos_label']}\" or \"{data['neg_label']}\", "
                   f"\"confidence\": 0-1}}")

        return [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    ckpt = OUT / f"{dataset_name}_glm47_proper_cot_{mode}_ckpt.npz"
    if ckpt.exists():
        preds = list(np.load(ckpt)["preds"])
        start = len(preds)
        print(f"  Resuming from {start}/{len(data['cids_test'])}", flush=True)
    else:
        preds = []; start = 0

    BATCH = 5
    t0 = time.time()
    for bs in range(start, len(data["cids_test"]), BATCH):
        be = min(bs + BATCH, len(data["cids_test"]))
        with ThreadPoolExecutor(max_workers=BATCH) as ex:
            futs = {ex.submit(call_glm_proper, build_msg(i), api_key, reasoning_on,
                              data["pos_label"], data["neg_label"]): i for i in range(bs, be)}
            br = {futs[f]: f.result() for f in as_completed(futs)}
        for i in range(bs, be):
            preds.append(br[i])

        if len(preds) % 50 < BATCH:
            np.savez(ckpt, preds=np.array(preds))
            auc = roc_auc_score(data["y_test"][:len(preds)], preds)
            rate = (len(preds)-start) / max(time.time()-t0, 0.1)
            unique = len(set(round(p, 3) for p in preds))
            print(f"    {len(preds)}/{len(data['cids_test'])} "
                  f"({rate:.1f}/s, AUC={auc:.4f}, {unique} unique probs)", flush=True)

        if not budget.check():
            np.savez(ckpt, preds=np.array(preds)); return 0.5

    auc = roc_auc_score(data["y_test"], preds)
    print(f"  GLM-4.7 reasoning={mode} + kNN (proper): AUC={auc:.4f}", flush=True)

    with open(cache, "w") as f:
        json.dump({"auc": auc, "n_test": len(preds), "reasoning": mode, "method": "proper_cot_json"}, f, indent=2)
    np.savez(OUT / f"{dataset_name}_glm47_proper_cot_{mode}_preds.npz",
             preds=np.array(preds), cids=data["cids_test"], y_test=data["y_test"])
    if ckpt.exists(): ckpt.unlink()
    return auc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gender")
    ap.add_argument("--budget", type=float, default=1.5)
    args = ap.parse_args()

    budget.max_budget = args.budget
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key: exit("Set OPENROUTER_API_KEY")

    # Test with 10 clients first
    print("=== Sanity check on 10 clients ===", flush=True)
    data = load_dataset(args.dataset)
    for i in range(5):
        cid = data["cids_test"][i]
        k = data["knn_ctx"][cid]
        profile = data["serialize"](int(cid))
        system = f"{data['system_expert']} You also have analysis from an ML model."
        user = (f"{profile}\nSimilar clients: {k['pos']} {data['pos_label']}, "
               f"{k['neg']} {data['neg_label']}.\n\n"
               f"Classify this client. Output JSON: "
               f'{{"label": "{data["pos_label"]}" or "{data["neg_label"]}", "confidence": 0-1}}')
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prob = call_glm_proper(messages, api_key, True, data["pos_label"], data["neg_label"])
        true = data["pos_label"] if data["y_test"][i] == 1 else data["neg_label"]
        print(f"  Client {i}: kNN={k['pos']}/10, true={true}, prob={prob:.3f}")

    print("\n=== Full runs ===", flush=True)
    auc_off = run_glm_kncot(args.dataset, False, api_key)
    auc_on = run_glm_kncot(args.dataset, True, api_key)

    print(f"\n{'='*50}")
    print(f"GLM-4.7 PROPER CoT RESULTS (Gender, kNN CoT):")
    print(f"  reasoning=off: AUC={auc_off:.4f}")
    print(f"  reasoning=on:  AUC={auc_on:.4f}")
    print(f"  Δ = {(auc_on-auc_off)*100:+.2f}pp")
