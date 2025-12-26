"""Call LLM to get pseudo-labels, probabilities, and explanations."""

import json
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm


CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "llm_cache"

GENDER_PROMPT = """\
Predict gender from bank transactions. Base rate: 55%F, 45%M.
FEMALE: cosmetics, pharmacy, flowers, jewelry, children's goods.
MALE: auto/fuel/gas, electronics, hardware, sports, computers.
NEUTRAL (ignore): ATM, groceries, telecom, transfers, restaurants.
Examples: fuel+auto=male(0.78). Only ATM+groceries=base rate(0.45). Pharmacy+cosmetics=female(0.15).

{client_text}

{{"step1":"categories","step2":"signals","step3":"conclusion","probability":float,"label":0or1}}"""

CHURN_PROMPT = """\
Analyze bank transaction patterns to predict if a client will churn (leave the bank).

Client transaction summary:
{client_text}

Consider: declining transaction frequency, reduced spending, fewer unique merchants = churn risk.
Active usage, diverse spending, regular patterns = likely to stay.

Output JSON:
{{"reasoning": "analysis", "probability": P(churn) 0-1, "label": 0 or 1}}
Where: 0=stays, 1=churns."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "step1_categories": {"type": "string"},
        "step2_signals": {"type": "string"},
        "step3_conclusion": {"type": "string"},
        "probability": {"type": "number"},
        "label": {"type": "integer", "enum": [0, 1]},
    },
    "required": [
        "step1_categories", "step2_signals", "step3_conclusion",
        "probability", "label",
    ],
}


def _cache_key(client_text: str, model: str, task: str) -> str:
    return hashlib.md5(f"{model}:{task}:{client_text}".encode()).hexdigest()


def _load_cache(cache_dir: Path = CACHE_DIR) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "responses.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict, cache_dir: Path = CACHE_DIR):
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "responses.json", "w") as f:
        json.dump(cache, f)


def _parse_response(text: str) -> dict:
    """Parse LLM response into label, probability, explanation."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end])
            prob = float(obj.get("probability", 0.5))
            prob = max(0.05, min(0.95, prob))
            label = 1 if prob >= 0.5 else 0
            reasoning_parts = []
            for key in [
                "step1_categories", "step2_signals", "step3_conclusion",
                "reasoning", "explanation",
            ]:
                val = obj.get(key)
                if val:
                    reasoning_parts.append(str(val))
            explanation = " | ".join(reasoning_parts) if reasoning_parts else ""
            return {
                "label": label,
                "probability": prob,
                "explanation": explanation,
            }
        except (json.JSONDecodeError, ValueError):
            pass
    return {"label": 0, "probability": 0.45, "explanation": text}


def _call_single_openai(
    cid: int,
    text: str,
    prompt: str,
    client,
    model_name: str,
    use_guided: bool,
) -> tuple[int, dict]:
    """Call vLLM via openai client. Supports guided JSON output."""
    extra = {}
    if use_guided:
        extra["extra_body"] = {
            "guided_json": json.dumps(RESPONSE_SCHEMA),
        }

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
                **extra,
            )
            raw = response.choices[0].message.content
            return cid, _parse_response(raw)
        except Exception as e:
            if attempt < 2:
                time.sleep((attempt + 1) * 2)
            else:
                return cid, {
                    "label": 0,
                    "probability": 0.45,
                    "explanation": f"API error: {e}",
                }
    return cid, {"label": 0, "probability": 0.45, "explanation": "unreachable"}


def call_llm(
    client_texts: dict[int, str],
    task: str = "gender",
    model: str = "gpt-4o-mini",
    api_base: str | None = None,
    cache_dir: Path = CACHE_DIR,
    max_workers: int = 16,
    batch_save_every: int = 200,
) -> dict[int, dict]:
    """Call LLM for each client text and return predictions.

    Uses ThreadPoolExecutor for concurrent requests.
    For local vLLM, uses openai client with guided_json.
    For remote APIs, uses litellm.
    """
    prompt_template = GENDER_PROMPT if task == "gender" else CHURN_PROMPT
    cache = _load_cache(cache_dir)
    results = {}

    uncached = {}
    for cid, text in client_texts.items():
        key = _cache_key(text, model, task)
        if key in cache:
            results[cid] = cache[key]
        else:
            uncached[cid] = text

    if not uncached:
        return results

    print(f"  {len(results)} cached, {len(uncached)} to process "
          f"({max_workers} workers)")

    if api_base:
        # Use openai client directly for vLLM (better guided generation support)
        from openai import OpenAI
        client = OpenAI(base_url=api_base, api_key="not-needed")
        # Extract model name (remove openai/ prefix if present)
        model_name = model.replace("openai/", "")

        prompts = {
            cid: prompt_template.format(client_text=text)
            for cid, text in uncached.items()
        }

        new_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _call_single_openai,
                    cid, text, prompts[cid], client, model_name, True,
                ): cid
                for cid, text in uncached.items()
            }
            pbar = tqdm(
                as_completed(futures), total=len(futures),
                desc="LLM predictions",
            )
            for future in pbar:
                cid, parsed = future.result()
                results[cid] = parsed
                key = _cache_key(uncached[cid], model, task)
                cache[key] = parsed
                new_count += 1
                if new_count % batch_save_every == 0:
                    _save_cache(cache, cache_dir)
    else:
        # Use litellm for remote APIs
        import litellm

        for cid, text in tqdm(uncached.items(), desc="LLM predictions"):
            prompt = prompt_template.format(client_text=text)
            for attempt in range(3):
                try:
                    response = litellm.completion(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=500,
                    )
                    raw = response.choices[0].message.content
                    parsed = _parse_response(raw)
                    results[cid] = parsed
                    key = _cache_key(text, model, task)
                    cache[key] = parsed
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep((attempt + 1) * 2)
                    else:
                        results[cid] = {
                            "label": 0, "probability": 0.45,
                            "explanation": f"error: {e}",
                        }

    _save_cache(cache, cache_dir)
    return results


def mock_llm(
    client_texts: dict[int, str],
    true_labels: dict[int, int] | None = None,
    noise: float = 0.15,
    seed: int = 42,
) -> dict[int, dict]:
    """Generate mock LLM predictions for testing without API access."""
    rng = np.random.RandomState(seed)
    results = {}

    for cid, text in client_texts.items():
        if true_labels and cid in true_labels:
            true_label = true_labels[cid]
            if rng.random() < noise:
                pred_label = 1 - true_label
            else:
                pred_label = true_label
            if pred_label == 1:
                prob = min(1.0, 0.6 + rng.random() * 0.35)
            else:
                prob = max(0.0, rng.random() * 0.4)
        else:
            pred_label = rng.randint(0, 2)
            prob = rng.random()

        n_tx = (
            text.split("has ")[1].split(" transactions")[0]
            if "has " in text else "unknown"
        )
        explanation = (
            f"Based on {n_tx} transactions, pattern suggests "
            f"{'male' if pred_label == 1 else 'female'}."
        )
        results[cid] = {
            "label": pred_label,
            "probability": float(prob),
            "explanation": explanation,
        }

    return results


def get_pseudo_labels(
    client_texts: dict[int, str],
    task: str = "gender",
    model: str = "gpt-4o-mini",
    api_base: str | None = None,
    true_labels: dict[int, int] | None = None,
    use_mock: bool = False,
) -> dict[int, dict]:
    """Get pseudo-labels from LLM or mock.

    Auto-detects if API key or local server is available.
    """
    if use_mock:
        print("Using mock LLM predictions")
        return mock_llm(client_texts, true_labels)

    if api_base:
        print(f"Using local LLM at {api_base}")
        return call_llm(
            client_texts, task=task, model=model, api_base=api_base,
        )

    has_key = any(
        os.environ.get(k)
        for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
    )

    if not has_key:
        print("No LLM API key found, using mock predictions")
        return mock_llm(client_texts, true_labels)

    return call_llm(client_texts, task=task, model=model)
