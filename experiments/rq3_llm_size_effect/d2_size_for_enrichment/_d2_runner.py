"""Shared runner for D2 size ladder kNN-CoT experiments (E6.0/E6.1/E6.1.5).

Runs ONE local Qwen model in 4-bit NF4 on Gender, two strategies:
  - zero_shot baseline (no enrichment) → "No enrichment" column
  - zero_shot + kNN context           → "+ kNN" column

Predictions are derived via logit-based scoring of pos/neg label tokens
(deterministic, faster than generation). Outputs:

  results/{output_dir_name}/
    summary.json              — AUCs for both strategies + delta
    predictions_no_enrich.npz — raw probabilities + cids + y_test
    predictions_knn.npz       — same with kNN
"""
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score

# Reproducibility (seed=42)
import random as _random
_SEED = 42
_random.seed(_SEED); np.random.seed(_SEED)
torch.manual_seed(_SEED); torch.cuda.manual_seed_all(_SEED)
import pytorch_lightning as _pl
_pl.seed_everything(_SEED, workers=True)
os.environ["PYTHONHASHSEED"] = str(_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from run_openrouter_experiments import load_dataset

from distil.results import save_experiment_result


def _get_token_ids(tokenizer, token_strings: list[str]) -> list[int]:
    ids = set()
    for token_string in token_strings:
        encoded = tokenizer.encode(token_string, add_special_tokens=False)
        if encoded:
            ids.add(encoded[0])
    return sorted(ids)


def _predict_pos_probability(
    model,
    tokenizer,
    messages: list,
    pos_token_ids: list[int],
    neg_token_ids: list[int],
) -> float:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        last_logits = model(**inputs).logits[0, -1, :]
    target_logits = last_logits[pos_token_ids + neg_token_ids].float()
    probabilities = torch.softmax(target_logits, dim=0)
    pos_mass = probabilities[: len(pos_token_ids)].sum().item()
    neg_mass = probabilities[len(pos_token_ids):].sum().item()
    total = pos_mass + neg_mass
    del inputs
    return pos_mass / total if total > 1e-8 else 0.5


_GENDER_POS_TOKENS = ["male", " male", "Male", " Male"]
_GENDER_NEG_TOKENS = ["female", " female", "Female", " Female"]


def _build_messages(profile_text: str, knn_context: str | None, system_expert: str) -> list:
    if knn_context is None:
        user_content = (
            f"{profile_text}\n\n"
            "Based on the transaction profile, predict the customer's gender. "
            "Answer with exactly one word: male or female."
        )
    else:
        user_content = (
            f"{profile_text}\n"
            f"{knn_context}\n\n"
            "Based on the transaction profile and the similar clients above, "
            "predict the customer's gender. Answer with exactly one word: male or female."
        )
    return [
        {"role": "system", "content": system_expert},
        {"role": "user", "content": user_content},
    ]


def _knn_enrichment_text(knn_entry: dict, pos_label: str, neg_label: str) -> str:
    return (
        f"Similar clients (top-10 nearest by transaction patterns): "
        f"{knn_entry['pos']} {pos_label}, {knn_entry['neg']} {neg_label} "
        f"(majority: {knn_entry['majority']})."
    )


def run_d2_size_ladder(
    experiment_id: str,
    model_id: str,
    output_dir_name: str,
    rq: str = "RQ3",
    method_label: str = "kNN-CoT enrichment (Qwen2.5 ladder)",
) -> None:
    overall_start = time.time()
    output_directory = Path(f"results/{output_dir_name}")
    output_directory.mkdir(parents=True, exist_ok=True)

    print(f"=== {experiment_id} ===", flush=True)
    print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, seed={_SEED}", flush=True)
    print(f"  model:  {model_id}", flush=True)
    print(f"  output: {output_directory}", flush=True)

    print("\n[Stage 1] Loading dataset (Gender) + kNN context...", flush=True)
    data = load_dataset("gender")
    cids_test = data["cids_test"]
    y_test = data["y_test"]
    pos_label = data["pos_label"]
    neg_label = data["neg_label"]
    system_expert = data["system_expert"]
    serialize = data["serialize"]
    knn_ctx = data["knn_ctx"]
    print(f"  test customers: {len(cids_test)}", flush=True)

    print(f"\n[Stage 2] Loading {model_id} (4-bit NF4)...", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    if torch.cuda.is_available():
        print(f"  loaded. GPU memory: {torch.cuda.memory_allocated() / 1024 ** 3:.1f}GB", flush=True)

    pos_token_ids = _get_token_ids(tokenizer, _GENDER_POS_TOKENS)
    neg_token_ids = _get_token_ids(tokenizer, _GENDER_NEG_TOKENS)
    print(f"  pos_ids={pos_token_ids}, neg_ids={neg_token_ids}", flush=True)

    profile_cache: dict[int, str] = {}
    for cid in cids_test:
        profile_cache[int(cid)] = serialize(int(cid))

    strategy_results: dict[str, dict] = {}

    for strategy_name in ["no_enrich", "knn"]:
        print(f"\n[Stage 3] Strategy: {strategy_name}", flush=True)
        checkpoint_path = output_directory / f"predictions_{strategy_name}_ckpt.npz"

        if checkpoint_path.exists():
            saved = np.load(checkpoint_path)
            predictions: list[float] = list(saved["predictions"])
            start_index = len(predictions)
            print(f"  resuming from {start_index}/{len(cids_test)}", flush=True)
        else:
            predictions = []
            start_index = 0

        strategy_start = time.time()

        for index in range(start_index, len(cids_test)):
            cid = int(cids_test[index])
            profile_text = profile_cache[cid]

            if strategy_name == "no_enrich":
                knn_context_text = None
            else:
                knn_context_text = _knn_enrichment_text(knn_ctx[cid], pos_label, neg_label)

            messages = _build_messages(profile_text, knn_context_text, system_expert)
            probability = _predict_pos_probability(
                model, tokenizer, messages, pos_token_ids, neg_token_ids,
            )
            predictions.append(probability)

            if (index + 1) % 200 == 0:
                np.savez(checkpoint_path, predictions=np.array(predictions))
                running_auc = roc_auc_score(y_test[: len(predictions)], predictions)
                rate = (len(predictions) - start_index) / max(time.time() - strategy_start, 0.1)
                eta_seconds = (len(cids_test) - index - 1) / max(rate, 0.1)
                print(
                    f"    {len(predictions)}/{len(cids_test)} "
                    f"({rate:.1f}/s, ETA {eta_seconds / 60:.0f}min, running AUC={running_auc:.4f})",
                    flush=True,
                )

        predictions_array = np.array(predictions)
        final_auc = float(roc_auc_score(y_test, predictions_array))
        binary_predictions = (predictions_array >= 0.5).astype(int)
        final_accuracy = float(accuracy_score(y_test, binary_predictions))
        strategy_elapsed = time.time() - strategy_start
        print(
            f"  {strategy_name}: AUC={final_auc:.4f}, acc={final_accuracy:.4f}, "
            f"time={strategy_elapsed:.0f}s",
            flush=True,
        )

        np.savez(
            output_directory / f"predictions_{strategy_name}.npz",
            predictions=predictions_array,
            customer_ids=cids_test,
            y_test=y_test,
        )
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        strategy_results[strategy_name] = {
            "auc": final_auc,
            "accuracy": final_accuracy,
            "time_seconds": strategy_elapsed,
        }

    delta = strategy_results["knn"]["auc"] - strategy_results["no_enrich"]["auc"]
    elapsed_total = time.time() - overall_start

    print("\n=== SUMMARY ===", flush=True)
    print(f"  no_enrich AUC: {strategy_results['no_enrich']['auc']:.4f}", flush=True)
    print(f"  + kNN     AUC: {strategy_results['knn']['auc']:.4f}", flush=True)
    print(f"  Δ            : {delta * 100:+.2f} pp", flush=True)
    print(f"  total time   : {elapsed_total:.0f}s", flush=True)

    summary = {
        "experiment": experiment_id,
        "model": model_id,
        "quantization": "4-bit NF4",
        "dataset": "gender",
        "n_test": int(len(cids_test)),
        "no_enrich_auc": strategy_results["no_enrich"]["auc"],
        "knn_auc": strategy_results["knn"]["auc"],
        "delta_pp": delta * 100,
        "strategies": strategy_results,
        "seed": _SEED,
        "runtime_seconds": elapsed_total,
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }
    with (output_directory / "summary.json").open("w") as file_handle:
        json.dump(summary, file_handle, indent=2)

    save_experiment_result(
        experiment_id=experiment_id,
        rq=rq,
        method=method_label,
        dataset="gender",
        task_type="binary",
        metrics={
            "roc_auc": strategy_results["knn"]["auc"],
            "no_enrich_auc": strategy_results["no_enrich"]["auc"],
            "delta_pp": delta * 100,
        },
        config={
            "model": model_id,
            "quantization": "4-bit NF4",
            "strategies": ["no_enrich", "knn"],
        },
        seed=_SEED,
        runtime_seconds=elapsed_total,
        artifacts={"output_directory": str(output_directory)},
    )

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
