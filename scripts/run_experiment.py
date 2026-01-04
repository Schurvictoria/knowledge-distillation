#!/usr/bin/env python3
"""Full experiment pipeline: data -> text -> LLM -> train -> eval."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distil.data import prepare_dataset
from distil.evaluate import compare_variants, print_results, save_results
from distil.pseudo_labels import get_pseudo_labels
from distil.text_convert import batch_to_text, build_mcc_map
from distil.train import run_all_variants


def extract_arrays(
    ids: np.ndarray, pseudo_dict: dict,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract label/prob/explanation arrays aligned with client IDs."""
    labels = np.array(
        [pseudo_dict.get(cid, {}).get("label", 0) for cid in ids],
        dtype=np.float32,
    )
    probs = np.array(
        [pseudo_dict.get(cid, {}).get("probability", 0.5) for cid in ids],
        dtype=np.float32,
    )
    explanations = [
        pseudo_dict.get(cid, {}).get("explanation", "") for cid in ids
    ]
    return labels, probs, explanations


def main():
    parser = argparse.ArgumentParser(description="Run distillation experiment")
    parser.add_argument(
        "--dataset", default="gender", choices=["gender", "rosbank"],
    )
    parser.add_argument(
        "--model-type", default="both",
        choices=["xgboost", "catboost", "both"],
    )
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument(
        "--api-base", default=None,
        help="Base URL for OpenAI-compatible API (e.g. http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--use-mock", action="store_true",
        help="Use mock LLM (no API key needed)",
    )
    parser.add_argument(
        "--max-clients", type=int, default=None,
        help="Limit number of clients (for testing)",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Step 1: Load and prepare dataset
    print(f"\n[1/5] Loading {args.dataset} dataset...")
    data = prepare_dataset(args.dataset, seed=args.seed)
    print(f"  Train: {len(data['X_train'])} clients, "
          f"Test: {len(data['X_test'])} clients")
    print(f"  Features: {len(data['feature_names'])}")
    print(f"  Label distribution (train): {np.bincount(data['y_train'])}")

    # Step 2: Convert transactions to text
    print("\n[2/5] Converting transactions to text...")
    mcc_map = build_mcc_map(data.get("mcc_codes"))

    train_ids = data["customer_ids_train"]
    test_ids = data["customer_ids_test"]
    if args.max_clients:
        train_ids = train_ids[:args.max_clients]
        test_ids = test_ids[:max(1, args.max_clients // 5)]

    texts_train = batch_to_text(train_ids, data["transactions"], mcc_map)
    texts_test = batch_to_text(test_ids, data["transactions"], mcc_map)
    print(f"  Generated text for {len(texts_train)} train "
          f"+ {len(texts_test)} test clients")

    sample_id = next(iter(texts_train))
    print(f"\n  Sample text (client {sample_id}):")
    print(f"  {texts_train[sample_id][:200]}...")

    # Step 3: Get pseudo-labels from LLM
    print("\n[3/5] Getting LLM pseudo-labels...")
    task = "gender" if args.dataset == "gender" else "churn"
    target_col = "gender" if args.dataset == "gender" else "target"

    labels_dict = dict(zip(
        data["labels"]["customer_id"],
        data["labels"][target_col],
    ))

    all_texts = {**texts_train, **texts_test}
    pseudo = get_pseudo_labels(
        all_texts,
        task=task,
        model=args.llm_model,
        api_base=args.api_base,
        true_labels=labels_dict,
        use_mock=args.use_mock,
    )

    pl_train, pp_train, expl_train = extract_arrays(
        data["customer_ids_train"], pseudo,
    )
    pl_test, pp_test, expl_test = extract_arrays(
        data["customer_ids_test"], pseudo,
    )

    llm_acc = accuracy_score(data["y_test"], pl_test)
    llm_auc = roc_auc_score(data["y_test"], pp_test)
    print(f"  LLM standalone accuracy: {llm_acc:.4f}")
    print(f"  LLM standalone ROC-AUC: {llm_auc:.4f}")

    # Step 4: Train all variants
    all_results = []
    model_types = (
        ["xgboost", "catboost"]
        if args.model_type == "both"
        else [args.model_type]
    )

    for mt in model_types:
        print(f"\n[4/5] Training {mt} variants...")
        predictions = run_all_variants(
            data["X_train"], data["y_train"],
            data["X_test"], data["y_test"],
            pl_train, pp_train, pl_test, pp_test,
            expl_train, expl_test,
            model_type=mt, seed=args.seed,
        )
        predictions["e_llm_raw"] = pp_test

        # Step 5: Evaluate
        print(f"\n[5/5] Evaluating {mt}...")
        df = compare_variants(data["y_test"], predictions, args.dataset, mt)
        all_results.append(df)

    final = pd.concat(all_results, ignore_index=True)
    print_results(final)

    out_path = output_dir / f"{args.dataset}_results.csv"
    save_results(final, str(out_path))

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
