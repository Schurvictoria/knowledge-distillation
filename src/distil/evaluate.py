"""Evaluation metrics and comparison tables."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    average_precision_score,
)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute all metrics for a single model."""
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "avg_precision": average_precision_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba),
    }
    return metrics


def bootstrap_auc(
    y_true: np.ndarray, y_proba: np.ndarray, n_bootstrap: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for ROC-AUC.

    Returns (mean_auc, lower_95, upper_95).
    """
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))

    aucs = np.array(aucs)
    return float(aucs.mean()), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def compare_variants(
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
    dataset_name: str = "",
    model_type: str = "",
) -> pd.DataFrame:
    """Compare all variants and return a summary DataFrame."""
    rows = []
    for variant_name, y_proba in predictions.items():
        metrics = compute_metrics(y_test, y_proba)
        auc_mean, auc_lo, auc_hi = bootstrap_auc(y_test, y_proba)
        rows.append({
            "dataset": dataset_name,
            "model": model_type,
            "variant": variant_name,
            "roc_auc": metrics["roc_auc"],
            "auc_95ci_lo": auc_lo,
            "auc_95ci_hi": auc_hi,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "avg_precision": metrics["avg_precision"],
            "log_loss": metrics["log_loss"],
        })

    df = pd.DataFrame(rows)
    return df


def print_results(df: pd.DataFrame):
    """Pretty-print comparison table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS")
    print("=" * 80)

    for (dataset, model), group in df.groupby(["dataset", "model"]):
        print(f"\nDataset: {dataset} | Model: {model}")
        print("-" * 80)
        print(f"{'Variant':<25} {'ROC-AUC':>10} {'95% CI':>20} {'Accuracy':>10} {'F1':>10} {'LogLoss':>10}")
        print("-" * 80)
        for _, row in group.iterrows():
            ci = f"[{row['auc_95ci_lo']:.4f}, {row['auc_95ci_hi']:.4f}]"
            print(
                f"{row['variant']:<25} "
                f"{row['roc_auc']:>10.4f} "
                f"{ci:>20} "
                f"{row['accuracy']:>10.4f} "
                f"{row['f1']:>10.4f} "
                f"{row['log_loss']:>10.4f}"
            )
        print()

    # Reference baselines
    print("Reference baselines (from literature):")
    print("  CoLES (Gender): ~0.875 ROC-AUC")
    print("  LLM4ES (Gender): ~0.875 ROC-AUC")
    print("=" * 80)


def save_results(df: pd.DataFrame, path: str):
    """Save results to CSV."""
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")
