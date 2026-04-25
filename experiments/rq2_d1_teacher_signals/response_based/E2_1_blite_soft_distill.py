#!/usr/bin/env python3
"""
Track B-lite: Soft-label distillation into XGBoost/LightGBM.

Compare three variants:
  1. No distillation: train on hard labels y_true
  2. Hard labels from LLM: replace y_true with LLM hard predictions
  3. Soft labels from LLM: custom objective mixing BCE(y_true) + KL(p_model, p_llm)

Uses pre-computed:
  - CoLES embeddings (Phase 1)
  - LLM predictions from Phase 2 (zero-shot, few-shot, shap-enriched)
  - kNN CoT predictions from RAMD

All 3 datasets. CPU only, ~5 min total.
"""

import json, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MaxAbsScaler
from lightgbm import LGBMClassifier
import lightgbm as lgb

OUTPUT_DIR = Path("results/blite_soft_distill")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LGBM_BIN = dict(n_estimators=500, learning_rate=0.02, max_depth=6, subsample=0.5,
                colsample_bytree=0.75, reg_alpha=1, reg_lambda=1, min_child_samples=50, verbosity=-1)
LGBM_MULTI = dict(n_estimators=1000, learning_rate=0.02, objective="multiclass", num_class=4,
                  max_depth=12, num_leaves=50, subsample=0.75, colsample_bytree=0.75,
                  reg_alpha=1, reg_lambda=1, min_child_samples=50, verbosity=-1)


def soft_label_lgbm_binary(X_train, y_train, llm_probs_train, X_test, y_test,
                            alpha=0.3, n_estimators=500):
    """Train LGBM with soft targets: (1-alpha)*y_true + alpha*p_llm."""
    # Create soft targets
    y_soft = (1 - alpha) * y_train.astype(np.float64) + alpha * llm_probs_train

    # LightGBM with regression objective on soft targets, then eval as classification
    params = {
        "objective": "regression", "metric": "rmse",
        "learning_rate": 0.02, "max_depth": 6, "num_leaves": 31,
        "subsample": 0.5, "colsample_bytree": 0.75,
        "reg_alpha": 1, "reg_lambda": 1, "min_child_samples": 50,
        "n_estimators": n_estimators, "verbose": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_soft)
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, 1)
    return roc_auc_score(y_test, preds)


def run_dataset(dataset_name):
    print(f"\n{'='*60}")
    print(f"B-LITE: {dataset_name.upper()}")
    print(f"{'='*60}")

    # Load CoLES embeddings
    coles_train = np.load(f"embeddings/{dataset_name}/emb_train_seed42.npy")
    coles_test = np.load(f"embeddings/{dataset_name}/emb_test_seed42.npy")
    y_train = np.load(f"embeddings/{dataset_name}/y_train_seed42.npy")
    y_test = np.load(f"embeddings/{dataset_name}/y_test_seed42.npy")

    sc = MaxAbsScaler()
    X_train = sc.fit_transform(coles_train)
    X_test = sc.transform(coles_test)

    is_binary = dataset_name in ["gender", "rosbank"]
    results = {}

    # 1. No distillation (baseline)
    if is_binary:
        lgbm = LGBMClassifier(**LGBM_BIN, random_state=42)
        lgbm.fit(X_train, y_train)
        p = lgbm.predict_proba(X_test)[:, 1]
        results["no_distill"] = roc_auc_score(y_test, p)
    else:
        lgbm = LGBMClassifier(**LGBM_MULTI, random_state=42)
        lgbm.fit(X_train, y_train)
        results["no_distill"] = accuracy_score(y_test, lgbm.predict(X_test))

    print(f"  No distillation:        {results['no_distill']:.4f}")

    # Load LLM predictions
    llm_dir = Path(f"results/{dataset_name}_llm")
    strategies = ["zero_shot", "few_shot", "shap_enriched"]

    # Also try kNN CoT if available
    cot_dir = Path("results/gender_rosbank_cot") if dataset_name != "age" else Path("results/age_structured_cot")

    for strat in strategies:
        pred_file = llm_dir / f"{dataset_name}_{strat}_predictions.csv"
        if not pred_file.exists():
            continue

        df = pd.read_csv(pred_file)

        if is_binary:
            if "pred_prob" not in df.columns:
                continue

            # Align predictions with train/test split
            cids_train = np.load(f"embeddings/{dataset_name}/cids_train_seed42.npy")
            cids_test_emb = np.load(f"embeddings/{dataset_name}/cids_test_seed42.npy")

            cid_to_prob = dict(zip(df["customer_id"], df["pred_prob"]))
            llm_train = np.array([cid_to_prob.get(c, 0.5) for c in cids_train])
            llm_test = np.array([cid_to_prob.get(c, 0.5) for c in cids_test_emb])

            # 2. Hard labels from LLM
            y_hard_llm = (llm_train > 0.5).astype(int)
            lgbm = LGBMClassifier(**LGBM_BIN, random_state=42)
            lgbm.fit(X_train, y_hard_llm)
            p = lgbm.predict_proba(X_test)[:, 1]
            results[f"hard_llm_{strat}"] = roc_auc_score(y_test, p)

            # 3. Soft labels (various alpha)
            for alpha in [0.1, 0.2, 0.3, 0.5]:
                auc = soft_label_lgbm_binary(X_train, y_train, llm_train, X_test, y_test, alpha=alpha)
                results[f"soft_{strat}_α{alpha}"] = auc

            print(f"  {strat}:")
            print(f"    hard_llm:   {results[f'hard_llm_{strat}']:.4f}")
            best_alpha = max((a for a in [0.1, 0.2, 0.3, 0.5]),
                           key=lambda a: results.get(f"soft_{strat}_α{a}", 0))
            print(f"    soft best:  {results[f'soft_{strat}_α{best_alpha}']:.4f} (α={best_alpha})")

    # kNN CoT predictions (if available)
    if dataset_name in ["gender", "rosbank"]:
        cot_file = cot_dir / f"{dataset_name}_cot_results.json"
        knn_pred_file = Path(f"results/gender_rosbank_cot") if dataset_name != "age" else None

        # Use RAMD round 0 predictions if available
        ramd_file = Path(f"results/{dataset_name}_ramd/ramd_results.json")
        if not ramd_file.exists() and dataset_name == "gender":
            ramd_file = Path("results/gender_ramd/ramd_results.json")

    # Also: CoLES + LLM soft probs as extra features (stacking-style but with soft targets)
    for strat in strategies:
        pred_file = llm_dir / f"{dataset_name}_{strat}_predictions.csv"
        if not pred_file.exists() or not is_binary:
            continue

        df = pd.read_csv(pred_file)
        cids_train_arr = np.load(f"embeddings/{dataset_name}/cids_train_seed42.npy")
        cids_test_arr = np.load(f"embeddings/{dataset_name}/cids_test_seed42.npy")
        cid_to_prob = dict(zip(df["customer_id"], df["pred_prob"]))
        llm_tr = np.array([cid_to_prob.get(c, 0.5) for c in cids_train_arr]).reshape(-1, 1)
        llm_te = np.array([cid_to_prob.get(c, 0.5) for c in cids_test_arr]).reshape(-1, 1)

        # Concat CoLES + LLM prob as feature + soft label training
        X_tr_aug = np.hstack([X_train, llm_tr])
        X_te_aug = np.hstack([X_test, llm_te])
        for alpha in [0.2, 0.3]:
            y_soft = (1 - alpha) * y_train.astype(np.float64) + alpha * llm_tr.ravel()
            model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.02, max_depth=6,
                                       subsample=0.5, colsample_bytree=0.75, reg_alpha=1,
                                       reg_lambda=1, min_child_samples=50, verbose=-1)
            model.fit(X_tr_aug, y_soft)
            p = np.clip(model.predict(X_te_aug), 0, 1)
            results[f"concat+soft_{strat}_α{alpha}"] = roc_auc_score(y_test, p)

        best_cs = max((a for a in [0.2, 0.3]),
                     key=lambda a: results.get(f"concat+soft_{strat}_α{a}", 0))
        print(f"    concat+soft: {results[f'concat+soft_{strat}_α{best_cs}']:.4f} (α={best_cs})")

    # Summary
    print(f"\n  {dataset_name.upper()} SUMMARY:")
    base = results["no_distill"]
    for n, v in sorted(results.items(), key=lambda x: -x[1]):
        d = v - base
        print(f"    {n:<35} {v:.4f} ({'+' if d>=0 else ''}{d:.4f})")

    return results


# ---- Run all datasets ----
all_results = {}
for ds in ["gender", "rosbank"]:
    all_results[ds] = run_dataset(ds)

# Save
for ds, r in all_results.items():
    with open(OUTPUT_DIR / f"{ds}_blite_results.json", "w") as f:
        json.dump(r, f, indent=2)

print("\n" + "=" * 60)
print("B-LITE SUMMARY (all datasets)")
print("=" * 60)
for ds, r in all_results.items():
    base = r["no_distill"]
    best_name = max(r, key=r.get)
    best_val = r[best_name]
    print(f"  {ds}: baseline={base:.4f}, best={best_val:.4f} ({best_name}, {'+' if best_val-base>=0 else ''}{best_val-base:.4f})")
