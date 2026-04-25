"""Shared runner for D1 size ladder LATTE experiments (E5.0/E5.1/E5.2-Instruct/E5.3).

Two-stage pipeline:
  Stage 0: extract LLM4ES embeddings via E5_x_extract_llm_embeddings.py if missing
  Stage 1: load canonical CoLES baseline checkpoint (from E1.1 run); train if missing
  Stage 2: LATTE fine-tune with teacher-specific text embeddings
  Stage 3: save checkpoint, summary, registry entry

All experiments are Gender-only, single seed=42.
"""
import gc
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

warnings.filterwarnings("ignore")

from distil.data import load_gender_dataset
from distil.models import (
    ColesConfig,
    LatteFinetuneConfig,
    build_coles_encoder,
    standardize_text_embeddings,
    train_coles_baseline,
    train_latte_finetune,
)
from distil.reproducibility import require_coles_embeddings, seed_everything
from distil.results import save_experiment_result


SEED = 42
DATASET_NAME = "gender"
TASK_TYPE = "binary"

CANONICAL_DISTILLATION_WEIGHT = 0.1
CANONICAL_COLES_CHECKPOINT_PATH = Path(f"results/{DATASET_NAME}_coles/coles_encoder_seed{SEED}.pt")

EXTRACTOR_SCRIPT = Path(__file__).parent / "E5_x_extract_llm_embeddings.py"


def _load_aligned_teacher_embeddings(
    teacher_short: str,
    train_customer_ids: list,
    test_customer_ids: list,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read teacher embeddings from teacher-specific path and align to CoLES cids."""
    require_coles_embeddings(DATASET_NAME, seed=SEED)

    teacher_path = Path(f"results/{DATASET_NAME}_{teacher_short}_llm_embeddings/llm_embeddings.npz")
    if not teacher_path.exists():
        raise FileNotFoundError(
            f"Teacher embeddings not found: {teacher_path}\n"
            f"Run extractor first (or stage 0 in run_latte_ladder)."
        )

    archive = np.load(teacher_path)
    all_embeddings = archive["embeddings"].astype(np.float32)
    cid_order = archive["cid_order"]

    cid_to_embedding = {int(cid): all_embeddings[i] for i, cid in enumerate(cid_order)}

    coles_train_cids = np.load(f"embeddings/{DATASET_NAME}/cids_train_seed{SEED}.npy")
    coles_test_cids = np.load(f"embeddings/{DATASET_NAME}/cids_test_seed{SEED}.npy")
    expected_total = len(coles_train_cids) + len(coles_test_cids)
    assert len(all_embeddings) == expected_total, (
        f"Teacher emb count ({len(all_embeddings)}) != cids count ({expected_total})"
    )

    missing_train = [cid for cid in train_customer_ids if int(cid) not in cid_to_embedding]
    missing_test = [cid for cid in test_customer_ids if int(cid) not in cid_to_embedding]
    assert not missing_train, f"Missing teacher emb for {len(missing_train)} train cids"
    assert not missing_test, f"Missing teacher emb for {len(missing_test)} test cids"

    train_teacher_embeddings = np.array([cid_to_embedding[int(cid)] for cid in train_customer_ids])
    test_teacher_embeddings = np.array([cid_to_embedding[int(cid)] for cid in test_customer_ids])

    return standardize_text_embeddings(train_teacher_embeddings, test_teacher_embeddings, device)


def _compute_baseline_lgbm_score(
    encoder,
    train_records,
    test_records,
    train_targets,
    test_targets,
    device: torch.device,
) -> float:
    from distil.models.latte import extract_sequence_embeddings

    train_embeddings = extract_sequence_embeddings(encoder, train_records, device)
    test_embeddings = extract_sequence_embeddings(encoder, test_records, device)

    scaler = MaxAbsScaler()
    scaled_train = scaler.fit_transform(train_embeddings)
    scaled_test = scaler.transform(test_embeddings)

    from lightgbm import LGBMClassifier
    classifier = LGBMClassifier(
        n_estimators=500, learning_rate=0.02, max_depth=6, subsample=0.5,
        colsample_bytree=0.75, reg_alpha=1, reg_lambda=1, min_child_samples=50,
        verbosity=-1, random_state=SEED,
    ).fit(scaled_train, train_targets)
    positive_class_probabilities = classifier.predict_proba(scaled_test)[:, 1]
    return float(roc_auc_score(test_targets, positive_class_probabilities))


def _ensure_teacher_embeddings(model_id: str, teacher_short: str) -> None:
    teacher_path = Path(f"results/{DATASET_NAME}_{teacher_short}_llm_embeddings/llm_embeddings.npz")
    if teacher_path.exists():
        print(f"  [Stage 0] Teacher embeddings cached: {teacher_path}", flush=True)
        return

    print(f"  [Stage 0] Extracting teacher embeddings: {model_id} → {teacher_short}", flush=True)
    completed_process = subprocess.run(
        [
            sys.executable, str(EXTRACTOR_SCRIPT),
            "--model", model_id,
            "--teacher", teacher_short,
            "--datasets", DATASET_NAME,
        ],
        check=False,
    )
    if completed_process.returncode != 0:
        sys.exit(f"E5_x extractor failed with code {completed_process.returncode}")
    if not teacher_path.exists():
        sys.exit(f"Extractor finished but {teacher_path} not produced — check logs above.")


def run_latte_ladder(
    experiment_id: str,
    model_id: str,
    teacher_short: str,
    rq: str = "RQ3",
    method_label: str = "LATTE (Qwen2.5 ladder)",
) -> None:
    overall_start = time.time()
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_directory = Path(f"results/{DATASET_NAME}_latte_{teacher_short}")
    output_directory.mkdir(parents=True, exist_ok=True)
    finetuned_checkpoint_path = (
        output_directory / f"coles_finetuned_alpha{CANONICAL_DISTILLATION_WEIGHT}.pt"
    )

    print(f"=== {experiment_id} ===", flush=True)
    print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, seed={SEED}", flush=True)
    print(f"  teacher: {model_id}", flush=True)
    print(f"  output:  {output_directory}", flush=True)

    _ensure_teacher_embeddings(model_id, teacher_short)

    coles_config = ColesConfig.for_dataset(DATASET_NAME)
    full_dataset = load_gender_dataset(seed=SEED)

    test_customer_ids = [record["customer_id"] for record in full_dataset.test_records]
    train_targets_full = full_dataset.train_targets
    test_targets = full_dataset.test_targets

    train_indices_relative, val_indices_relative = train_test_split(
        np.arange(len(full_dataset.train_records)),
        test_size=0.1,
        random_state=SEED,
        stratify=train_targets_full,
    )
    train_records = [full_dataset.train_records[i] for i in train_indices_relative]
    val_records = [full_dataset.train_records[i] for i in val_indices_relative]
    train_targets = train_targets_full[train_indices_relative]
    val_targets = train_targets_full[val_indices_relative]
    train_customer_ids = [record["customer_id"] for record in train_records]

    print(f"  data: train={len(train_records)}, val={len(val_records)}, "
          f"test={len(full_dataset.test_records)}", flush=True)

    text_embeddings_train, _ = _load_aligned_teacher_embeddings(
        teacher_short=teacher_short,
        train_customer_ids=train_customer_ids,
        test_customer_ids=test_customer_ids,
        device=device,
    )
    print(f"  teacher emb aligned: train={text_embeddings_train.shape}", flush=True)

    print("\n[Stage 1] CoLES baseline", flush=True)
    if CANONICAL_COLES_CHECKPOINT_PATH.exists():
        print(f"  loading canonical checkpoint: {CANONICAL_COLES_CHECKPOINT_PATH}", flush=True)
        sequence_encoder = build_coles_encoder(full_dataset.feature_dimensions, coles_config)
        sequence_encoder.load_state_dict(
            torch.load(CANONICAL_COLES_CHECKPOINT_PATH, map_location="cpu")
        )
        sequence_encoder = sequence_encoder.to(device)
    else:
        print(f"  canonical checkpoint missing — training CoLES (~30 min)", flush=True)
        sequence_encoder = build_coles_encoder(full_dataset.feature_dimensions, coles_config)
        coles_module, trainer = train_coles_baseline(sequence_encoder, train_records, coles_config)
        CANONICAL_COLES_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(coles_module._seq_encoder.state_dict(), CANONICAL_COLES_CHECKPOINT_PATH)
        print(f"  saved canonical checkpoint to {CANONICAL_COLES_CHECKPOINT_PATH}", flush=True)
        sequence_encoder = coles_module._seq_encoder.to(device)
        del coles_module, trainer
        torch.cuda.empty_cache()
        gc.collect()

    baseline_test_score = _compute_baseline_lgbm_score(
        sequence_encoder, train_records, full_dataset.test_records,
        train_targets, test_targets, device,
    )
    print(f"  baseline CoLES LGBM AUC: {baseline_test_score:.4f}", flush=True)

    print(f"\n[Stage 2] LATTE finetune (alpha={CANONICAL_DISTILLATION_WEIGHT})", flush=True)
    finetune_config = LatteFinetuneConfig(
        distillation_weight=CANONICAL_DISTILLATION_WEIGHT,
        learning_rate=5e-4,
        weight_decay=1e-4,
        num_epochs=10,
        batch_size=128,
    )
    finetune_results = train_latte_finetune(
        sequence_encoder=sequence_encoder,
        train_records=train_records,
        val_records=val_records,
        test_records=full_dataset.test_records,
        train_targets=train_targets,
        val_targets=val_targets,
        test_targets=test_targets,
        text_embeddings_train=text_embeddings_train,
        coles_baseline_checkpoint_path=CANONICAL_COLES_CHECKPOINT_PATH,
        finetuned_checkpoint_path=finetuned_checkpoint_path,
        sequence_embedding_dimension=coles_config.hidden_size,
        text_embedding_dimension=text_embeddings_train.shape[1],
        config=finetune_config,
        task_type=TASK_TYPE,
        device=device,
        baseline_test_score=baseline_test_score,
        seed=SEED,
    )

    best_test_score = finetune_results["best_test_score"]
    elapsed_total = time.time() - overall_start
    print(f"\n  best: val={finetune_results['best_val_score']:.4f} "
          f"test={best_test_score:.4f} at epoch {finetune_results['best_epoch']}", flush=True)
    print(f"  Δ vs baseline: {best_test_score - baseline_test_score:+.4f}", flush=True)
    print(f"  total time: {elapsed_total:.0f}s", flush=True)

    summary_payload = {
        "experiment": experiment_id,
        "teacher_model": model_id,
        "teacher_short": teacher_short,
        "baseline_coles_auc": baseline_test_score,
        "latte_auc": best_test_score,
        "delta": best_test_score - baseline_test_score,
        "best_epoch": finetune_results["best_epoch"],
        "config": {
            "distillation_weight": CANONICAL_DISTILLATION_WEIGHT,
            "learning_rate": finetune_config.learning_rate,
            "weight_decay": finetune_config.weight_decay,
            "num_epochs": finetune_config.num_epochs,
            "batch_size": finetune_config.batch_size,
        },
        "seed": SEED,
        "runtime_seconds": elapsed_total,
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }
    with (output_directory / "ladder_results.json").open("w") as file_handle:
        json.dump(summary_payload, file_handle, indent=2)
    pd.DataFrame([{
        "experiment": experiment_id,
        "teacher": teacher_short,
        "baseline_auc": baseline_test_score,
        "latte_auc": best_test_score,
    }]).to_csv(output_directory / "ladder_results.csv", index=False)

    save_experiment_result(
        experiment_id=experiment_id,
        rq=rq,
        method=method_label,
        dataset=DATASET_NAME,
        task_type=TASK_TYPE,
        metrics={"roc_auc": best_test_score, "baseline_roc_auc": baseline_test_score},
        config={
            "teacher_model": model_id,
            "teacher_short": teacher_short,
            "distillation_weight": CANONICAL_DISTILLATION_WEIGHT,
            "learning_rate": finetune_config.learning_rate,
            "weight_decay": finetune_config.weight_decay,
            "num_epochs": finetune_config.num_epochs,
            "batch_size": finetune_config.batch_size,
            "best_epoch": finetune_results["best_epoch"],
        },
        seed=SEED,
        runtime_seconds=elapsed_total,
        artifacts={"checkpoint": str(finetuned_checkpoint_path)},
    )
