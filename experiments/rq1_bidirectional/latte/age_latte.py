import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

warnings.filterwarnings("ignore")

from distil.data import load_age_dataset
from distil.models import (
    ColesConfig,
    LatteFinetuneConfig,
    build_coles_encoder,
    standardize_text_embeddings,
    train_coles_baseline,
    train_latte_finetune,
)
from distil.reproducibility import (
    require_coles_embeddings,
    require_llm4es_embeddings,
    seed_everything,
)
from distil.results import save_experiment_result

SEED = 42
DATASET_NAME = "age"
EXPERIMENT_BASE_ID = "latte_age"
TASK_TYPE = "multiclass"

OUTPUT_DIRECTORY = Path("results/age_true_latte")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
COLES_CHECKPOINT_PATH = OUTPUT_DIRECTORY / "coles_baseline.pt"

CANONICAL_DISTILLATION_WEIGHT = 0.1

def _load_aligned_llm4es_embeddings(train_customer_ids, test_customer_ids, device):
    require_llm4es_embeddings(DATASET_NAME)
    require_coles_embeddings(DATASET_NAME, seed=SEED)

    llm_embeddings_archive = np.load(f"results/{DATASET_NAME}_llm4es/llm4es_embeddings.npz")
    all_llm_embeddings = llm_embeddings_archive["embeddings"].astype(np.float32)

    coles_train_cids = np.load(f"embeddings/{DATASET_NAME}/cids_train_seed{SEED}.npy")
    coles_test_cids = np.load(f"embeddings/{DATASET_NAME}/cids_test_seed{SEED}.npy")
    all_coles_cids = np.concatenate([coles_train_cids, coles_test_cids])

    assert len(all_llm_embeddings) == len(all_coles_cids), (
        f"LLM emb count ({len(all_llm_embeddings)}) != cids count ({len(all_coles_cids)})"
    )
    cid_to_llm_embedding = {
        customer_id: all_llm_embeddings[index]
        for index, customer_id in enumerate(all_coles_cids)
    }

    missing_train = [cid for cid in train_customer_ids if cid not in cid_to_llm_embedding]
    missing_test = [cid for cid in test_customer_ids if cid not in cid_to_llm_embedding]
    assert not missing_train, f"Missing LLM embedding for {len(missing_train)} train cids"
    assert not missing_test, f"Missing LLM embedding for {len(missing_test)} test cids"

    train_llm_embeddings = np.array([cid_to_llm_embedding[cid] for cid in train_customer_ids])
    test_llm_embeddings = np.array([cid_to_llm_embedding[cid] for cid in test_customer_ids])
    return standardize_text_embeddings(train_llm_embeddings, test_llm_embeddings, device)

def _compute_baseline_lgbm_accuracy(encoder, train_records, test_records, train_targets, test_targets, device):
    from distil.models.latte import extract_sequence_embeddings

    train_embeddings = extract_sequence_embeddings(encoder, train_records, device)
    test_embeddings = extract_sequence_embeddings(encoder, test_records, device)

    scaler = MaxAbsScaler()
    scaled_train = scaler.fit_transform(train_embeddings)
    scaled_test = scaler.transform(test_embeddings)

    from lightgbm import LGBMClassifier
    num_classes = int(np.unique(np.concatenate([train_targets, test_targets])).size)
    classifier = LGBMClassifier(
        n_estimators=1000, learning_rate=0.02, objective="multiclass", num_class=num_classes,
        max_depth=12, num_leaves=50, subsample=0.75, colsample_bytree=0.75,
        reg_alpha=1, reg_lambda=1, min_child_samples=50, verbosity=-1, random_state=SEED,
    ).fit(scaled_train, train_targets)
    return float(accuracy_score(test_targets, classifier.predict(scaled_test)))

def main() -> None:
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coles_config = ColesConfig.for_dataset(DATASET_NAME)

    full_dataset = load_age_dataset(seed=SEED)
    train_targets_full = full_dataset.train_targets
    test_targets = full_dataset.test_targets
    test_customer_ids = [record["customer_id"] for record in full_dataset.test_records]

    train_indices_relative, val_indices_relative = train_test_split(
        np.arange(len(full_dataset.train_records)),
        test_size=0.1,
        random_state=SEED,
        stratify=train_targets_full,
    )
    train_records = [full_dataset.train_records[index] for index in train_indices_relative]
    val_records = [full_dataset.train_records[index] for index in val_indices_relative]
    train_targets = train_targets_full[train_indices_relative]
    val_targets = train_targets_full[val_indices_relative]
    train_customer_ids = [record["customer_id"] for record in train_records]

    text_embeddings_train, _ = _load_aligned_llm4es_embeddings(
        train_customer_ids, test_customer_ids, device,
    )
    print(f"  LLM4ES aligned: train={text_embeddings_train.shape}")

    if COLES_CHECKPOINT_PATH.exists():
        print("\nPhase 1: Loading CoLES checkpoint")
        sequence_encoder = build_coles_encoder(full_dataset.feature_dimensions, coles_config)
        sequence_encoder.load_state_dict(torch.load(COLES_CHECKPOINT_PATH, map_location="cpu"))
        sequence_encoder = sequence_encoder.to(device)
    else:
        print("\nPhase 1: Train CoLES baseline")
        sequence_encoder = build_coles_encoder(full_dataset.feature_dimensions, coles_config)
        coles_module, trainer = train_coles_baseline(sequence_encoder, train_records, coles_config)
        torch.save(coles_module._seq_encoder.state_dict(), COLES_CHECKPOINT_PATH)
        print(f"  Saved baseline checkpoint to {COLES_CHECKPOINT_PATH}")
        sequence_encoder = coles_module._seq_encoder.to(device)
        del coles_module, trainer
        torch.cuda.empty_cache()
        gc.collect()

    baseline_test_score = _compute_baseline_lgbm_accuracy(
        sequence_encoder, train_records, full_dataset.test_records,
        train_targets, test_targets, device,
    )
    print(f"  Baseline CoLES LGBM accuracy: {baseline_test_score:.4f}")

    print(f"\nPhase 2: LATTE finetune (alpha={CANONICAL_DISTILLATION_WEIGHT})")

    finetune_config = LatteFinetuneConfig(
        distillation_weight=CANONICAL_DISTILLATION_WEIGHT,
        learning_rate=3e-4,
        weight_decay=1e-4,
        num_epochs=20,
        batch_size=128,
        cosine_scheduler_t_max=20,
    )
    finetuned_checkpoint_path = OUTPUT_DIRECTORY / f"coles_finetuned_alpha{CANONICAL_DISTILLATION_WEIGHT}.pt"

    finetune_results = train_latte_finetune(
        sequence_encoder=sequence_encoder,
        train_records=train_records,
        val_records=val_records,
        test_records=full_dataset.test_records,
        train_targets=train_targets,
        val_targets=val_targets,
        test_targets=test_targets,
        text_embeddings_train=text_embeddings_train,
        coles_baseline_checkpoint_path=COLES_CHECKPOINT_PATH,
        finetuned_checkpoint_path=finetuned_checkpoint_path,
        sequence_embedding_dimension=coles_config.hidden_size,
        text_embedding_dimension=text_embeddings_train.shape[1],
        config=finetune_config,
        task_type=TASK_TYPE,
        device=device,
        baseline_test_score=baseline_test_score,
        seed=SEED,
    )

    best_test_accuracy = finetune_results["best_test_score"]
    print(f"\nBest: val={finetune_results['best_val_score']:.4f} "
          f"test={best_test_accuracy:.4f} at epoch {finetune_results['best_epoch']}")

    summary_payload = {
        "baseline_coles": baseline_test_score,
        f"finetune_alpha{CANONICAL_DISTILLATION_WEIGHT}": best_test_accuracy,
    }
    with (OUTPUT_DIRECTORY / "true_latte_results.json").open("w") as file_handle:
        json.dump(summary_payload, file_handle, indent=2)
    pd.DataFrame(
        [{"method": name, "accuracy": value} for name, value in summary_payload.items()]
    ).to_csv(OUTPUT_DIRECTORY / "true_latte_results.csv", index=False)

    save_experiment_result(
        experiment_id=EXPERIMENT_BASE_ID,
        rq="RQ1",
        method="LATTE",
        dataset=DATASET_NAME,
        task_type=TASK_TYPE,
        metrics={"accuracy": best_test_accuracy, "baseline_accuracy": baseline_test_score},
        config={
            "distillation_weight": CANONICAL_DISTILLATION_WEIGHT,
            "learning_rate": finetune_config.learning_rate,
            "weight_decay": finetune_config.weight_decay,
            "num_epochs": finetune_config.num_epochs,
            "batch_size": finetune_config.batch_size,
            "best_epoch": finetune_results["best_epoch"],
        },
        seed=SEED,
        artifacts={"checkpoint": str(finetuned_checkpoint_path)},
    )

if __name__ == "__main__":
    main()
