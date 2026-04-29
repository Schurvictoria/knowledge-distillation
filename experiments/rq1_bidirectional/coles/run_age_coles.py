import json
import time
import warnings
from pathlib import Path

import gc
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from distil.data import load_age_dataset, save_coles_embeddings
from distil.downstream import evaluate_all_classifiers
from distil.models import (
    ColesConfig,
    build_coles_encoder,
    train_coles_baseline,
    extract_embeddings,
)
from distil.reproducibility import seed_everything
from distil.results import save_experiment_result

SEEDS = [42]
DATASET_NAME = "age"
EXPERIMENT_BASE_ID = "coles_age"

OUTPUT_DIRECTORY = Path("results/age_coles")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

def main() -> None:
    coles_config = ColesConfig.for_dataset(DATASET_NAME)
    print("AGE CoLES (paper config)")
    print(f"  {coles_config.rnn_type.upper()}-{coles_config.hidden_size}, "
          f"lr={coles_config.learning_rate}, epochs={coles_config.num_epochs}")
    print(f"  Metric: accuracy (4-class)")
    print(f"  Seeds: {SEEDS}")

    all_per_seed_metrics = []
    overall_start_time = time.time()

    for seed in SEEDS:
        per_seed_start_time = time.time()
        seed_everything(seed)

        dataset = load_age_dataset(seed=seed)
        print(f"  train={len(dataset.train_records)}, test={len(dataset.test_records)}, "
              f"features={dataset.feature_dimensions}")

        encoder = build_coles_encoder(dataset.feature_dimensions, coles_config)
        coles_module, trainer = train_coles_baseline(encoder, dataset.train_records, coles_config)

        encoder_checkpoint_path = OUTPUT_DIRECTORY / f"coles_encoder_seed{seed}.pt"
        torch.save(coles_module._seq_encoder.state_dict(), encoder_checkpoint_path)

        torch.cuda.empty_cache()
        train_embeddings = extract_embeddings(
            coles_module, trainer, dataset.train_records, inference_batch_size=128
        )
        test_embeddings = extract_embeddings(
            coles_module, trainer, dataset.test_records, inference_batch_size=128
        )

        train_customer_ids = np.array([record["customer_id"] for record in dataset.train_records])
        test_customer_ids = np.array([record["customer_id"] for record in dataset.test_records])
        embedding_directory = save_coles_embeddings(
            dataset_name=DATASET_NAME,
            seed=seed,
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
            train_targets=dataset.train_targets,
            test_targets=dataset.test_targets,
            train_customer_ids=train_customer_ids,
            test_customer_ids=test_customer_ids,
        )
        downstream_metrics = evaluate_all_classifiers(
            train_embeddings=train_embeddings,
            train_targets=dataset.train_targets,
            test_embeddings=test_embeddings,
            test_targets=dataset.test_targets,
            task_type="multiclass",
            seed=seed,
        )
        for classifier_name, metric_dict in downstream_metrics.items():
            accuracy = metric_dict["accuracy"]
            all_per_seed_metrics.append({"seed": seed, "model": classifier_name, "accuracy": accuracy})
            print(f"  {classifier_name:<8} acc={accuracy:.4f}")

        del coles_module, trainer
        torch.cuda.empty_cache()
        gc.collect()

    elapsed_total = time.time() - overall_start_time

    metrics_dataframe = pd.DataFrame(all_per_seed_metrics)
    metrics_dataframe.to_csv(OUTPUT_DIRECTORY / "age_coles_per_seed.csv", index=False)

    for classifier_name in ["lgbm", "logreg", "xgboost"]:
        subset = metrics_dataframe[metrics_dataframe["model"] == classifier_name]
        print(f"  {classifier_name:<8} acc = {subset['accuracy'].mean():.4f} ± {subset['accuracy'].std():.4f}")

    summary_path = OUTPUT_DIRECTORY / "age_summary.json"
    with summary_path.open("w") as file_handle:
        json.dump(
            {
                "experiment": "CoLES Age (paper config)",
                "config": {
                    "hidden_size": coles_config.hidden_size,
                    "rnn_type": coles_config.rnn_type,
                    "batch_size": coles_config.batch_size,
                    "learning_rate": coles_config.learning_rate,
                    "n_epochs": coles_config.num_epochs,
                    "split_count": coles_config.split_count,
                    "cnt_min": coles_config.minimum_sequence_length,
                    "cnt_max": coles_config.maximum_sequence_length,
                    "embeddings_noise": coles_config.embeddings_noise,
                    "lr_step_size": coles_config.lr_step_size,
                    "lr_gamma": coles_config.lr_gamma,
                },
                "emb_dims": coles_config.embedding_dimensions,
                "seeds": SEEDS,
                "time": elapsed_total,
                "date": time.strftime("%Y-%m-%d %H:%M"),
            },
            file_handle,
            indent=2,
        )

    canonical_metric = metrics_dataframe[metrics_dataframe["model"] == "lgbm"]["accuracy"].mean()
    save_experiment_result(
        experiment_id=EXPERIMENT_BASE_ID,
        rq="RQ1",
        method="CoLES baseline",
        dataset=DATASET_NAME,
        task_type="multiclass",
        metrics={"accuracy": float(canonical_metric)},
        config={
            "hidden_size": coles_config.hidden_size,
            "rnn_type": coles_config.rnn_type,
            "batch_size": coles_config.batch_size,
            "learning_rate": coles_config.learning_rate,
            "epochs": coles_config.num_epochs,
        },
        seed=SEEDS[0],
        runtime_seconds=elapsed_total,
        artifacts={"embeddings_directory": str(embedding_directory)},
    )

if __name__ == "__main__":
    main()
