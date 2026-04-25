"""
LATTE finetune — Phase 2 для E1.2.

После того как CoLES baseline обучен (Phase 1), эта функция файнтьюнит его
с двумя лоссами одновременно:
  1. Classification (BCE для binary / CE для multiclass)
  2. InfoNCE contrastive между CoLES embeddings и LLM4ES text embeddings

Total loss = (1-alpha) * classification + alpha * contrastive.
α=0.1 канонично для всех 3 датасетов (см. ablations/E1_2_gender_latte_alpha_ablation.py).

Honest model selection: track best by VAL, save checkpoint at best-val epoch.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MaxAbsScaler, StandardScaler


# CLIP-style temperature для InfoNCE
_DEFAULT_INFONCE_TEMPERATURE = 0.07
_DEFAULT_PROJECTION_HIDDEN = 256
_DEFAULT_PROJECTION_OUTPUT = 128


@dataclass
class LatteFinetuneConfig:
    distillation_weight: float = 0.1
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    num_epochs: int = 10
    batch_size: int = 128
    projection_hidden_dimension: int = _DEFAULT_PROJECTION_HIDDEN
    projection_output_dimension: int = _DEFAULT_PROJECTION_OUTPUT
    classifier_dropout: float = 0.3
    eval_every_n_epochs: int = 5
    infonce_temperature: float = _DEFAULT_INFONCE_TEMPERATURE
    cosine_scheduler_t_max: int = 30


def build_projection_head(input_dimension: int, hidden_dimension: int, output_dimension: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dimension, hidden_dimension),
        nn.ReLU(),
        nn.Linear(hidden_dimension, output_dimension),
    )


def build_classifier_head(
    input_dimension: int,
    hidden_dimension: int,
    output_dimension: int,
    dropout_rate: float,
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dimension, hidden_dimension),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dimension, output_dimension),
    )


def initialize_xavier(module: nn.Module) -> None:
    for parameter in module.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)


def compute_symmetric_infonce_loss(
    sequence_projections: torch.Tensor,
    text_projections: torch.Tensor,
    temperature: float = _DEFAULT_INFONCE_TEMPERATURE,
) -> torch.Tensor:
    normalized_sequence = F.normalize(sequence_projections, dim=1)
    normalized_text = F.normalize(text_projections, dim=1)

    similarity_logits = normalized_sequence @ normalized_text.T / temperature
    diagonal_targets = torch.arange(len(normalized_sequence), device=normalized_sequence.device)

    loss_sequence_to_text = F.cross_entropy(similarity_logits, diagonal_targets)
    loss_text_to_sequence = F.cross_entropy(similarity_logits.T, diagonal_targets)
    return (loss_sequence_to_text + loss_text_to_sequence) / 2


def standardize_text_embeddings(
    train_text_embeddings: np.ndarray,
    test_text_embeddings: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    scaler = StandardScaler()
    standardized_train = scaler.fit_transform(train_text_embeddings)
    standardized_test = scaler.transform(test_text_embeddings)
    return (
        torch.FloatTensor(standardized_train).to(device),
        torch.FloatTensor(standardized_test).to(device),
    )


def extract_sequence_embeddings(encoder, records, device, batch_size: int = 64):
    from ptls.data_load.datasets import inference_data_loader

    encoder.eval()
    inference_loader = inference_data_loader(records, num_workers=0, batch_size=batch_size)
    embedding_chunks = []
    with torch.no_grad():
        for batch in inference_loader:
            batch_on_device = batch.to(device)
            embedding_chunks.append(encoder(batch_on_device).cpu())
    return torch.cat(embedding_chunks).numpy()


def _evaluate_with_lgbm(
    encoder,
    train_records,
    val_records,
    test_records,
    train_targets,
    val_targets,
    test_targets,
    task_type: str,
    device: torch.device,
    seed: int,
) -> tuple[float, float]:
    train_embeddings = extract_sequence_embeddings(encoder, train_records, device)
    val_embeddings = extract_sequence_embeddings(encoder, val_records, device)
    test_embeddings = extract_sequence_embeddings(encoder, test_records, device)

    scaler = MaxAbsScaler()
    scaled_train = scaler.fit_transform(train_embeddings)
    scaled_val = scaler.transform(val_embeddings)
    scaled_test = scaler.transform(test_embeddings)

    from lightgbm import LGBMClassifier

    if task_type == "binary":
        from sklearn.metrics import roc_auc_score
        params = dict(
            n_estimators=500, learning_rate=0.02, max_depth=6, subsample=0.5,
            colsample_bytree=0.75, reg_alpha=1, reg_lambda=1, min_child_samples=50,
            verbosity=-1,
        )
        classifier = LGBMClassifier(**params, random_state=seed).fit(scaled_train, train_targets)
        val_score = roc_auc_score(val_targets, classifier.predict_proba(scaled_val)[:, 1])
        test_score = roc_auc_score(test_targets, classifier.predict_proba(scaled_test)[:, 1])
        return float(val_score), float(test_score)

    from sklearn.metrics import accuracy_score
    num_classes = len(np.unique(np.concatenate([train_targets, val_targets, test_targets])))
    params = dict(
        n_estimators=1000, learning_rate=0.02, objective="multiclass",
        num_class=num_classes, max_depth=12, num_leaves=50, subsample=0.75,
        colsample_bytree=0.75, reg_alpha=1, reg_lambda=1, min_child_samples=50,
        verbosity=-1,
    )
    classifier = LGBMClassifier(**params, random_state=seed).fit(scaled_train, train_targets)
    val_score = accuracy_score(val_targets, classifier.predict(scaled_val))
    test_score = accuracy_score(test_targets, classifier.predict(scaled_test))
    return float(val_score), float(test_score)


def train_latte_finetune(
    sequence_encoder,
    train_records: list[dict],
    val_records: list[dict],
    test_records: list[dict],
    train_targets: np.ndarray,
    val_targets: np.ndarray,
    test_targets: np.ndarray,
    text_embeddings_train: torch.Tensor,
    coles_baseline_checkpoint_path: Path,
    finetuned_checkpoint_path: Path,
    sequence_embedding_dimension: int,
    text_embedding_dimension: int,
    config: LatteFinetuneConfig,
    task_type: str,
    device: torch.device,
    baseline_test_score: float,
    seed: int = 42,
) -> dict[str, float]:
    """Phase 2 of LATTE: contrastive alignment with classification co-training."""
    sequence_encoder.load_state_dict(torch.load(coles_baseline_checkpoint_path, map_location=device))
    sequence_encoder.train()

    num_classifier_outputs = 1 if task_type == "binary" else int(np.unique(train_targets).size)

    sequence_projection_head = build_projection_head(
        sequence_embedding_dimension,
        config.projection_hidden_dimension,
        config.projection_output_dimension,
    ).to(device)
    text_projection_head = build_projection_head(
        text_embedding_dimension,
        config.projection_hidden_dimension,
        config.projection_output_dimension,
    ).to(device)
    classifier_head = build_classifier_head(
        sequence_embedding_dimension,
        config.projection_hidden_dimension,
        num_classifier_outputs,
        config.classifier_dropout,
    ).to(device)

    for module_to_initialize in (sequence_projection_head, text_projection_head, classifier_head):
        initialize_xavier(module_to_initialize)

    trainable_parameters = (
        list(sequence_encoder.parameters())
        + list(sequence_projection_head.parameters())
        + list(text_projection_head.parameters())
        + list(classifier_head.parameters())
    )
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.cosine_scheduler_t_max)

    classification_loss_fn = (
        nn.BCEWithLogitsLoss() if task_type == "binary" else nn.CrossEntropyLoss()
    )

    initial_val_score, _ = _evaluate_with_lgbm(
        sequence_encoder, train_records, val_records, test_records,
        train_targets, val_targets, test_targets,
        task_type, device, seed,
    )
    best_val_score = initial_val_score
    best_test_score = baseline_test_score
    best_epoch = 0

    from ptls.data_load.datasets import inference_data_loader

    for epoch_index in range(config.num_epochs):
        sequence_encoder.train()
        sequence_projection_head.train()
        classifier_head.train()

        shuffled_indices = torch.randperm(len(train_records))
        total_loss = 0.0
        num_batches = 0

        for start_index in range(0, len(train_records), config.batch_size):
            batch_indices = shuffled_indices[start_index : start_index + config.batch_size].tolist()
            batch_records = [train_records[index] for index in batch_indices]

            inference_loader = inference_data_loader(
                batch_records, num_workers=0, batch_size=config.batch_size
            )
            for prepared_batch in inference_loader:
                prepared_batch = prepared_batch.to(device)
                sequence_embedding = sequence_encoder(prepared_batch)

                contrastive_loss = compute_symmetric_infonce_loss(
                    sequence_projection_head(sequence_embedding),
                    text_projection_head(text_embeddings_train[batch_indices]),
                    temperature=config.infonce_temperature,
                )

                if task_type == "binary":
                    classification_logits = classifier_head(sequence_embedding).squeeze(-1)
                    target_tensor = torch.FloatTensor(
                        [train_records[index]["target"] for index in batch_indices]
                    ).to(device)
                else:
                    classification_logits = classifier_head(sequence_embedding)
                    target_tensor = torch.LongTensor(
                        [train_records[index]["target"] for index in batch_indices]
                    ).to(device)

                classification_loss = classification_loss_fn(classification_logits, target_tensor)

                combined_loss = (
                    (1 - config.distillation_weight) * classification_loss
                    + config.distillation_weight * contrastive_loss
                )

                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()

                total_loss += float(combined_loss)
                num_batches += 1

        scheduler.step()

        if (epoch_index + 1) % config.eval_every_n_epochs == 0:
            val_score, test_score = _evaluate_with_lgbm(
                sequence_encoder, train_records, val_records, test_records,
                train_targets, val_targets, test_targets,
                task_type, device, seed,
            )
            if val_score > best_val_score:
                best_val_score = val_score
                best_test_score = test_score
                best_epoch = epoch_index + 1
                finetuned_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(sequence_encoder.state_dict(), finetuned_checkpoint_path)
            print(
                f"  ep {epoch_index+1}: loss={total_loss/num_batches:.4f}, "
                f"val={val_score:.4f} test={test_score:.4f} "
                f"best_val={best_val_score:.4f} best_test={best_test_score:.4f}"
            )

    return {
        "best_val_score": best_val_score,
        "best_test_score": best_test_score,
        "best_epoch": best_epoch,
    }
