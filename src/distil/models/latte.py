from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


_DEFAULT_INFONCE_TEMPERATURE = 0.07


@dataclass
class LatteFinetuneConfig:
    distillation_weight: float = 0.1
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    num_epochs: int = 10
    batch_size: int = 128
    projection_dimension: int = 128
    projection_hidden_size: int = 256
    classifier_dropout: float = 0.3
    eval_every_n_epochs: int = 5
    infonce_temperature: float = _DEFAULT_INFONCE_TEMPERATURE


def build_projection_head(input_dimension: int, hidden_dimension: int, output_dimension: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dimension, hidden_dimension),
        nn.ReLU(),
        nn.Linear(hidden_dimension, output_dimension),
    )


def build_binary_classifier_head(input_dimension: int, hidden_dimension: int, dropout_rate: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dimension, hidden_dimension),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dimension, 1),
    )


def build_multiclass_classifier_head(
    input_dimension: int,
    hidden_dimension: int,
    num_classes: int,
    dropout_rate: float,
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dimension, hidden_dimension),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dimension, num_classes),
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
