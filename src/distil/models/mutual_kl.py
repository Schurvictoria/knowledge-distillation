from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MutualKLConfig:
    classification_weight: float = 0.7
    contrastive_weight: float = 0.2
    mutual_kl_weight: float = 0.1
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    num_epochs: int = 10
    batch_size: int = 16
    eval_every_n_epochs: int = 5
    kl_temperature: float = 2.0


def compute_symmetric_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    student_log_probabilities = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probabilities = F.log_softmax(teacher_logits / temperature, dim=-1)

    student_probabilities = student_log_probabilities.exp()
    teacher_probabilities = teacher_log_probabilities.exp()

    student_to_teacher_kl = F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="batchmean",
    )
    teacher_to_student_kl = F.kl_div(
        teacher_log_probabilities,
        student_probabilities,
        reduction="batchmean",
    )
    return (student_to_teacher_kl + teacher_to_student_kl) / 2 * (temperature ** 2)


def compute_combined_bidirectional_loss(
    classification_loss: torch.Tensor,
    contrastive_loss: torch.Tensor,
    mutual_kl_loss: torch.Tensor,
    config: MutualKLConfig,
) -> torch.Tensor:
    return (
        config.classification_weight * classification_loss
        + config.contrastive_weight * contrastive_loss
        + config.mutual_kl_weight * mutual_kl_loss
    )
