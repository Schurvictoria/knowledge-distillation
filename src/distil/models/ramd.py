from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MaxAbsScaler


_DEFAULT_KNN_NEIGHBORS = 10
_DEFAULT_NUM_FOLDS = 5
_DEFAULT_KNN_METRIC = "cosine"


@dataclass
class RamdStep1Config:
    num_folds: int = _DEFAULT_NUM_FOLDS
    knn_neighbors: int = _DEFAULT_KNN_NEIGHBORS
    knn_metric: str = _DEFAULT_KNN_METRIC
    api_seed: int = 42
    api_temperature: float = 0.0
    api_max_tokens: int = 200


@dataclass
class RamdStep2Config:
    num_rounds: int = 3
    finetune_epochs_per_round: int = 10
    finetune_batch_size: int = 32
    finetune_learning_rate: float = 5e-4
    soft_label_distillation_weight: float = 0.3
    classification_weight: float = 0.7


def compute_oof_fold_indices(
    targets: np.ndarray,
    num_folds: int = _DEFAULT_NUM_FOLDS,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(targets)), targets))


def find_knn_neighbors_for_fold(
    train_embeddings_fold: np.ndarray,
    query_embeddings_fold: np.ndarray,
    train_targets_fold: np.ndarray,
    knn_neighbors: int = _DEFAULT_KNN_NEIGHBORS,
    metric: str = _DEFAULT_KNN_METRIC,
) -> np.ndarray:
    scaler = MaxAbsScaler()
    train_scaled = scaler.fit_transform(train_embeddings_fold)
    query_scaled = scaler.transform(query_embeddings_fold)

    nearest_neighbor_searcher = NearestNeighbors(n_neighbors=knn_neighbors, metric=metric)
    nearest_neighbor_searcher.fit(train_scaled)
    _, neighbor_indices = nearest_neighbor_searcher.kneighbors(query_scaled)
    return neighbor_indices


def aggregate_neighbor_class_counts(
    neighbor_indices: np.ndarray,
    train_targets: np.ndarray,
    binary_label_pair: tuple[str, str] | None = None,
) -> list[dict[str, int]]:
    aggregated_counts = []
    for query_neighbor_indices in neighbor_indices:
        neighbor_class_labels = train_targets[query_neighbor_indices]
        if binary_label_pair is not None:
            positive_label, negative_label = binary_label_pair
            class_counts = {
                positive_label: int((neighbor_class_labels == 1).sum()),
                negative_label: int((neighbor_class_labels == 0).sum()),
            }
        else:
            unique_classes, counts = np.unique(neighbor_class_labels, return_counts=True)
            class_counts = {str(class_label): int(count) for class_label, count in zip(unique_classes, counts)}
        aggregated_counts.append(class_counts)
    return aggregated_counts


def cache_oof_predictions(
    output_path: Path,
    soft_predictions: np.ndarray,
    target_labels: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, probs=soft_predictions, y=target_labels)


def load_cached_oof_predictions(cache_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not cache_path.exists():
        return None
    cached_data = np.load(cache_path)
    return cached_data["probs"], cached_data["y"]
