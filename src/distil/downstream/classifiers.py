import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import MaxAbsScaler


_LGBM_BINARY_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.02,
    boosting_type="gbdt",
    max_depth=6,
    subsample=0.5,
    subsample_freq=1,
    colsample_bytree=0.75,
    reg_alpha=1.0,
    reg_lambda=1.0,
    min_child_samples=50,
    verbosity=-1,
)

_LGBM_MULTICLASS_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.02,
    boosting_type="gbdt",
    objective="multiclass",
    max_depth=12,
    num_leaves=50,
    subsample=0.75,
    subsample_freq=1,
    colsample_bytree=0.75,
    reg_alpha=1.0,
    reg_lambda=1.0,
    min_child_samples=50,
    verbosity=-1,
)


def _scale_embeddings(train_embeddings: np.ndarray, test_embeddings: np.ndarray):
    scaler = MaxAbsScaler()
    return scaler.fit_transform(train_embeddings), scaler.transform(test_embeddings)


def evaluate_lgbm_classifier(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    test_embeddings: np.ndarray,
    test_targets: np.ndarray,
    task_type: str,
    seed: int = 42,
) -> dict[str, float]:
    from lightgbm import LGBMClassifier

    scaled_train, scaled_test = _scale_embeddings(train_embeddings, test_embeddings)

    if task_type == "binary":
        classifier = LGBMClassifier(**_LGBM_BINARY_PARAMS, random_state=seed)
        classifier.fit(scaled_train, train_targets)
        positive_class_probabilities = classifier.predict_proba(scaled_test)[:, 1]
        binary_predictions = (positive_class_probabilities >= 0.5).astype(int)
        return {
            "roc_auc": float(roc_auc_score(test_targets, positive_class_probabilities)),
            "accuracy": float(accuracy_score(test_targets, binary_predictions)),
            "f1": float(f1_score(test_targets, binary_predictions)),
        }

    if task_type == "multiclass":
        unique_classes = np.unique(np.concatenate([train_targets, test_targets]))
        params = dict(_LGBM_MULTICLASS_PARAMS, num_class=len(unique_classes))
        classifier = LGBMClassifier(**params, random_state=seed)
        classifier.fit(scaled_train, train_targets)
        predictions = classifier.predict(scaled_test)
        return {
            "accuracy": float(accuracy_score(test_targets, predictions)),
            "f1_macro": float(f1_score(test_targets, predictions, average="macro")),
        }

    raise ValueError(f"Unknown task_type: {task_type!r}. Expected 'binary' or 'multiclass'.")


def evaluate_xgboost_classifier(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    test_embeddings: np.ndarray,
    test_targets: np.ndarray,
    task_type: str,
    seed: int = 42,
) -> dict[str, float]:
    from xgboost import XGBClassifier

    scaled_train, scaled_test = _scale_embeddings(train_embeddings, test_embeddings)

    base_params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbosity=0,
    )

    if task_type == "binary":
        classifier = XGBClassifier(eval_metric="auc", **base_params)
        classifier.fit(scaled_train, train_targets)
        positive_class_probabilities = classifier.predict_proba(scaled_test)[:, 1]
        return {"roc_auc": float(roc_auc_score(test_targets, positive_class_probabilities))}

    if task_type == "multiclass":
        unique_classes = np.unique(np.concatenate([train_targets, test_targets]))
        classifier = XGBClassifier(
            objective="multi:softmax",
            num_class=len(unique_classes),
            **base_params,
        )
        classifier.fit(scaled_train, train_targets)
        predictions = classifier.predict(scaled_test)
        return {"accuracy": float(accuracy_score(test_targets, predictions))}

    raise ValueError(f"Unknown task_type: {task_type!r}")


def evaluate_logistic_regression(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    test_embeddings: np.ndarray,
    test_targets: np.ndarray,
    task_type: str,
    seed: int = 42,
) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression

    scaled_train, scaled_test = _scale_embeddings(train_embeddings, test_embeddings)
    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(scaled_train, train_targets)

    if task_type == "binary":
        positive_class_probabilities = classifier.predict_proba(scaled_test)[:, 1]
        return {"roc_auc": float(roc_auc_score(test_targets, positive_class_probabilities))}

    predictions = classifier.predict(scaled_test)
    return {"accuracy": float(accuracy_score(test_targets, predictions))}


def evaluate_all_classifiers(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    test_embeddings: np.ndarray,
    test_targets: np.ndarray,
    task_type: str,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    return {
        "lgbm": evaluate_lgbm_classifier(
            train_embeddings, train_targets, test_embeddings, test_targets, task_type, seed
        ),
        "xgboost": evaluate_xgboost_classifier(
            train_embeddings, train_targets, test_embeddings, test_targets, task_type, seed
        ),
        "logreg": evaluate_logistic_regression(
            train_embeddings, train_targets, test_embeddings, test_targets, task_type, seed
        ),
    }
