"""Train XGBoost/CatBoost on different feature variants."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _make_xgb(seed: int = 42):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=seed,
        verbosity=0,
    )


def _make_catboost(seed: int = 42):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        eval_metric="AUC",
        random_seed=seed,
        verbose=0,
    )


def train_variant_a(X_train, y_train, X_test, model_type: str = "xgboost", seed: int = 42):
    """Variant (a): baseline features only."""
    model = _make_xgb(seed) if model_type == "xgboost" else _make_catboost(seed)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return model, proba


def train_variant_b(
    X_train, y_train, X_test,
    pseudo_labels_train: np.ndarray,
    pseudo_probs_train: np.ndarray,
    pseudo_labels_test: np.ndarray,
    pseudo_probs_test: np.ndarray,
    model_type: str = "xgboost",
    seed: int = 42,
):
    """Variant (b): features + LLM pseudo-label and probability as extra features."""
    X_train_aug = np.column_stack([X_train, pseudo_labels_train, pseudo_probs_train])
    X_test_aug = np.column_stack([X_test, pseudo_labels_test, pseudo_probs_test])

    model = _make_xgb(seed) if model_type == "xgboost" else _make_catboost(seed)
    model.fit(X_train_aug, y_train)
    proba = model.predict_proba(X_test_aug)[:, 1]
    return model, proba


def train_variant_c(
    X_train, y_train, X_test,
    explanations_train: list[str],
    explanations_test: list[str],
    model_type: str = "xgboost",
    seed: int = 42,
    max_tfidf_features: int = 100,
):
    """Variant (c): features + TF-IDF of LLM explanations."""
    tfidf = TfidfVectorizer(max_features=max_tfidf_features, stop_words="english")
    tfidf_train = tfidf.fit_transform(explanations_train)
    tfidf_test = tfidf.transform(explanations_test)

    X_train_aug = np.hstack([X_train, tfidf_train.toarray()])
    X_test_aug = np.hstack([X_test, tfidf_test.toarray()])

    model = _make_xgb(seed) if model_type == "xgboost" else _make_catboost(seed)
    model.fit(X_train_aug, y_train)
    proba = model.predict_proba(X_test_aug)[:, 1]
    return model, proba


def train_variant_d(
    y_train,
    pseudo_labels_train: np.ndarray,
    pseudo_probs_train: np.ndarray,
    pseudo_labels_test: np.ndarray,
    pseudo_probs_test: np.ndarray,
    model_type: str = "xgboost",
    seed: int = 42,
):
    """Variant (d): LLM pseudo-labels only (measures raw LLM quality)."""
    X_train_d = np.column_stack([pseudo_labels_train, pseudo_probs_train])
    X_test_d = np.column_stack([pseudo_labels_test, pseudo_probs_test])

    model = _make_xgb(seed) if model_type == "xgboost" else _make_catboost(seed)
    model.fit(X_train_d, y_train)
    proba = model.predict_proba(X_test_d)[:, 1]
    return model, proba


def run_all_variants(
    X_train, y_train, X_test, y_test,
    pseudo_labels_train: np.ndarray,
    pseudo_probs_train: np.ndarray,
    pseudo_labels_test: np.ndarray,
    pseudo_probs_test: np.ndarray,
    explanations_train: list[str],
    explanations_test: list[str],
    model_type: str = "xgboost",
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Train all 4 variants and return predictions.

    Returns dict mapping variant name -> predicted probabilities on test set.
    """
    results = {}

    print(f"Training variant (a): baseline features [{model_type}]...")
    _, proba_a = train_variant_a(X_train, y_train, X_test, model_type, seed)
    results["a_baseline"] = proba_a

    print(f"Training variant (b): features + pseudo-labels [{model_type}]...")
    _, proba_b = train_variant_b(
        X_train, y_train, X_test,
        pseudo_labels_train, pseudo_probs_train,
        pseudo_labels_test, pseudo_probs_test,
        model_type, seed,
    )
    results["b_with_pseudo"] = proba_b

    print(f"Training variant (c): features + TF-IDF explanations [{model_type}]...")
    _, proba_c = train_variant_c(
        X_train, y_train, X_test,
        explanations_train, explanations_test,
        model_type, seed,
    )
    results["c_with_tfidf"] = proba_c

    print(f"Training variant (d): pseudo-labels only [{model_type}]...")
    _, proba_d = train_variant_d(
        y_train,
        pseudo_labels_train, pseudo_probs_train,
        pseudo_labels_test, pseudo_probs_test,
        model_type, seed,
    )
    results["d_pseudo_only"] = proba_d

    return results
