"""Model training and serialization module for Amazon ML Challenge 2026.

Supports:
- Logistic Regression (with StandardScaler)
- XGBoost (hist method)
- LightGBM

All models adhere to the challenge requirements:
- Permissive licenses (Apache 2.0 / MIT)
- Size <= 8B parameters (< 10 MB)
- Fully reproducible with fixed random states
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features import FEATURE_NAMES


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    max_iter: int = 40,
    tol: float = 1e-3,
    penalty: str = "l2",
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = 0,
) -> Tuple[LogisticRegression, StandardScaler]:
    """Train Logistic Regression with StandardScaler.

    Uses 'saga' solver with l2 regularization for fast convergence on large matrices.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        solver="saga",
        penalty=penalty,
        C=C,
        max_iter=max_iter,
        tol=tol,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
    )
    model.fit(X_scaled, y)
    return model, scaler


def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    min_child_weight: int = 5,
    subsample: float = 0.8,
    colsample_bytree: float = 0.9,
    reg_lambda: float = 5.0,
    random_state: int = 42,
    verbose: int = 0,
) -> XGBClassifier:
    """Train XGBoost binary classifier using histogram-based tree method."""
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X, y, verbose=bool(verbose))
    return model


def train_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 250,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_child_samples: int = 100,
    subsample: float = 0.8,
    colsample_bytree: float = 0.9,
    reg_lambda: float = 5.0,
    random_state: int = 42,
    verbose: int = -1,
) -> LGBMClassifier:
    """Train LightGBM binary classifier."""
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=-1,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        objective="binary",
        random_state=random_state,
        n_jobs=-1,
        verbose=verbose,
    )
    model.fit(X, y)
    return model


def predict_probabilities(
    model: Any,
    X: np.ndarray,
    scaler: Optional[StandardScaler] = None,
) -> np.ndarray:
    """Compute binary match probabilities (class 1) from features."""
    if scaler is not None:
        X = scaler.transform(X)
    probs = model.predict_proba(X)
    return probs[:, 1]


def save_model_bundle(
    model_dir: Union[str, Path],
    model_name: str,
    model: Any,
    scaler: Optional[StandardScaler] = None,
    feature_names: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model, optional scaler, feature names, and metadata into a directory."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if feature_names is None:
        feature_names = FEATURE_NAMES

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "metadata": metadata or {},
    }

    filepath = model_dir / f"{model_name}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    if metadata:
        meta_path = model_dir / f"{model_name}_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def load_model_bundle(
    model_dir: Union[str, Path],
    model_name: str,
) -> Dict[str, Any]:
    """Load model bundle containing model, scaler, feature names, and metadata."""
    model_dir = Path(model_dir)
    filepath = model_dir / f"{model_name}.pkl"
    if not filepath.exists():
        raise FileNotFoundError(f"Model bundle not found: {filepath}")

    with open(filepath, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def run_model_tests():
    """Unit assertions to verify model training, prediction, and serialization."""
    np.random.seed(42)
    N = 200
    K = len(FEATURE_NAMES)
    X = np.random.randn(N, K).astype(np.float32)
    # Simple linear decision boundary with noise
    y = ((X[:, 0] + X[:, 1] * 2.0 - X[:, 6]) > 0).astype(np.int32)

    # 1. Logistic
    lr_model, scaler = train_logistic_regression(X, y, max_iter=20)
    probs_lr = predict_probabilities(lr_model, X, scaler=scaler)
    assert probs_lr.shape == (N,)
    assert (probs_lr >= 0.0).all() and (probs_lr <= 1.0).all()

    # 2. XGBoost
    xgb = train_xgboost(X, y, n_estimators=10, max_depth=3)
    probs_xgb = predict_probabilities(xgb, X)
    assert probs_xgb.shape == (N,)
    assert (probs_xgb >= 0.0).all() and (probs_xgb <= 1.0).all()

    # 3. LightGBM
    lgb = train_lightgbm(X, y, n_estimators=10, num_leaves=7)
    probs_lgb = predict_probabilities(lgb, X)
    assert probs_lgb.shape == (N,)
    assert (probs_lgb >= 0.0).all() and (probs_lgb <= 1.0).all()

    # 4. Save and load test
    test_dir = Path("/tmp/amazon_ml_model_test")
    save_model_bundle(test_dir, "test_lr", lr_model, scaler=scaler, metadata={"test": True})
    loaded = load_model_bundle(test_dir, "test_lr")
    probs_loaded = predict_probabilities(loaded["model"], X, scaler=loaded["scaler"])
    np.testing.assert_allclose(probs_lr, probs_loaded)

    print("All model unit tests PASSED.")


if __name__ == "__main__":
    run_model_tests()
