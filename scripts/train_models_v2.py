"""Phase 5: Retrain baseline models (Logistic, XGBoost, LightGBM) on corrected features (v2).

Ensures:
1. Strict use of output/training_features_v2/
2. Dimensionality, binary label, and finite value assertions
3. Model training with exact baseline configurations
4. Prediction probability sanity checks (non-degenerate outputs)
5. Model persistence and reload reproducibility tests
6. Preservation of old v1 models
"""

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import FEATURE_NAMES, N_FEATURES
from src.model import (
    load_model_bundle,
    predict_probabilities,
    save_model_bundle,
    train_lightgbm,
    train_logistic_regression,
    train_xgboost,
)


def train_and_validate_models_v2(
    features_dir: Path = PROJECT_ROOT / "output" / "training_features_v2",
    models_dir: Path = PROJECT_ROOT / "output" / "models_v2",
    smoke_sample_size: int = 50_000,
) -> Dict[str, Any]:
    print("=" * 70)
    print("PHASE 5: RETRAIN BASELINE MODELS ON CORRECTED FEATURES (v2)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Load and validate training feature matrix
    # -------------------------------------------------------------------------
    print(f"\n[1/5] Loading feature matrix from {features_dir.name}...")
    t0 = time.time()
    features_path = features_dir / "features.npy"
    labels_path = features_dir / "labels.npy"
    feature_names_path = features_dir / "feature_names.pkl"

    assert features_path.exists(), f"Features file not found: {features_path}"
    assert labels_path.exists(), f"Labels file not found: {labels_path}"
    assert feature_names_path.exists(), f"Feature names file not found: {feature_names_path}"

    X = np.load(features_path)
    y = np.load(labels_path)
    with open(feature_names_path, "rb") as f:
        loaded_feature_names = pickle.load(f)

    n_rows, n_cols = X.shape
    print(f"  Loaded X shape: {X.shape}, y shape: {y.shape} in {time.time() - t0:.1f}s")

    # Assertions
    assert n_cols == N_FEATURES, f"Expected {N_FEATURES} features, got {n_cols}"
    assert n_rows == len(y), f"Row mismatch: X has {n_rows} rows, y has {len(y)}"
    assert loaded_feature_names == FEATURE_NAMES, "Feature names do not match expected order!"
    assert np.isfinite(X).all(), "Non-finite values found in feature matrix!"
    assert np.isnan(X).sum() == 0, "NaN values found in feature matrix!"
    assert np.isinf(X).sum() == 0, "Inf values found in feature matrix!"

    unique_labels = set(np.unique(y))
    assert unique_labels == {0, 1}, f"Labels must be strictly binary {{0, 1}}, got {unique_labels}"

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"  Positives: {n_pos:,} ({n_pos / n_rows:.2%})")
    print(f"  Negatives: {n_neg:,} ({n_neg / n_rows:.2%})")
    print("  Input validation checks: ALL PASSED")

    # Prepare smoke test sample (stratified)
    np.random.seed(42)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    sample_pos = np.random.choice(pos_idx, size=smoke_sample_size // 2, replace=False)
    sample_neg = np.random.choice(neg_idx, size=smoke_sample_size // 2, replace=False)
    smoke_idx = np.concatenate([sample_pos, sample_neg])
    np.random.shuffle(smoke_idx)

    X_smoke = X[smoke_idx]
    y_smoke = y[smoke_idx]
    print(f"  Prepared smoke sample: {len(X_smoke):,} rows ({len(sample_pos):,} pos, {len(sample_neg):,} neg)")

    report_results = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "models": {},
    }

    models_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 2: Logistic Regression
    # -------------------------------------------------------------------------
    print("\n[2/5] Training Logistic Regression...")
    t_start = time.time()
    lr_config = {
        "model_family": "logistic_regression",
        "solver": "saga",
        "penalty": "l2",
        "C": 1.0,
        "max_iter": 40,
        "tol": 1e-3,
        "random_state": 42,
        "scaler": "StandardScaler",
    }
    lr_model, scaler = train_logistic_regression(
        X,
        y,
        C=lr_config["C"],
        max_iter=lr_config["max_iter"],
        tol=lr_config["tol"],
        penalty=lr_config["penalty"],
        random_state=lr_config["random_state"],
    )
    lr_train_time = time.time() - t_start
    print(f"  Logistic Regression trained in {lr_train_time:.1f}s.")

    # Smoke test
    lr_probs = predict_probabilities(lr_model, X_smoke, scaler=scaler)
    lr_stats = evaluate_probabilities(lr_probs)
    print(f"  Smoke stats: min={lr_stats['min']:.4f}, max={lr_stats['max']:.4f}, "
          f"mean={lr_stats['mean']:.4f}, std={lr_stats['std']:.4f}, unique={lr_stats['n_unique']:,}")

    # Persistence & reload test
    lr_dir = models_dir / "logistic"
    save_model_bundle(
        lr_dir,
        "logistic_regression",
        lr_model,
        scaler=scaler,
        feature_names=FEATURE_NAMES,
        metadata={
            **lr_config,
            "n_samples": n_rows,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "train_time_sec": lr_train_time,
            "feature_version": "v2",
        },
    )
    loaded_lr_bundle = load_model_bundle(lr_dir, "logistic_regression")
    reloaded_lr_probs = predict_probabilities(
        loaded_lr_bundle["model"], X_smoke, scaler=loaded_lr_bundle["scaler"]
    )
    np.testing.assert_allclose(lr_probs, reloaded_lr_probs, rtol=1e-5, atol=1e-5)
    print("  Logistic Regression save/reload assertion: PASSED")

    report_results["models"]["logistic"] = {
        "train_status": "PASS",
        "smoke_status": "PASS" if lr_stats["std"] > 0.05 and lr_stats["n_unique"] > 100 else "FAIL",
        "reload_status": "PASS",
        "stats": lr_stats,
    }

    # -------------------------------------------------------------------------
    # Step 3: XGBoost
    # -------------------------------------------------------------------------
    print("\n[3/5] Training XGBoost (hist)...")
    t_start = time.time()
    xgb_config = {
        "model_family": "xgboost",
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "reg_lambda": 5.0,
        "tree_method": "hist",
        "random_state": 42,
    }
    xgb_model = train_xgboost(
        X,
        y,
        n_estimators=xgb_config["n_estimators"],
        max_depth=xgb_config["max_depth"],
        learning_rate=xgb_config["learning_rate"],
        min_child_weight=xgb_config["min_child_weight"],
        subsample=xgb_config["subsample"],
        colsample_bytree=xgb_config["colsample_bytree"],
        reg_lambda=xgb_config["reg_lambda"],
        random_state=xgb_config["random_state"],
    )
    xgb_train_time = time.time() - t_start
    print(f"  XGBoost trained in {xgb_train_time:.1f}s.")

    # Smoke test
    xgb_probs = predict_probabilities(xgb_model, X_smoke)
    xgb_stats = evaluate_probabilities(xgb_probs)
    print(f"  Smoke stats: min={xgb_stats['min']:.4f}, max={xgb_stats['max']:.4f}, "
          f"mean={xgb_stats['mean']:.4f}, std={xgb_stats['std']:.4f}, unique={xgb_stats['n_unique']:,}")

    # Persistence & reload test
    xgb_dir = models_dir / "xgboost"
    save_model_bundle(
        xgb_dir,
        "xgboost",
        xgb_model,
        scaler=None,
        feature_names=FEATURE_NAMES,
        metadata={
            **xgb_config,
            "n_samples": n_rows,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "train_time_sec": xgb_train_time,
            "feature_version": "v2",
        },
    )
    loaded_xgb_bundle = load_model_bundle(xgb_dir, "xgboost")
    reloaded_xgb_probs = predict_probabilities(loaded_xgb_bundle["model"], X_smoke)
    np.testing.assert_allclose(xgb_probs, reloaded_xgb_probs, rtol=1e-5, atol=1e-5)
    print("  XGBoost save/reload assertion: PASSED")

    report_results["models"]["xgboost"] = {
        "train_status": "PASS",
        "smoke_status": "PASS" if xgb_stats["std"] > 0.05 and xgb_stats["n_unique"] > 100 else "FAIL",
        "reload_status": "PASS",
        "stats": xgb_stats,
    }

    # -------------------------------------------------------------------------
    # Step 4: LightGBM
    # -------------------------------------------------------------------------
    print("\n[4/5] Training LightGBM...")
    t_start = time.time()
    lgb_config = {
        "model_family": "lightgbm",
        "n_estimators": 250,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "reg_lambda": 5.0,
        "random_state": 42,
    }
    lgb_model = train_lightgbm(
        X,
        y,
        n_estimators=lgb_config["n_estimators"],
        learning_rate=lgb_config["learning_rate"],
        num_leaves=lgb_config["num_leaves"],
        min_child_samples=lgb_config["min_child_samples"],
        subsample=lgb_config["subsample"],
        colsample_bytree=lgb_config["colsample_bytree"],
        reg_lambda=lgb_config["reg_lambda"],
        random_state=lgb_config["random_state"],
    )
    lgb_train_time = time.time() - t_start
    print(f"  LightGBM trained in {lgb_train_time:.1f}s.")

    # Smoke test
    lgb_probs = predict_probabilities(lgb_model, X_smoke)
    lgb_stats = evaluate_probabilities(lgb_probs)
    print(f"  Smoke stats: min={lgb_stats['min']:.4f}, max={lgb_stats['max']:.4f}, "
          f"mean={lgb_stats['mean']:.4f}, std={lgb_stats['std']:.4f}, unique={lgb_stats['n_unique']:,}")

    # Persistence & reload test
    lgb_dir = models_dir / "lightgbm"
    save_model_bundle(
        lgb_dir,
        "lightgbm",
        lgb_model,
        scaler=None,
        feature_names=FEATURE_NAMES,
        metadata={
            **lgb_config,
            "n_samples": n_rows,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "train_time_sec": lgb_train_time,
            "feature_version": "v2",
        },
    )
    loaded_lgb_bundle = load_model_bundle(lgb_dir, "lightgbm")
    reloaded_lgb_probs = predict_probabilities(loaded_lgb_bundle["model"], X_smoke)
    np.testing.assert_allclose(lgb_probs, reloaded_lgb_probs, rtol=1e-5, atol=1e-5)
    print("  LightGBM save/reload assertion: PASSED")

    report_results["models"]["lightgbm"] = {
        "train_status": "PASS",
        "smoke_status": "PASS" if lgb_stats["std"] > 0.05 and lgb_stats["n_unique"] > 100 else "FAIL",
        "reload_status": "PASS",
        "stats": lgb_stats,
    }

    # -------------------------------------------------------------------------
    # Step 5: Directory & artifact verification
    # -------------------------------------------------------------------------
    print("\n[5/5] Verifying directory integrity...")
    models_v1_dir = PROJECT_ROOT / "output" / "models_v1"
    assert models_v1_dir.exists(), "models_v1 directory missing!"
    assert (models_v1_dir / "xgboost_model.pkl").exists(), "models_v1/xgboost_model.pkl was modified or deleted!"
    print("  Old v1 models preserved: PASS")

    # Save summary json
    summary_path = models_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report_results, f, indent=2)
    print(f"  Summary saved to {summary_path}")

    return report_results


def evaluate_probabilities(probs: np.ndarray) -> Dict[str, Any]:
    assert probs.ndim == 1, f"Expected 1D probabilities, got shape {probs.shape}"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities outside [0, 1]!"
    assert np.isfinite(probs).all(), "Non-finite values in probabilities!"

    p_min = float(probs.min())
    p_max = float(probs.max())
    p_mean = float(probs.mean())
    p_std = float(probs.std())
    n_unique = int(len(np.unique(np.round(probs, decimals=6))))

    # Assert non-degenerate behavior
    if p_std < 0.01 or (p_max - p_min) < 0.10:
        raise ValueError(f"Degenerate probability output detected! min={p_min}, max={p_max}, std={p_std}")

    return {
        "min": p_min,
        "max": p_max,
        "mean": p_mean,
        "std": p_std,
        "n_unique": n_unique,
    }


if __name__ == "__main__":
    train_and_validate_models_v2()
