"""Phase 6: Full validation inference and Macro F0.5 threshold sweep (v2).

Evaluates Logistic Regression, XGBoost, and LightGBM on the complete validation set:
- 220,682 validation S1 entities
- 10,320,219 targets (5,034,616 S2 + 5,285,603 S3)
- Blocker: blocking_artifacts_v2.pkl
- Threshold sweep from 0.10 to 0.99
- Entity-level Macro F0.5 (official challenge metric)
- Stability analysis, source breakdown, entity type breakdown, country breakdown
- Comprehensive error classification
"""

import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Prevent thread oversubscription in worker processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking import generate_candidates, load_blocking_artifact
from src.data_processing import normalize_country
from src.features import S1RecordStore, compute_candidate_features_batch
from src.inference import compute_entity_f05
from src.model import load_model_bundle, predict_probabilities

# Global state for worker processes
_w_artifact = None
_w_s1_store = None
_w_lr_model = None
_w_lr_scaler = None
_w_xgb_model = None
_w_lgb_model = None
_w_true_targets = None
_w_thresholds = None


def init_worker(artifact_path: Path, models_dir: Path, s1_path: Path, gt_path: Path, thresholds: np.ndarray):
    global _w_artifact, _w_s1_store, _w_lr_model, _w_lr_scaler, _w_xgb_model, _w_lgb_model
    global _w_true_targets, _w_thresholds

    _w_thresholds = thresholds
    _w_artifact = load_blocking_artifact(artifact_path)
    val_s1_ids = _w_artifact["s1_ids"]
    val_s1_id_to_idx = _w_artifact["s1_id_to_idx"]

    # S1 Store
    _w_s1_store = S1RecordStore(len(val_s1_ids))
    for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, chunksize=100_000, keep_default_na=False):
        for row in chunk.itertuples(index=False):
            idx = val_s1_id_to_idx.get(row.entity_id)
            if idx is not None:
                _w_s1_store.set_record(idx, row.entity_id, row.business_name, row.business_address, row.country)

    # Models
    lr_b = load_model_bundle(models_dir / "logistic", "logistic_regression")
    xgb_b = load_model_bundle(models_dir / "xgboost", "xgboost")
    lgb_b = load_model_bundle(models_dir / "lightgbm", "lightgbm")

    _w_lr_model = lr_b["model"]
    _w_lr_scaler = lr_b["scaler"]
    _w_xgb_model = xgb_b["model"]
    _w_lgb_model = lgb_b["model"]

    try:
        _w_xgb_model.set_params(n_jobs=1)
    except Exception:
        pass
    try:
        _w_lgb_model.set_params(n_jobs=1)
    except Exception:
        pass

    # Ground truth mapping: target_id -> s1_idx
    _w_true_targets = {}
    for chunk in pd.read_csv(gt_path, sep="\t", dtype=str, chunksize=100_000, keep_default_na=False):
        for row in chunk.itertuples(index=False):
            s1_idx = val_s1_id_to_idx.get(row.source1_entity_id)
            if s1_idx is not None and row.matched_entity_ids:
                for tid in row.matched_entity_ids.split(","):
                    tid = tid.strip()
                    if tid:
                        _w_true_targets[tid] = s1_idx


def process_target_chunk(chunk_data: Tuple[List[Tuple[str, str, str, str]], int]):
    rows, source_num = chunk_data
    results = {
        "source_num": source_num,
        "n_rows": len(rows),
        "n_candidates": 0,
        "updates": {"logistic": [], "xgboost": [], "lightgbm": []},
        "fp_samples": [],
    }

    feature_blocks = []
    target_meta = []

    for entity_id, name, addr, country in rows:
        cands = generate_candidates(name, addr, source_num, _w_artifact)
        if cands:
            X = compute_candidate_features_batch(cands, name, addr, country, source_num, _w_s1_store)
            feature_blocks.append(X)
            target_meta.append((entity_id, cands, len(cands), country))
            results["n_candidates"] += len(cands)

    if not feature_blocks:
        return results

    X_chunk = np.vstack(feature_blocks)
    p_lr = predict_probabilities(_w_lr_model, X_chunk, scaler=_w_lr_scaler)
    p_xgb = predict_probabilities(_w_xgb_model, X_chunk)
    p_lgb = predict_probabilities(_w_lgb_model, X_chunk)

    offset = 0
    min_thresh = _w_thresholds[0]

    for entity_id, cands, count, t_country in target_meta:
        sub_lr = p_lr[offset : offset + count]
        sub_xgb = p_xgb[offset : offset + count]
        sub_lgb = p_lgb[offset : offset + count]
        sub_X = X_chunk[offset : offset + count]
        offset += count

        true_s1 = _w_true_targets.get(entity_id)
        c_arr = np.array(cands, dtype=np.int32)
        tp_mask = (c_arr == true_s1) if true_s1 is not None else np.zeros(len(c_arr), dtype=bool)

        for m_key, probs in [("logistic", sub_lr), ("xgboost", sub_xgb), ("lightgbm", sub_lgb)]:
            valid = (probs >= min_thresh)
            if valid.any():
                v_s1 = c_arr[valid]
                v_bins = np.searchsorted(_w_thresholds, probs[valid], side="right").astype(np.int16)
                v_tp = tp_mask[valid].astype(np.uint8)
                results["updates"][m_key].append((v_s1, v_bins, v_tp))

        # Sample FP records for XGBoost at prob >= 0.70
        if len(results["fp_samples"]) < 5:
            high_xgb_fp = (sub_xgb >= 0.70) & (~tp_mask)
            if high_xgb_fp.any():
                for idx_in_sub in np.flatnonzero(high_xgb_fp)[:2]:
                    results["fp_samples"].append({
                        "target_id": entity_id,
                        "source_num": source_num,
                        "s1_idx": int(c_arr[idx_in_sub]),
                        "xgb_prob": float(sub_xgb[idx_in_sub]),
                        "features": sub_X[idx_in_sub].tolist(),
                    })

    return results


def compute_vectorized_f05(
    true_counts: np.ndarray, pred_counts: np.ndarray, tp_counts: np.ndarray
) -> np.ndarray:
    """Vectorized calculation of macro F0.5 per S1 entity matching challenge rules."""
    n = len(true_counts)
    f05 = np.zeros(n, dtype=np.float64)

    # 1. Singletons (true_count == 0): 1.0 if pred == 0 else 0.0
    zero_true = (true_counts == 0)
    f05[zero_true] = np.where(pred_counts[zero_true] == 0, 1.0, 0.0)

    # 2. Positive match entities (true_count > 0):
    pos_true = ~zero_true
    has_tp = pos_true & (tp_counts > 0)

    if np.any(has_tp):
        p = tp_counts[has_tp] / pred_counts[has_tp].astype(np.float64)
        r = tp_counts[has_tp] / true_counts[has_tp].astype(np.float64)
        denom = 0.25 * p + r
        valid_denom = denom > 0
        f05[has_tp] = np.where(valid_denom, (1.25 * p * r) / np.maximum(denom, 1e-12), 0.0)

    return f05


def run_validation_evaluation(
    output_dir: Path = PROJECT_ROOT / "output",
    data_dir: Path = PROJECT_ROOT / "data" / "train",
    chunk_size: int = 10_000,
    n_workers: int = 8,
) -> Dict[str, Any]:
    start_time = time.time()
    print("=" * 70)
    print("PHASE 6: FULL VALIDATION INFERENCE & MACRO F0.5 THRESHOLD SWEEP")
    print("=" * 70)

    # 1. Thresholds grid: 0.10 to 0.99 with 0.01 step
    thresholds = np.round(np.arange(0.10, 0.995, 0.01), 2)
    n_thresholds = len(thresholds)
    n_bins = n_thresholds + 1
    print(f"Threshold sweep grid: {n_thresholds} thresholds from {thresholds[0]:.2f} to {thresholds[-1]:.2f}")

    # 2. Load artifact & ground truth
    artifact_path = output_dir / "blocking_artifacts_v2.pkl"
    artifact = load_blocking_artifact(artifact_path)
    val_s1_ids = artifact["s1_ids"]
    val_s1_id_to_idx = artifact["s1_id_to_idx"]
    n_val_s1 = len(val_s1_ids)
    print(f"Validation S1 entities: {n_val_s1:,}")

    # 3. Load S1 countries & ground truth
    print("\nLoading validation S1 ground truth & country metadata...")
    s1_countries = [""] * n_val_s1
    for chunk in pd.read_csv(data_dir / "train_source1.tsv", sep="\t", dtype=str, chunksize=100_000, keep_default_na=False):
        for row in chunk.itertuples(index=False):
            idx = val_s1_id_to_idx.get(row.entity_id)
            if idx is not None:
                s1_countries[idx] = normalize_country(row.country)

    true_counts = np.zeros(n_val_s1, dtype=np.int32)
    s2_true_counts = np.zeros(n_val_s1, dtype=np.int32)
    s3_true_counts = np.zeros(n_val_s1, dtype=np.int32)
    val_gt_map: Dict[str, Set[str]] = {}

    for chunk in pd.read_csv(data_dir / "train_ground_truth.tsv", sep="\t", dtype=str, chunksize=100_000, keep_default_na=False):
        for row in chunk.itertuples(index=False):
            idx = val_s1_id_to_idx.get(row.source1_entity_id)
            if idx is not None:
                matches = {m.strip() for m in row.matched_entity_ids.split(",") if m.strip()}
                val_gt_map[row.source1_entity_id] = matches
                true_counts[idx] = len(matches)
                for m in matches:
                    if m.startswith("S2-"):
                        s2_true_counts[idx] += 1
                    elif m.startswith("S3-"):
                        s3_true_counts[idx] += 1

    total_true_rel = int(true_counts.sum())
    total_s2_true = int(s2_true_counts.sum())
    total_s3_true = int(s3_true_counts.sum())
    print(f"Total true validation relationships: {total_true_rel:,} (S2: {total_s2_true:,}, S3: {total_s3_true:,})")
    assert total_true_rel == 764_045, f"Expected 764,045 true links, got {total_true_rel}"

    # Entity masks
    no_match_mask = (true_counts == 0)
    singleton_mask = (true_counts == 1)
    multi_match_mask = (true_counts > 1)
    us_mask = np.array([c == "us" for c in s1_countries], dtype=bool)
    india_mask = np.array([c == "india" for c in s1_countries], dtype=bool)

    print(f"  No-match S1:    {int(no_match_mask.sum()):,}")
    print(f"  Singleton S1:   {int(singleton_mask.sum()):,}")
    print(f"  Multi-match S1: {int(multi_match_mask.sum()):,}")
    print(f"  US S1:          {int(us_mask.sum()):,}")
    print(f"  India S1:       {int(india_mask.sum()):,}")

    # 4. Initialize histograms for all 3 models: pred_hist[model][src], tp_hist[model][src]
    models = ["logistic", "xgboost", "lightgbm"]
    # Separate for source 2 and source 3
    pred_hists = {
        m: {2: np.zeros((n_val_s1, n_bins), dtype=np.int32), 3: np.zeros((n_val_s1, n_bins), dtype=np.int32)}
        for m in models
    }
    tp_hists = {
        m: {2: np.zeros((n_val_s1, n_bins), dtype=np.int32), 3: np.zeros((n_val_s1, n_bins), dtype=np.int32)}
        for m in models
    }

    # 5. Multiprocessing Pool
    print(f"\nLaunching inference worker pool ({n_workers} workers)...")
    models_dir = output_dir / "models_v2"
    s1_path = data_dir / "train_source1.tsv"
    gt_path = data_dir / "train_ground_truth.tsv"

    pool = mp.Pool(
        n_workers,
        initializer=init_worker,
        initargs=(artifact_path, models_dir, s1_path, gt_path, thresholds),
    )

    # Generator for chunks across S2 and S3
    def generate_chunks():
        for src_num, fname in [(2, "train_source2.tsv"), (3, "train_source3.tsv")]:
            target_file = data_dir / fname
            chunk_rows = []
            for df_chunk in pd.read_csv(target_file, sep="\t", dtype=str, chunksize=100_000, keep_default_na=False):
                for row in df_chunk.itertuples(index=False):
                    chunk_rows.append((row.entity_id, row.business_name, row.business_address, row.country))
                    if len(chunk_rows) >= chunk_size:
                        yield (chunk_rows, src_num)
                        chunk_rows = []
            if chunk_rows:
                yield (chunk_rows, src_num)

    total_candidates_scored = 0
    total_targets_scanned = 0
    collected_fp_samples = []
    t_scan_start = time.time()

    print("\nStreaming validation inference across S2 and S3...")
    for res in pool.imap_unordered(process_target_chunk, generate_chunks(), chunksize=1):
        total_targets_scanned += res["n_rows"]
        total_candidates_scored += res["n_candidates"]
        src = res["source_num"]

        for m in models:
            for v_s1, v_bins, v_tp in res["updates"][m]:
                flat_idx = v_s1.astype(np.int64) * n_bins + v_bins.astype(np.int64)
                block_pred = np.bincount(flat_idx, minlength=n_val_s1 * n_bins).reshape(n_val_s1, n_bins)
                pred_hists[m][src] += block_pred.astype(np.int32, copy=False)

                pos_mask = (v_tp == 1)
                if pos_mask.any():
                    block_tp = np.bincount(flat_idx[pos_mask], minlength=n_val_s1 * n_bins).reshape(n_val_s1, n_bins)
                    tp_hists[m][src] += block_tp.astype(np.int32, copy=False)

        if len(collected_fp_samples) < 500 and res["fp_samples"]:
            collected_fp_samples.extend(res["fp_samples"])

        if total_targets_scanned % 500_000 < chunk_size:
            elapsed = time.time() - t_scan_start
            rate = total_targets_scanned / max(elapsed, 1.0)
            print(f"  Targets scanned: {total_targets_scanned:,} / 10,320,219 "
                  f"({total_targets_scanned / 10_320_219:.1%}) | Candidates: {total_candidates_scored:,} | {rate:.0f} rows/s")

    pool.close()
    pool.join()
    t_scan = time.time() - t_scan_start
    print(f"\nAll 10,320,219 targets scanned in {t_scan:.1f}s. Scored {total_candidates_scored:,} candidates.")

    # 6. Threshold Sweep Analysis for Each Model
    print("\nComputing Macro F0.5 threshold sweep across all models...")
    model_evaluations: Dict[str, Any] = {}

    for m in models:
        m_start = time.time()
        # Combine S2 and S3 histograms
        tot_pred_hist = pred_hists[m][2] + pred_hists[m][3]
        tot_tp_hist = tp_hists[m][2] + tp_hists[m][3]

        # Suffix cumulative sums: pred_counts_at_t[s1_idx] = sum(tot_pred_hist[s1_idx, t_idx:])
        # Reverse cumsum along axis 1
        cum_pred = np.cumsum(tot_pred_hist[:, ::-1], axis=1)[:, ::-1]
        cum_tp = np.cumsum(tot_tp_hist[:, ::-1], axis=1)[:, ::-1]

        # Separate cumulative counts for source breakdown
        cum_pred_s2 = np.cumsum(pred_hists[m][2][:, ::-1], axis=1)[:, ::-1]
        cum_tp_s2 = np.cumsum(tp_hists[m][2][:, ::-1], axis=1)[:, ::-1]
        cum_pred_s3 = np.cumsum(pred_hists[m][3][:, ::-1], axis=1)[:, ::-1]
        cum_tp_s3 = np.cumsum(tp_hists[m][3][:, ::-1], axis=1)[:, ::-1]

        sweep_rows = []
        best_f05 = -1.0
        best_t_idx = -1

        for t_idx in range(n_thresholds):
            # Threshold index k corresponds to bin k + 1
            bin_idx = t_idx + 1
            preds_t = cum_pred[:, bin_idx]
            tp_t = cum_tp[:, bin_idx]

            # Vectorized Macro F0.5
            f05_scores = compute_vectorized_f05(true_counts, preds_t, tp_t)
            macro_f05 = float(f05_scores.mean())

            # Pair-level metrics
            tp_pair = int(tp_t.sum())
            pred_pair = int(preds_t.sum())
            fp_pair = pred_pair - tp_pair
            fn_pair = total_true_rel - tp_pair

            precision = (tp_pair / float(pred_pair)) if pred_pair > 0 else 0.0
            recall = (tp_pair / float(total_true_rel)) if total_true_rel > 0 else 0.0

            nonempty_rate = float((preds_t > 0).mean() * 100.0)
            avg_preds = float(preds_t.mean())
            median_preds = float(np.median(preds_t))

            row_data = {
                "threshold": float(thresholds[t_idx]),
                "t_idx": t_idx,
                "macro_f05": macro_f05,
                "precision": precision,
                "recall": recall,
                "tp": tp_pair,
                "fp": fp_pair,
                "fn": fn_pair,
                "nonempty_rate": nonempty_rate,
                "avg_preds_per_s1": avg_preds,
                "median_preds_per_s1": median_preds,
            }
            sweep_rows.append(row_data)

            if macro_f05 > best_f05:
                best_f05 = macro_f05
                best_t_idx = t_idx

        best_row = sweep_rows[best_t_idx]
        best_bin = best_t_idx + 1
        best_preds = cum_pred[:, best_bin]
        best_tp = cum_tp[:, best_bin]

        # -------------------------------------------------------------
        # Threshold Stability (best-0.02, best-0.01, best, best+0.01, best+0.02)
        # -------------------------------------------------------------
        stability = {}
        for offset_val in [-0.02, -0.01, 0.0, 0.01, 0.02]:
            target_t = round(best_row["threshold"] + offset_val, 2)
            match_row = next((r for r in sweep_rows if abs(r["threshold"] - target_t) < 1e-4), None)
            key_name = f"best{offset_val:+.2f}" if offset_val != 0 else "best"
            stability[key_name] = {
                "threshold": target_t,
                "macro_f05": match_row["macro_f05"] if match_row else None,
            }

        # -------------------------------------------------------------
        # Entity Breakdown at best threshold
        # -------------------------------------------------------------
        # A. No-match entities
        no_match_preds = best_preds[no_match_mask]
        no_match_zero = int((no_match_preds == 0).sum())
        no_match_nonzero = int((no_match_preds > 0).sum())

        # B. Singleton entities
        sing_preds = best_preds[singleton_mask]
        sing_tp = best_tp[singleton_mask]
        sing_correct_match = int((sing_preds == 1) & (sing_tp == 1)).sum()
        sing_correct_empty = int(sing_preds == 0).sum() # note: true count is 1, so empty is FN
        sing_wrong_match = int((sing_preds > 0) & (sing_tp == 0)).sum()
        sing_exact_match_rate = float(sing_correct_match / max(int(singleton_mask.sum()), 1))
        sing_fp_rate = float(sing_wrong_match / max(int(singleton_mask.sum()), 1))

        # C. Multi-match entities
        multi_preds = best_preds[multi_match_mask]
        multi_tp = best_tp[multi_match_mask]
        multi_true = true_counts[multi_match_mask]
        multi_all = int(multi_tp == multi_true).sum()
        multi_some = int((multi_tp > 0) & (multi_tp < multi_true)).sum()
        multi_none = int(multi_tp == 0).sum()

        entity_breakdown = {
            "no_match": {
                "total": int(no_match_mask.sum()),
                "zero_predictions": no_match_zero,
                "nonzero_predictions": no_match_nonzero,
                "zero_pred_pct": float(no_match_zero / int(no_match_mask.sum()) * 100.0),
            },
            "singleton": {
                "total": int(singleton_mask.sum()),
                "exact_singleton_matches": int(sing_correct_match),
                "exact_singleton_match_rate": sing_exact_match_rate,
                "false_positive_rate": sing_fp_rate,
                "incorrectly_matched": int(sing_wrong_match),
                "unmatched_empty": int(sing_correct_empty),
            },
            "multi_match": {
                "total": int(multi_match_mask.sum()),
                "all_captured": int(multi_all),
                "some_captured": int(multi_some),
                "none_captured": int(multi_none),
                "all_captured_pct": float(multi_all / int(multi_match_mask.sum()) * 100.0),
            },
        }

        # -------------------------------------------------------------
        # Source Breakdown at best threshold
        # -------------------------------------------------------------
        s2_pred_best = int(cum_pred_s2[:, best_bin].sum())
        s2_tp_best = int(cum_tp_s2[:, best_bin].sum())
        s2_prec = float(s2_tp_best / s2_pred_best) if s2_pred_best > 0 else 0.0
        s2_rec = float(s2_tp_best / total_s2_true) if total_s2_true > 0 else 0.0

        s3_pred_best = int(cum_pred_s3[:, best_bin].sum())
        s3_tp_best = int(cum_tp_s3[:, best_bin].sum())
        s3_prec = float(s3_tp_best / s3_pred_best) if s3_pred_best > 0 else 0.0
        s3_rec = float(s3_tp_best / total_s3_true) if total_s3_true > 0 else 0.0

        source_breakdown = {
            "S2": {
                "predicted": s2_pred_best,
                "correct": s2_tp_best,
                "precision": s2_prec,
                "recall": s2_rec,
            },
            "S3": {
                "predicted": s3_pred_best,
                "correct": s3_tp_best,
                "precision": s3_prec,
                "recall": s3_rec,
            },
        }

        # -------------------------------------------------------------
        # Country Breakdown at best threshold
        # -------------------------------------------------------------
        us_f05 = compute_vectorized_f05(true_counts[us_mask], best_preds[us_mask], best_tp[us_mask])
        india_f05 = compute_vectorized_f05(true_counts[india_mask], best_preds[india_mask], best_tp[india_mask])

        us_tp = int(best_tp[us_mask].sum())
        us_pred = int(best_preds[us_mask].sum())
        us_true = int(true_counts[us_mask].sum())

        india_tp = int(best_tp[india_mask].sum())
        india_pred = int(best_preds[india_mask].sum())
        india_true = int(true_counts[india_mask].sum())

        country_breakdown = {
            "US": {
                "s1_count": int(us_mask.sum()),
                "macro_f05": float(us_f05.mean()),
                "precision": float(us_tp / us_pred) if us_pred > 0 else 0.0,
                "recall": float(us_tp / us_true) if us_true > 0 else 0.0,
                "tp": us_tp,
                "predicted": us_pred,
            },
            "India": {
                "s1_count": int(india_mask.sum()),
                "macro_f05": float(india_f05.mean()),
                "precision": float(india_tp / india_pred) if india_pred > 0 else 0.0,
                "recall": float(india_tp / india_true) if india_true > 0 else 0.0,
                "tp": india_tp,
                "predicted": india_pred,
            },
        }

        # -------------------------------------------------------------
        # Error Breakdown at best threshold
        # -------------------------------------------------------------
        # Blocking misses vs model threshold misses
        blocking_misses = 36_841 # Fixed ground truth candidate recall ceiling
        candidate_captured = 727_204
        model_misses = candidate_captured - best_row["tp"]

        error_breakdown = {
            "type_a_blocking_misses": blocking_misses,
            "type_b_model_threshold_misses": model_misses,
            "total_false_negatives": best_row["fn"],
            "total_false_positives": best_row["fp"],
        }

        model_evaluations[m] = {
            "best_threshold": best_row["threshold"],
            "best_macro_f05": best_row["macro_f05"],
            "precision": best_row["precision"],
            "recall": best_row["recall"],
            "tp": best_row["tp"],
            "fp": best_row["fp"],
            "fn": best_row["fn"],
            "nonempty_rate": best_row["nonempty_rate"],
            "avg_preds_per_s1": best_row["avg_preds_per_s1"],
            "median_preds_per_s1": best_row["median_preds_per_s1"],
            "stability": stability,
            "entity_breakdown": entity_breakdown,
            "source_breakdown": source_breakdown,
            "country_breakdown": country_breakdown,
            "error_breakdown": error_breakdown,
            "sweep_results": sweep_rows,
        }
        print(f"  [{m.upper()}] Best threshold = {best_row['threshold']:.2f} | "
              f"Macro F0.5 = {best_row['macro_f05']:.4f} | Prec: {best_row['precision']:.4f} | "
              f"Rec: {best_row['recall']:.4f} (took {time.time() - m_start:.1f}s)")

    # 7. FP Pattern Analysis from sampled FPs
    fp_pattern_summary = analyze_fp_samples(collected_fp_samples)

    # 8. Save results to output
    final_output = {
        "n_validation_s1": n_val_s1,
        "total_true_relationships": total_true_rel,
        "blocking_candidate_recall": 0.9517816358984091,
        "blocking_misses": 36_841,
        "models": {m: {k: v for k, v in model_evaluations[m].items() if k != "sweep_results"} for m in models},
        "fp_patterns": fp_pattern_summary,
        "elapsed_seconds": time.time() - start_time,
    }

    json_path = output_dir / "validation_model_results_v2.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved validation results to {json_path}")

    # Export sweep CSV
    csv_rows = []
    for m in models:
        for r in model_evaluations[m]["sweep_results"]:
            csv_rows.append({
                "model": m,
                "threshold": r["threshold"],
                "macro_f05": r["macro_f05"],
                "precision": r["precision"],
                "recall": r["recall"],
                "tp": r["tp"],
                "fp": r["fp"],
                "fn": r["fn"],
                "nonempty_rate": r["nonempty_rate"],
                "avg_predictions_per_s1": r["avg_preds_per_s1"],
                "median_predictions_per_s1": r["median_preds_per_s1"],
            })
    csv_path = output_dir / "validation_model_results_v2.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Saved threshold sweep CSV to {csv_path}")

    return {
        "evaluations": model_evaluations,
        "fp_patterns": fp_pattern_summary,
        "summary": final_output,
    }


def analyze_fp_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {"sample_count": 0}

    total_fps = len(samples)
    high_name_low_addr = 0
    high_addr_low_name = 0
    weak_both = 0
    same_country = 0
    s2_count = 0
    s3_count = 0

    for s in samples:
        feat = s["features"]
        n_ratio = feat[1]
        a_ratio = feat[7]
        c_same = feat[12]
        src_is_s3 = feat[15]

        if c_same == 1.0:
            same_country += 1
        if src_is_s3 == 1.0:
            s3_count += 1
        else:
            s2_count += 1

        if n_ratio >= 0.70 and a_ratio < 0.40:
            high_name_low_addr += 1
        elif a_ratio >= 0.70 and n_ratio < 0.40:
            high_addr_low_name += 1
        else:
            weak_both += 1

    return {
        "sample_count": total_fps,
        "same_country_pct": float(same_country / total_fps * 100.0),
        "s2_pct": float(s2_count / total_fps * 100.0),
        "s3_pct": float(s3_count / total_fps * 100.0),
        "high_name_low_addr_pct": float(high_name_low_addr / total_fps * 100.0),
        "high_addr_low_name_pct": float(high_addr_low_name / total_fps * 100.0),
        "mixed_or_weak_both_pct": float(weak_both / total_fps * 100.0),
    }


if __name__ == "__main__":
    run_validation_evaluation()
