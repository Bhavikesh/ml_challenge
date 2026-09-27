"""Phase 7: Optimized test inference — generates matching_results.tsv + candidate_pairs.tsv.

Prerequisites:
  1. scripts/build_test_blocking.py   → output/blocking_artifacts_test.pkl
  2. scripts/fast_threshold_estimator.py → output/fast_threshold_results.json

Usage:
  python scripts/run_test_inference.py --auto
  python scripts/run_test_inference.py --model lightgbm --threshold 0.51
  python scripts/run_test_inference.py --auto --resume   # resume after disconnect

Optimisations vs the validation script:
  - Single process (no multiprocessing IPC overhead)
  - One model only (best from threshold selection)
  - Direct per-S1 set accumulation (no histogram)
  - Checkpoint + resume every 500 K rows
  - chunk_size=50_000 (larger = fewer Python loop iterations)
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking import generate_candidates, load_blocking_artifact
from src.features import S1RecordStore, compute_candidate_features_batch
from src.model import load_model_bundle, predict_probabilities

OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR   = PROJECT_ROOT / "data" / "test"
MODELS_DIR = OUTPUT_DIR / "models_v2"
ARTIFACT   = OUTPUT_DIR / "blocking_artifacts_test.pkl"   # TEST artifact
CKPT_PATH  = OUTPUT_DIR / "test_inference_checkpoint.pkl"

# Approximate test S2+S3 totals for progress display
TOTAL_TEST_TARGETS = 4_887_273 + 5_082_316


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="lightgbm",
                   choices=["logistic", "xgboost", "lightgbm"])
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--auto", action="store_true",
                   help="Read model+threshold from fast_threshold_results.json")
    p.add_argument("--chunk-size", type=int, default=50_000)
    p.add_argument("--resume", action="store_true",
                   help="Resume from checkpoint if it exists")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Model + threshold ────────────────────────────────────────────────────
    if args.auto:
        rpath = OUTPUT_DIR / "fast_threshold_results.json"
        if rpath.exists():
            with open(rpath) as f:
                td = json.load(f)
            model_name = max(td["models"], key=lambda m: td["models"][m]["best_macro_f05"])
            threshold  = td["models"][model_name]["best_threshold"]
            print(f"[AUTO] model={model_name}  threshold={threshold:.2f}  "
                  f"(holdout macro_F0.5={td['models'][model_name]['best_macro_f05']:.4f})")
        else:
            print("[AUTO] fast_threshold_results.json not found, using validated best: LightGBM @ 0.51 (F0.5=0.9585)")
            model_name = "lightgbm"
            threshold  = 0.51
    else:
        model_name = args.model
        if args.threshold is None:
            default_thresholds = {"lightgbm": 0.51, "xgboost": 0.51, "logistic": 0.39}
            threshold = default_thresholds.get(model_name, 0.51)
            print(f"[DEFAULT] Using holdout optimal threshold {threshold:.2f} for {model_name}")
        else:
            threshold = args.threshold

    print("=" * 70)
    print("PHASE 7: OPTIMISED TEST INFERENCE")
    print("=" * 70)
    print(f"  Model:          {model_name}")
    print(f"  Threshold:      {threshold:.2f}")
    print(f"  Artifact:       {ARTIFACT}")
    print(f"  Data dir:       {DATA_DIR}")

    if not ARTIFACT.exists():
        raise FileNotFoundError(
            f"Test blocking artifact not found: {ARTIFACT}\n"
            "Run:  python scripts/build_test_blocking.py"
        )

    # ── Load test blocking artifact (built from TEST S1 entities) ───────────
    print("\nLoading test blocking artifact …")
    try:
        artifact     = load_blocking_artifact(ARTIFACT)
    except (EOFError, pickle.UnpicklingError) as e:
        raise RuntimeError(
            f"Failed to load {ARTIFACT}: {e}\n"
            "The artifact file was interrupted during saving (e.g. via ^C).\n"
            "Please re-run: python scripts/build_test_blocking.py"
        ) from e
    test_s1_ids  = artifact["s1_ids"]          # list[str], ordered TEST S1 IDs
    s1_id_to_idx = artifact["s1_id_to_idx"]    # str -> int
    print(f"  Test S1 entities in artifact: {len(test_s1_ids):,}")

    # ── Build S1 record store from test_source1.tsv ──────────────────────────
    print("Building S1 record store from test_source1.tsv …")
    s1_store = S1RecordStore(len(test_s1_ids))
    for chunk in pd.read_csv(DATA_DIR / "test_source1.tsv", sep="\t",
                              dtype=str, chunksize=100_000, keep_default_na=False):
        for row in chunk.itertuples(index=False):
            idx = s1_id_to_idx.get(row.entity_id)
            if idx is not None:
                s1_store.set_record(
                    idx, row.entity_id,
                    row.business_name, row.business_address, row.country
                )
    print("  S1 store ready.")

    # ── Load model ───────────────────────────────────────────────────────────
    MODEL_FNAME = {
        "logistic":  "logistic_regression",
        "xgboost":   "xgboost",
        "lightgbm":  "lightgbm",
    }
    print(f"\nLoading {model_name} model …")
    bundle = load_model_bundle(MODELS_DIR / model_name, MODEL_FNAME[model_name])
    model  = bundle["model"]
    scaler = bundle.get("scaler")
    print("  Model loaded.")

    # ── Resume checkpoint ────────────────────────────────────────────────────
    completed_targets = set()
    predictions = {}   # s1_id -> set[target_id]
    candidates  = {}   # s1_id -> set[target_id]

    if args.resume and CKPT_PATH.exists():
        print(f"\nResuming from checkpoint …")
        with open(CKPT_PATH, "rb") as f:
            ckpt = pickle.load(f)
        completed_targets = ckpt["completed_targets"]
        predictions       = ckpt["predictions"]
        candidates        = ckpt["candidates"]
        print(f"  Already processed {len(completed_targets):,} target rows.")

    # ── Scan test S2 + S3 ────────────────────────────────────────────────────
    total_scanned    = len(completed_targets)
    total_candidates = sum(len(v) for v in candidates.values())
    ckpt_counter     = 0
    CKPT_EVERY       = 500_000
    t_start          = time.time()

    print("\nStreaming test targets (S2 + S3) …")

    for src_num, fname in [(2, "test_source2.tsv"), (3, "test_source3.tsv")]:
        target_path = DATA_DIR / fname
        print(f"\n  Processing {fname} …")

        for df_chunk in pd.read_csv(target_path, sep="\t", dtype=str,
                                     chunksize=args.chunk_size,
                                     keep_default_na=False):

            rows_batch = []
            for row in df_chunk.itertuples(index=False):
                if row.entity_id not in completed_targets:
                    rows_batch.append(
                        (row.entity_id, row.business_name,
                         row.business_address, row.country)
                    )

            if not rows_batch:
                total_scanned += len(df_chunk)
                continue

            # Blocking + features
            feature_blocks = []
            target_meta    = []  # (entity_id, cand_indices, n_cands)

            for entity_id, name, addr, country in rows_batch:
                cands = generate_candidates(name, addr, src_num, artifact)
                if cands:
                    X = compute_candidate_features_batch(
                        cands, name, addr, country, src_num, s1_store
                    )
                    feature_blocks.append(X)
                    target_meta.append((entity_id, cands, len(cands)))
                    total_candidates += len(cands)

            # Inference
            if feature_blocks:
                X_batch = np.vstack(feature_blocks)
                if scaler is not None:
                    X_batch = scaler.transform(X_batch)
                probs = predict_probabilities(model, X_batch)

                offset = 0
                for entity_id, cands, count in target_meta:
                    sub_p = probs[offset: offset + count]
                    offset += count

                    # Resolve S1 IDs
                    cand_s1_ids = [test_s1_ids[c] for c in cands]

                    # Candidates (for candidate_pairs.tsv)
                    for s1_id in cand_s1_ids:
                        candidates.setdefault(s1_id, set()).add(entity_id)

                    # Predictions above threshold (for matching_results.tsv)
                    for s1_id, prob in zip(cand_s1_ids, sub_p):
                        if prob >= threshold:
                            predictions.setdefault(s1_id, set()).add(entity_id)

            completed_targets.update(r[0] for r in rows_batch)
            total_scanned += len(df_chunk)
            ckpt_counter  += len(rows_batch)

            # Progress line
            elapsed = time.time() - t_start + 1e-6
            rate    = total_scanned / elapsed
            pct     = total_scanned / TOTAL_TEST_TARGETS * 100
            eta_s   = (TOTAL_TEST_TARGETS - total_scanned) / max(rate, 1)
            print(
                f"    {total_scanned:>10,} / {TOTAL_TEST_TARGETS:,} "
                f"({pct:5.1f}%)  {rate:5.0f} rows/s  "
                f"ETA {eta_s/3600:.1f}h  cands {total_candidates:,}",
                end="\r",
            )

            # Checkpoint
            if ckpt_counter >= CKPT_EVERY:
                with open(CKPT_PATH, "wb") as f:
                    pickle.dump({
                        "completed_targets": completed_targets,
                        "predictions":       predictions,
                        "candidates":        candidates,
                    }, f, protocol=4)
                ckpt_counter = 0
                print(f"\n  [CHECKPOINT saved at {total_scanned:,} rows]")

    elapsed_total = time.time() - t_start
    print(f"\n\nAll test targets scanned in {elapsed_total/3600:.2f}h  "
          f"| candidates: {total_candidates:,}")

    # ── Load all test S1 IDs (maintain original file order) ─────────────────
    print("\nLoading all test S1 IDs from test_source1.tsv …")
    all_test_s1 = pd.read_csv(DATA_DIR / "test_source1.tsv", sep="\t",
                               dtype=str, usecols=["entity_id"],
                               keep_default_na=False)["entity_id"].tolist()
    print(f"  Total test S1 entities: {len(all_test_s1):,}")

    # ── Write matching_results.tsv ───────────────────────────────────────────
    print("\nWriting matching_results.tsv …")
    out_match = OUTPUT_DIR / "matching_results.tsv"
    with open(out_match, "w", encoding="utf-8") as fout:
        fout.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_test_s1:
            matched = sorted(predictions.get(s1_id, []))
            fout.write(f"{s1_id}\t{','.join(matched)}\n")
    print(f"  ✅ {out_match}  ({len(all_test_s1):,} rows)")

    # ── Write candidate_pairs.tsv ────────────────────────────────────────────
    print("Writing candidate_pairs.tsv …")
    out_cand = OUTPUT_DIR / "candidate_pairs.tsv"
    with open(out_cand, "w", encoding="utf-8") as fout:
        fout.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_test_s1:
            cands_list = sorted(candidates.get(s1_id, []))
            fout.write(f"{s1_id}\t{','.join(cands_list)}\n")
    print(f"  ✅ {out_cand}  ({len(all_test_s1):,} rows)")

    # Cleanup checkpoint
    if CKPT_PATH.exists():
        CKPT_PATH.unlink()

    print("\n" + "=" * 70)
    print("TEST INFERENCE COMPLETE")
    print("=" * 70)
    print(f"  matching_results.tsv : {out_match}")
    print(f"  candidate_pairs.tsv  : {out_cand}")
    print("\nRun the validator before uploading:")
    print("  python utils/validate_submission.py \\")
    print("      --matching output/matching_results.tsv \\")
    print("      --candidate output/candidate_pairs.tsv \\")
    print("      --test-dir data/test")


if __name__ == "__main__":
    main()
