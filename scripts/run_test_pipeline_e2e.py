"""All-in-one End-to-End Test Pipeline for Amazon ML Challenge 2026.

This script executes the entire Phase 7 test pipeline in a single command:
  1. Checks for /content/blocking_artifacts_test.pkl (builds it on local SSD if missing)
  2. Loads the holdout-validated LightGBM model (threshold=0.51)
  3. Streams test_source2.tsv and test_source3.tsv, generates candidates, extracts features, predicts matches
  4. Writes output/matching_results.tsv and output/candidate_pairs.tsv
  5. Automatically executes utils/validate_submission.py

Usage:
  python scripts/run_test_pipeline_e2e.py
"""

import gc
import json
import os
import pickle
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking import (
    ADDRESS_TOKEN_FREQUENCY_THRESHOLD,
    NAME_TOKEN_FREQUENCY_THRESHOLD,
    PAIR_S1_THRESHOLD,
    TARGET_PAIR_POSTING_MAX_FREQUENCY,
    TARGET_PAIR_THRESHOLD,
    build_s1_blocking_indexes,
    generate_candidates,
    load_blocking_artifact,
    save_blocking_artifact,
    scan_target_pair_frequencies,
)
from src.features import S1RecordStore, compute_candidate_features_batch
from src.model import load_model_bundle, predict_probabilities

OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR   = PROJECT_ROOT / "data" / "test"
MODELS_DIR = OUTPUT_DIR / "models_v2"

# Pick fast local path on Colab
if Path("/content").exists():
    ARTIFACT_PATH = Path("/content/blocking_artifacts_test.pkl")
    CKPT_PATH     = Path("/content/test_inference_checkpoint.pkl")
else:
    ARTIFACT_PATH = OUTPUT_DIR / "blocking_artifacts_test.pkl"
    CKPT_PATH     = OUTPUT_DIR / "test_inference_checkpoint.pkl"

TOTAL_TEST_TARGETS = 4_887_273 + 5_082_316
CHUNK_SIZE = 5_000


def get_threshold_and_model():
    """Load model name and threshold, fallback to holdout-validated LightGBM @ 0.51."""
    rpath = OUTPUT_DIR / "fast_threshold_results.json"
    if rpath.exists():
        try:
            with open(rpath) as f:
                td = json.load(f)
            model_name = max(td["models"], key=lambda m: td["models"][m]["best_macro_f05"])
            threshold  = td["models"][model_name]["best_threshold"]
            print(f"[CONFIG] Holdout optimal: {model_name} @ {threshold:.2f} (Macro F0.5={td['models'][model_name]['best_macro_f05']:.4f})")
            return model_name, threshold
        except Exception:
            pass
    print("[CONFIG] Using default validated: lightgbm @ 0.51")
    return "lightgbm", 0.51


def ensure_blocking_artifact():
    """Ensure test blocking artifact exists on fast local disk."""
    if ARTIFACT_PATH.exists() and ARTIFACT_PATH.stat().st_size > 20 * 1024 * 1024:
        print(f"\n[STEP 1] Found existing valid blocking artifact: {ARTIFACT_PATH} ({ARTIFACT_PATH.stat().st_size/(1024*1024):.1f} MB)")
        try:
            return load_blocking_artifact(ARTIFACT_PATH)
        except Exception as e:
            print(f"Warning: Corrupted artifact ({e}), rebuilding fresh...")
            ARTIFACT_PATH.unlink()

    print("\n" + "=" * 70)
    print("[STEP 1/4] BUILDING TEST BLOCKING ARTIFACT ON FAST LOCAL DISK")
    print("=" * 70)
    t0 = time.time()

    # Load S1
    s1_path = DATA_DIR / "test_source1.tsv"
    print(f"Loading {s1_path.name}...")
    s1_rows: List[Tuple[str, str, str]] = []
    for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, chunksize=100_000, keep_default_na=False):
        for row in chunk.itertuples(index=False):
            s1_rows.append((row.entity_id, row.business_name, row.business_address))
    print(f"  Loaded {len(s1_rows):,} test S1 records.")

    # Build index
    print("Building indexes...")
    artifact = build_s1_blocking_indexes(
        s1_rows,
        name_token_threshold=NAME_TOKEN_FREQUENCY_THRESHOLD,
        address_token_threshold=ADDRESS_TOKEN_FREQUENCY_THRESHOLD,
        pair_s1_threshold=PAIR_S1_THRESHOLD,
        target_pair_posting_max_frequency=TARGET_PAIR_POSTING_MAX_FREQUENCY,
    )
    del s1_rows
    gc.collect()

    relevant_name_pairs    = set(artifact["name_pair_to_s1"].keys())
    relevant_address_pairs = set(artifact["address_pair_to_s1"].keys())
    token_to_id            = artifact["token_to_id"]

    for src_num, fname in [(2, "test_source2.tsv"), (3, "test_source3.tsv")]:
        print(f"Scanning pair frequencies for {fname}...")
        t_name, t_addr = scan_target_pair_frequencies(
            DATA_DIR / fname,
            relevant_name_pairs=relevant_name_pairs,
            relevant_address_pairs=relevant_address_pairs,
            token_to_id=token_to_id,
            chunk_size=100_000,
            source_label=f"Source {src_num}",
        )
        artifact[f"target_name_pair_frequency_{src_num}"]    = Counter({k: v for k, v in t_name.items() if 0 < v <= TARGET_PAIR_THRESHOLD})
        artifact[f"target_address_pair_frequency_{src_num}"] = Counter({k: v for k, v in t_addr.items() if 0 < v <= TARGET_PAIR_THRESHOLD})
        del t_name, t_addr
        gc.collect()

    print(f"✅ In-memory blocking index ready in {time.time()-t0:.0f}s. Proceeding directly to inference...")
    return artifact


def main():
    start_all = time.time()
    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026: END-TO-END TEST PIPELINE")
    print("=" * 70)
    print(f"Data directory:   {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    model_name, threshold = get_threshold_and_model()

    # Step 1: Blocking artifact
    artifact = ensure_blocking_artifact()
    test_s1_ids  = artifact["s1_ids"]
    s1_id_to_idx = artifact["s1_id_to_idx"]

    # Step 2: Build S1 record store
    print("\n" + "=" * 70)
    print("[STEP 2/4] BUILDING S1 IN-MEMORY STORE")
    print("=" * 70)
    s1_store = S1RecordStore(len(test_s1_ids))
    for chunk in pd.read_csv(DATA_DIR / "test_source1.tsv", sep="\t", dtype=str, chunksize=100_000, keep_default_na=False):
        for row in chunk.itertuples(index=False):
            idx = s1_id_to_idx.get(row.entity_id)
            if idx is not None:
                s1_store.set_record(idx, row.entity_id, row.business_name, row.business_address, row.country)
    print(f"✅ S1 store populated with {len(test_s1_ids):,} entities.")

    # Step 3: Load Model
    print("\n" + "=" * 70)
    print(f"[STEP 3/4] LOADING MODEL ({model_name})")
    print("=" * 70)
    MODEL_FNAME = {"logistic": "logistic_regression", "xgboost": "xgboost", "lightgbm": "lightgbm"}
    bundle = load_model_bundle(MODELS_DIR / model_name, MODEL_FNAME[model_name])
    model  = bundle["model"]
    scaler = bundle.get("scaler")
    print(f"✅ Loaded {model_name} model successfully.")

    # Step 4: Streaming Inference
    print("\n" + "=" * 70)
    print(f"[STEP 4/4] STREAMING INFERENCE ACROSS TEST SOURCES (Threshold: {threshold})")
    print("=" * 70)

    completed_targets: Set[str] = set()
    predictions: Dict[str, Set[str]] = {}  # s1_id -> set of target IDs
    candidates:  Dict[str, Set[str]] = {}  # s1_id -> set of candidate target IDs

    if CKPT_PATH.exists():
        try:
            print(f"Loading checkpoint from {CKPT_PATH}...")
            with open(CKPT_PATH, "rb") as f:
                ckpt = pickle.load(f)
            completed_targets = ckpt["completed_targets"]
            predictions       = ckpt["predictions"]
            candidates        = ckpt["candidates"]
            print(f"  Resumed from {len(completed_targets):,} already processed targets.")
        except Exception as e:
            print(f"Checkpoint unreadable ({e}), starting clean.")
            completed_targets.clear()

    total_scanned = len(completed_targets)
    total_candidates = sum(len(v) for v in candidates.values())
    ckpt_counter = 0
    CKPT_EVERY = 500_000
    t_start = time.time()

    for src_num, fname in [(2, "test_source2.tsv"), (3, "test_source3.tsv")]:
        target_path = DATA_DIR / fname
        print(f"\nProcessing {fname} ({src_num}/3)...")

        for df_chunk in pd.read_csv(target_path, sep="\t", dtype=str, chunksize=CHUNK_SIZE, keep_default_na=False):
            rows_batch = []
            for row in df_chunk.itertuples(index=False):
                if row.entity_id not in completed_targets:
                    rows_batch.append((row.entity_id, row.business_name, row.business_address, row.country))

            if not rows_batch:
                total_scanned += len(df_chunk)
                continue

            feature_blocks = []
            target_meta    = []

            for entity_id, name, addr, country in rows_batch:
                cands = generate_candidates(name, addr, src_num, artifact)
                if cands:
                    if len(cands) > 30:
                        cands = cands[:30]
                    X = compute_candidate_features_batch(cands, name, addr, country, src_num, s1_store)
                    feature_blocks.append(X)
                    target_meta.append((entity_id, cands, len(cands)))
                    total_candidates += len(cands)

            if feature_blocks:
                X_batch = np.vstack(feature_blocks)
                if scaler is not None:
                    X_batch = scaler.transform(X_batch)
                probs = predict_probabilities(model, X_batch)

                offset = 0
                for entity_id, cands, count in target_meta:
                    sub_p = probs[offset: offset + count]
                    offset += count

                    cand_s1_ids = [test_s1_ids[c] for c in cands]
                    for s1_id in cand_s1_ids:
                        candidates.setdefault(s1_id, set()).add(entity_id)

                    for s1_id, prob in zip(cand_s1_ids, sub_p):
                        if prob >= threshold:
                            predictions.setdefault(s1_id, set()).add(entity_id)

            completed_targets.update(r[0] for r in rows_batch)
            total_scanned += len(df_chunk)
            ckpt_counter  += len(rows_batch)

            elapsed = time.time() - t_start + 1e-6
            rate    = (total_scanned - len(completed_targets) + len(rows_batch)) / elapsed
            pct     = total_scanned / TOTAL_TEST_TARGETS * 100
            eta_s   = (TOTAL_TEST_TARGETS - total_scanned) / max(rate, 1)

            if total_scanned % 50_000 < len(df_chunk):
                print(
                    f"  [{pct:5.1f}%] {total_scanned:>10,} / {TOTAL_TEST_TARGETS:,} "
                    f"| {rate:5.0f} rows/s | ETA: {eta_s/3600:4.1f}h | cands: {total_candidates:,}",
                    flush=True,
                )
            else:
                print(
                    f"  [{pct:5.1f}%] {total_scanned:>10,} / {TOTAL_TEST_TARGETS:,} "
                    f"| {rate:5.0f} rows/s | ETA: {eta_s/3600:4.1f}h | cands: {total_candidates:,}",
                    end="\r",
                    flush=True,
                )

            if ckpt_counter >= CKPT_EVERY:
                with open(CKPT_PATH, "wb") as f:
                    pickle.dump({"completed_targets": completed_targets, "predictions": predictions, "candidates": candidates}, f, protocol=4)
                ckpt_counter = 0
                print(f"  [Checkpoint saved at {total_scanned:,} rows]", flush=True)

    print(f"\n\nInference completed in {(time.time()-t_start)/3600:.2f} hours.")

    # Step 5: Write Output TSVs
    print("\n" + "=" * 70)
    print("WRITING FINAL SUBMISSION FILES")
    print("=" * 70)

    all_test_s1 = pd.read_csv(DATA_DIR / "test_source1.tsv", sep="\t", dtype=str, usecols=["entity_id"], keep_default_na=False)["entity_id"].tolist()
    print(f"Total test S1 rows required: {len(all_test_s1):,}")

    out_match = OUTPUT_DIR / "matching_results.tsv"
    out_cand  = OUTPUT_DIR / "candidate_pairs.tsv"

    print(f"Writing {out_match}...")
    with open(out_match, "w", encoding="utf-8") as fout:
        fout.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_test_s1:
            matched = sorted(predictions.get(s1_id, []))
            fout.write(f"{s1_id}\t{','.join(matched)}\n")

    print(f"Writing {out_cand}...")
    with open(out_cand, "w", encoding="utf-8") as fout:
        fout.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_test_s1:
            cands_list = sorted(candidates.get(s1_id, []))
            fout.write(f"{s1_id}\t{','.join(cands_list)}\n")

    if CKPT_PATH.exists():
        CKPT_PATH.unlink()

    print("\n" + "=" * 70)
    print("VALIDATING GENERATED FILES WITH OFFICIAL VALIDATOR")
    print("=" * 70)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "utils" / "validate_submission.py"),
        "--matching", str(out_match),
        "--candidate", str(out_cand),
        "--test-dir", str(DATA_DIR),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

    total_pipeline_time = time.time() - start_all
    print("=" * 70)
    print(f"PIPELINE COMPLETE IN {total_pipeline_time/3600:.2f} HOURS")
    print(f"matching_results.tsv: {out_match.stat().st_size / (1024*1024):.2f} MB")
    print(f"candidate_pairs.tsv:  {out_cand.stat().st_size / (1024*1024):.2f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
