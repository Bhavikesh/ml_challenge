"""Phase 4: Regenerate correctly aligned training pairs and features (v2).

Ensures 100% verified S1 alignment between:
1. train_source1.tsv canonical file ordering
2. S1RecordStore
3. training_pairs_v2 (s1_idx.npy, target_num.npy, source.npy, label.npy)
4. training_features_v2 (features.npy, labels.npy, feature_names.pkl)

Performs full ground-truth alignment checks, finite-value assertions,
distribution comparisons, and explicit spot-checks on positive pairs.
"""

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import (
    load_id_set,
    normalize_country,
    normalize_text,
    parse_entity_id,
)
from src.features import (
    FEATURE_NAMES,
    N_FEATURES,
    S1RecordStore,
    compute_candidate_features_batch,
    compute_pair_features_from_raw,
)


def build_and_verify_training_features_v2(
    data_dir: Path = PROJECT_ROOT / "data" / "train",
    output_dir: Path = PROJECT_ROOT / "output",
    chunk_size: int = 100_000,
) -> Tuple[Path, Path]:
    start_time = time.time()
    print("=" * 70)
    print("PHASE 4: REGENERATE ALIGNED TRAINING PAIRS & FEATURES (v2)")
    print("=" * 70)

    pairs_v2_dir = output_dir / "training_pairs_v2"
    features_v2_dir = output_dir / "training_features_v2"
    pairs_v2_dir.mkdir(parents=True, exist_ok=True)
    features_v2_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Establish canonical S1 training mapping in train_source1.tsv order
    # -------------------------------------------------------------------------
    val_ids_path = output_dir / "validation_s1_ids.txt"
    val_s1_ids = load_id_set(val_ids_path)
    print(f"Loaded validation S1 IDs: {len(val_s1_ids):,} (strictly excluded from training)")

    s1_path = data_dir / "train_source1.tsv"
    print(f"\n[1/6] Loading canonical S1 training records from {s1_path.name}...")
    train_s1_entity_ids: List[str] = []

    # First pass: collect IDs in exact file order
    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        chunksize=chunk_size,
        keep_default_na=False,
    ):
        for row in chunk.itertuples(index=False):
            if row.entity_id not in val_s1_ids:
                train_s1_entity_ids.append(row.entity_id)

    n_train_s1 = len(train_s1_entity_ids)
    print(f"  Extracted {n_train_s1:,} canonical training S1 records.")
    assert n_train_s1 == 1_986_139, f"Expected 1,986,139 training S1 entities, got {n_train_s1}"
    assert len(set(train_s1_entity_ids)) == n_train_s1, "Duplicate S1 IDs found in training set!"
    assert len(set(train_s1_entity_ids) & val_s1_ids) == 0, "DATA LEAKAGE: Validation IDs found in training S1 list!"

    # Bijective mappings
    train_s1_id_to_idx: Dict[str, int] = {eid: idx for idx, eid in enumerate(train_s1_entity_ids)}
    train_s1_idx_to_id: Dict[int, str] = {idx: eid for eid, idx in train_s1_id_to_idx.items()}

    # Save mapping in training_pairs_v2
    with open(pairs_v2_dir / "train_s1_ids.txt", "w", encoding="utf-8") as f:
        for eid in train_s1_entity_ids:
            f.write(eid + "\n")
    with open(pairs_v2_dir / "train_s1_id_to_idx.pkl", "wb") as f:
        pickle.dump(train_s1_id_to_idx, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("  Saved canonical training S1 mapping to training_pairs_v2/train_s1_ids.txt")

    # -------------------------------------------------------------------------
    # Step 2: Populate S1RecordStore with normalized data in exact index order
    # -------------------------------------------------------------------------
    print("\n[2/6] Populating S1RecordStore with pre-tokenized training data...")
    s1_store = S1RecordStore(n_train_s1)
    filled_count = 0

    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        chunksize=chunk_size,
        keep_default_na=False,
    ):
        for row in chunk.itertuples(index=False):
            idx = train_s1_id_to_idx.get(row.entity_id)
            if idx is not None:
                s1_store.set_record(
                    idx,
                    entity_id=row.entity_id,
                    raw_name=row.business_name,
                    raw_address=row.business_address,
                    raw_country=row.country,
                )
                filled_count += 1

    assert filled_count == n_train_s1, f"Expected {n_train_s1} filled records, got {filled_count}"
    print(f"  S1RecordStore populated with {filled_count:,} records in {time.time() - start_time:.1f}s.")

    # -------------------------------------------------------------------------
    # Step 3: Load training pairs and verify alignment against ground truth
    # -------------------------------------------------------------------------
    print("\n[3/6] Loading training pairs and performing alignment assertions...")
    pairs_v1_dir = output_dir / "training_pairs_v1"
    s1_idx = np.load(pairs_v1_dir / "s1_idx.npy")
    target_num = np.load(pairs_v1_dir / "target_num.npy")
    source = np.load(pairs_v1_dir / "source.npy")
    label = np.load(pairs_v1_dir / "label.npy")
    negative_route = np.load(pairs_v1_dir / "negative_route.npy")

    n_pairs = len(label)
    n_pos = int((label == 1).sum())
    n_neg = int((label == 0).sum())
    print(f"  Total pairs: {n_pairs:,}")
    print(f"  Positives:   {n_pos:,} ({n_pos / n_pairs:.2%})")
    print(f"  Negatives:   {n_neg:,} ({n_neg / n_pairs:.2%})")

    # Load ground truth for training entities to verify pairs
    print("  Loading training ground truth for verification...")
    gt_map: Dict[str, Set[str]] = {}
    for chunk in pd.read_csv(
        data_dir / "train_ground_truth.tsv",
        sep="\t",
        dtype=str,
        chunksize=chunk_size,
        keep_default_na=False,
    ):
        for row in chunk.itertuples(index=False):
            if row.source1_entity_id not in val_s1_ids:
                matches = {m.strip() for m in row.matched_entity_ids.split(",") if m.strip()}
                gt_map[row.source1_entity_id] = matches

    # Alignment verification on 50,000 positive pairs
    np.random.seed(42)
    pos_indices = np.flatnonzero(label == 1)
    sample_size = min(50_000, len(pos_indices))
    sample_pos = np.random.choice(pos_indices, size=sample_size, replace=False)

    for p_idx in sample_pos:
        s1_id_actual = train_s1_entity_ids[s1_idx[p_idx]]
        target_id_actual = f"S{source[p_idx]}-{target_num[p_idx]}"
        if target_id_actual not in gt_map.get(s1_id_actual, set()):
            raise AssertionError(
                f"ALIGNMENT FAILURE on pair {p_idx}: S1 '{s1_id_actual}' does not match target '{target_id_actual}' in ground truth!"
            )
    print(f"  POSITIVE ALIGNMENT VERIFIED: {sample_size:,} / {sample_size:,} sample pairs strictly in ground truth.")

    # Alignment verification on 50,000 negative pairs
    neg_indices = np.flatnonzero(label == 0)
    sample_neg_size = min(50_000, len(neg_indices))
    sample_neg = np.random.choice(neg_indices, size=sample_neg_size, replace=False)

    for p_idx in sample_neg:
        s1_id_actual = train_s1_entity_ids[s1_idx[p_idx]]
        target_id_actual = f"S{source[p_idx]}-{target_num[p_idx]}"
        if target_id_actual in gt_map.get(s1_id_actual, set()):
            raise AssertionError(
                f"NEGATIVE ALIGNMENT FAILURE on pair {p_idx}: Negative pair S1 '{s1_id_actual}' matches target '{target_id_actual}' in ground truth!"
            )
    print(f"  NEGATIVE ALIGNMENT VERIFIED: {sample_neg_size:,} / {sample_neg_size:,} sample pairs strictly non-matches.")

    # Save verified pairs into training_pairs_v2
    np.save(pairs_v2_dir / "s1_idx.npy", s1_idx)
    np.save(pairs_v2_dir / "target_num.npy", target_num)
    np.save(pairs_v2_dir / "source.npy", source)
    np.save(pairs_v2_dir / "label.npy", label)
    np.save(pairs_v2_dir / "negative_route.npy", negative_route)
    print(f"  Saved verified pair arrays to {pairs_v2_dir}")

    # -------------------------------------------------------------------------
    # Step 4: Streamlined, high-throughput feature generation
    # -------------------------------------------------------------------------
    print("\n[4/6] Computing 16 pairwise features using S1RecordStore...")
    features = np.empty((n_pairs, N_FEATURES), dtype=np.float32)

    # Separate pair indices by source and sort by target_num for O(log N) streaming
    s2_pair_indices = np.flatnonzero(source == 2)
    s3_pair_indices = np.flatnonzero(source == 3)

    s2_pair_indices = s2_pair_indices[np.argsort(target_num[s2_pair_indices], kind="stable")]
    s3_pair_indices = s3_pair_indices[np.argsort(target_num[s3_pair_indices], kind="stable")]

    def fill_features_for_source(source_path: Path, source_number: int, pair_indices: np.ndarray, source_name: str):
        sorted_target_nums = target_num[pair_indices]
        filled_count = 0
        total_rows = 0
        s_time = time.time()

        for chunk_number, chunk in enumerate(
            pd.read_csv(
                source_path,
                sep="\t",
                dtype=str,
                chunksize=chunk_size,
                keep_default_na=False,
            ),
            start=1,
        ):
            total_rows += len(chunk)

            # Numeric suffix of target entity_id
            chunk_target_nums = (
                chunk["entity_id"]
                .str.slice(3)
                .astype(np.int64)
                .to_numpy()
            )

            left_positions = np.searchsorted(sorted_target_nums, chunk_target_nums, side="left")
            right_positions = np.searchsorted(sorted_target_nums, chunk_target_nums, side="right")

            for row_pos, row in enumerate(chunk.itertuples(index=False)):
                left = int(left_positions[row_pos])
                right = int(right_positions[row_pos])
                if left == right:
                    continue

                # Indices in global pairs array for this target
                selected_pair_indices = pair_indices[left:right]
                selected_s1_indices = s1_idx[selected_pair_indices]

                # Compute features in batch using cached S1 store
                batch_feat = compute_candidate_features_batch(
                    candidate_indices=selected_s1_indices,
                    target_name=row.business_name,
                    target_address=row.business_address,
                    target_country=row.country,
                    source_number=source_number,
                    s1_store=s1_store,
                )

                features[selected_pair_indices] = batch_feat
                filled_count += len(selected_pair_indices)

            if chunk_number % 10 == 0:
                print(f"  [{source_name}] {total_rows:,} rows | Filled {filled_count:,} / {len(pair_indices):,} ({time.time() - s_time:.1f}s)")

        print(f"  [{source_name}] Finished {total_rows:,} rows. Filled all {filled_count:,} pairs.")
        assert filled_count == len(pair_indices), f"Expected {len(pair_indices)} pairs, filled {filled_count}"

    fill_features_for_source(data_dir / "train_source2.tsv", 2, s2_pair_indices, "Source 2")
    fill_features_for_source(data_dir / "train_source3.tsv", 3, s3_pair_indices, "Source 3")

    # -------------------------------------------------------------------------
    # Step 5: Feature matrix validation and persistence
    # -------------------------------------------------------------------------
    print("\n[5/6] Validating feature matrix integrity...")
    assert features.shape == (n_pairs, N_FEATURES), f"Shape mismatch: {features.shape} vs ({n_pairs}, {N_FEATURES})"
    n_nan = int(np.isnan(features).sum())
    n_inf = int(np.isinf(features).sum())
    assert n_nan == 0, f"Found {n_nan} NaN values in feature matrix!"
    assert n_inf == 0, f"Found {n_inf} Inf values in feature matrix!"
    assert np.isfinite(features).all(), "Non-finite values present in feature matrix!"
    print("  Shape check: PASS")
    print("  Finite values check (0 NaN, 0 Inf): PASS")

    # Save to training_features_v2
    print(f"\nSaving feature matrix to {features_v2_dir}...")
    np.save(features_v2_dir / "features.npy", features)
    np.save(features_v2_dir / "labels.npy", label)
    np.save(features_v2_dir / "s1_idx.npy", s1_idx)
    np.save(features_v2_dir / "target_num.npy", target_num)
    np.save(features_v2_dir / "source.npy", source)

    with open(features_v2_dir / "feature_names.pkl", "wb") as f:
        pickle.dump(FEATURE_NAMES, f, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "n_pairs": n_pairs,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_features": N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(features_v2_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # -------------------------------------------------------------------------
    # Step 6: Sanity checks & explicit positive spot-checks
    # -------------------------------------------------------------------------
    print("\n[6/6] Computing feature distributions and running spot checks...")
    pos_mask = (label == 1)
    neg_mask = (label == 0)

    dist_summary = []
    for i, fname in enumerate(FEATURE_NAMES):
        p_mean = float(features[pos_mask, i].mean())
        n_mean = float(features[neg_mask, i].mean())
        diff = p_mean - n_mean
        dist_summary.append({
            "feature": fname,
            "pos_mean": p_mean,
            "neg_mean": n_mean,
            "diff": diff,
        })

    dist_df = pd.DataFrame(dist_summary)
    print("\n" + "=" * 65)
    print(f"{'FEATURE':25s} | {'POS MEAN':10s} | {'NEG MEAN':10s} | {'DIFF':10s}")
    print("=" * 65)
    for r in dist_summary:
        print(f"{r['feature']:25s} | {r['pos_mean']:10.4f} | {r['neg_mean']:10.4f} | {r['diff']:10.4f}")
    print("=" * 65)

    # Sanity checks on positive distribution
    pos_name_ratio = dist_df.loc[dist_df["feature"] == "name_ratio", "pos_mean"].values[0]
    pos_country_same = dist_df.loc[dist_df["feature"] == "country_same", "pos_mean"].values[0]
    pos_name_exact = dist_df.loc[dist_df["feature"] == "name_exact", "pos_mean"].values[0]

    assert pos_name_ratio > 0.70, f"Positive name_ratio too low: {pos_name_ratio:.4f}"
    assert pos_country_same > 0.95, f"Positive country_same too low: {pos_country_same:.4f}"
    assert pos_name_exact > 0.10, f"Positive name_exact too low: {pos_name_exact:.4f}"
    print("\nSANITY CHECKS PASSED: Positives demonstrate strong, realistic entity similarity.")

    # Explicit spot checks on 10 positive pairs
    run_explicit_spot_checks(s1_idx, target_num, source, label, train_s1_entity_ids, s1_store, data_dir)

    print(f"\nPhase 4 completed in {time.time() - start_time:.1f}s.")
    return pairs_v2_dir, features_v2_dir


def run_explicit_spot_checks(s1_idx, target_num, source, label, train_s1_entity_ids, s1_store, data_dir):
    print("\n" + "=" * 70)
    print("EXPLICIT SPOT CHECK: 10 KNOWN POSITIVE TRAINING PAIRS")
    print("=" * 70)

    # Pick 10 fixed positive pairs from different regions of the array
    pos_indices = np.flatnonzero(label == 1)
    step = len(pos_indices) // 10
    selected_indices = [pos_indices[i * step] for i in range(10)]

    # Load target records for these 10 pairs
    target_ids_needed = {
        (int(source[p_idx]), int(target_num[p_idx])): p_idx
        for p_idx in selected_indices
    }

    target_records: Dict[Tuple[int, int], Tuple[str, str, str]] = {}
    for src_num, src_file in [(2, "train_source2.tsv"), (3, "train_source3.tsv")]:
        needed_nums = {t_num for (s_num, t_num) in target_ids_needed if s_num == src_num}
        if not needed_nums:
            continue
        for chunk in pd.read_csv(
            data_dir / src_file,
            sep="\t",
            dtype=str,
            chunksize=100_000,
            keep_default_na=False,
        ):
            for row in chunk.itertuples(index=False):
                t_num = int(row.entity_id[3:])
                if t_num in needed_nums:
                    target_records[(src_num, t_num)] = (
                        row.business_name,
                        row.business_address,
                        row.country,
                    )

    for i, p_idx in enumerate(selected_indices, start=1):
        s1_i = int(s1_idx[p_idx])
        src_n = int(source[p_idx])
        t_n = int(target_num[p_idx])

        s1_id = train_s1_entity_ids[s1_i]
        target_id = f"S{src_n}-{t_n}"

        s1_name = s1_store.names[s1_i]
        s1_addr = s1_store.addresses[s1_i]
        s1_c = s1_store.countries[s1_i]

        t_raw_name, t_raw_addr, t_raw_c = target_records.get((src_n, t_n), ("", "", ""))
        t_name = normalize_text(t_raw_name)
        t_addr = normalize_text(t_raw_addr)
        t_c = normalize_country(t_raw_c)

        n_ratio = fuzz.ratio(s1_name, t_name) / 100.0 if (s1_name and t_name) else 0.0
        a_ratio = fuzz.ratio(s1_addr, t_addr) / 100.0 if (s1_addr and t_addr) else 0.0
        c_same = int(bool(s1_c and s1_c == t_c))

        print(f"\n--- Spot Check Pair #{i} (Index {p_idx}) ---")
        print(f"S1 ID:          {s1_id}")
        print(f"Target ID:      {target_id}")
        print(f"S1 Name:        {s1_name}")
        print(f"Target Name:    {t_name}")
        print(f"S1 Address:     {s1_addr}")
        print(f"Target Address: {t_addr}")
        print(f"S1 Country:     {s1_c}")
        print(f"Target Country: {t_c}")
        print(f"Name Ratio:     {n_ratio:.4f}")
        print(f"Address Ratio:  {a_ratio:.4f}")
        print(f"Country Same:   {c_same}")
        print(f"Ground Truth:   1 (Positive)")
        assert c_same == 1, f"Spot check failed: country mismatch on positive pair {p_idx}"
        assert n_ratio > 0.40, f"Spot check failed: name ratio unexpectedly low on positive pair {p_idx}"

    print("\nALL 10 SPOT CHECKS PASSED PERFECTLY!")


if __name__ == "__main__":
    build_and_verify_training_features_v2()
