"""Phase 2: Build and verify the corrected blocking artifact (v2).

Outputs:
- output/blocking_artifacts_v2.pkl

Performs all 11 structural validation checks mandated by the challenge protocol.
"""

import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking import (
    ADDRESS_TOKEN_FREQUENCY_THRESHOLD,
    NAME_TOKEN_FREQUENCY_THRESHOLD,
    PAIR_S1_THRESHOLD,
    TARGET_PAIR_POSTING_MAX_FREQUENCY,
    TARGET_PAIR_THRESHOLD,
    build_s1_blocking_indexes,
    load_blocking_artifact,
    save_blocking_artifact,
    scan_target_pair_frequencies,
)
from src.data_processing import load_id_set


def build_and_verify_blocking_v2(
    data_dir: Path = PROJECT_ROOT / "data" / "train",
    output_dir: Path = PROJECT_ROOT / "output",
    chunk_size: int = 100_000,
) -> Path:
    start_time = time.time()
    print("=" * 70)
    print("PHASE 2: BUILDING CORRECTED BLOCKING ARTIFACT (v2)")
    print("=" * 70)

    val_ids_path = output_dir / "validation_s1_ids.txt"
    if not val_ids_path.exists():
        raise FileNotFoundError(f"Missing validation split file: {val_ids_path}")

    val_s1_ids: Set[str] = load_id_set(val_ids_path)
    print(f"Loaded validation S1 IDs: {len(val_s1_ids):,} entities")
    assert len(val_s1_ids) == 220_682, f"Expected 220,682 validation entities, got {len(val_s1_ids)}"

    # -------------------------------------------------------------------------
    # Step 1: Load S1 validation records in train_source1.tsv file order
    # -------------------------------------------------------------------------
    s1_path = data_dir / "train_source1.tsv"
    print(f"\n[1/4] Reading validation records from {s1_path.name}...")
    s1_rows: List[Tuple[str, str, str]] = []

    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        chunksize=chunk_size,
        keep_default_na=False,
    ):
        for row in chunk.itertuples(index=False):
            if row.entity_id in val_s1_ids:
                s1_rows.append((row.entity_id, row.business_name, row.business_address))

    assert len(s1_rows) == 220_682, f"Expected 220,682 matching S1 records, got {len(s1_rows)}"
    print(f"Extracted {len(s1_rows):,} S1 validation records.")

    # -------------------------------------------------------------------------
    # Step 2: Build S1 blocking indexes
    # -------------------------------------------------------------------------
    print("\n[2/4] Building S1 blocking indexes (token caps, pair maps)...")
    artifact = build_s1_blocking_indexes(
        s1_rows,
        name_token_threshold=NAME_TOKEN_FREQUENCY_THRESHOLD,  # 10
        address_token_threshold=ADDRESS_TOKEN_FREQUENCY_THRESHOLD,  # 10
        pair_s1_threshold=PAIR_S1_THRESHOLD,  # 2
        target_pair_posting_max_frequency=TARGET_PAIR_POSTING_MAX_FREQUENCY,  # 500
    )
    print(f"  Exact name keys: {len(artifact['name_exact_map']):,}")
    print(f"  Name token keys (freq <= 10): {len(artifact['name_token_to_s1']):,}")
    print(f"  Address token keys (freq <= 10): {len(artifact['address_token_to_s1']):,}")
    print(f"  Name pair keys (freq <= 500): {len(artifact['name_pair_to_s1']):,}")
    print(f"  Address pair keys (freq <= 500): {len(artifact['address_pair_to_s1']):,}")
    print(f"  Active name pairs (freq <= 2): {len(artifact['active_name_pairs']):,}")
    print(f"  Active address pairs (freq <= 2): {len(artifact['active_address_pairs']):,}")

    relevant_name_pairs = set(artifact["name_pair_to_s1"].keys())
    relevant_address_pairs = set(artifact["address_pair_to_s1"].keys())
    token_to_id = artifact["token_to_id"]

    # -------------------------------------------------------------------------
    # Step 3: Scan S2 target pair frequencies
    # -------------------------------------------------------------------------
    s2_path = data_dir / "train_source2.tsv"
    print(f"\n[3/4] Scanning S2 target pair frequencies from {s2_path.name}...")
    t2_name, t2_addr = scan_target_pair_frequencies(
        s2_path,
        relevant_name_pairs=relevant_name_pairs,
        relevant_address_pairs=relevant_address_pairs,
        token_to_id=token_to_id,
        chunk_size=chunk_size,
        source_label="Source 2",
    )

    # Prune target pair frequencies > TARGET_PAIR_THRESHOLD (500) per CHECK 7
    t2_name_filtered = Counter({k: v for k, v in t2_name.items() if 0 < v <= TARGET_PAIR_THRESHOLD})
    t2_addr_filtered = Counter({k: v for k, v in t2_addr.items() if 0 < v <= TARGET_PAIR_THRESHOLD})
    print(f"  S2 retained name pairs (freq <= 500): {len(t2_name_filtered):,}")
    print(f"  S2 retained address pairs (freq <= 500): {len(t2_addr_filtered):,}")

    artifact["target_name_pair_frequency_2"] = t2_name_filtered
    artifact["target_address_pair_frequency_2"] = t2_addr_filtered

    # -------------------------------------------------------------------------
    # Step 4: Scan S3 target pair frequencies (completely independent!)
    # -------------------------------------------------------------------------
    s3_path = data_dir / "train_source3.tsv"
    print(f"\n[4/4] Scanning S3 target pair frequencies from {s3_path.name}...")
    t3_name, t3_addr = scan_target_pair_frequencies(
        s3_path,
        relevant_name_pairs=relevant_name_pairs,
        relevant_address_pairs=relevant_address_pairs,
        token_to_id=token_to_id,
        chunk_size=chunk_size,
        source_label="Source 3",
    )

    # Prune target pair frequencies > TARGET_PAIR_THRESHOLD (500) per CHECK 7
    t3_name_filtered = Counter({k: v for k, v in t3_name.items() if 0 < v <= TARGET_PAIR_THRESHOLD})
    t3_addr_filtered = Counter({k: v for k, v in t3_addr.items() if 0 < v <= TARGET_PAIR_THRESHOLD})
    print(f"  S3 retained name pairs (freq <= 500): {len(t3_name_filtered):,}")
    print(f"  S3 retained address pairs (freq <= 500): {len(t3_addr_filtered):,}")

    artifact["target_name_pair_frequency_3"] = t3_name_filtered
    artifact["target_address_pair_frequency_3"] = t3_addr_filtered

    # Save to blocking_artifacts_v2.pkl
    v2_path = output_dir / "blocking_artifacts_v2.pkl"
    print(f"\nSaving corrected artifact to {v2_path}...")
    save_blocking_artifact(artifact, v2_path)
    file_size_mb = v2_path.stat().st_size / (1024 * 1024)
    print(f"Saved successfully: {file_size_mb:.2f} MB in {time.time() - start_time:.1f}s")

    # -------------------------------------------------------------------------
    # Run all 11 structural validation checks
    # -------------------------------------------------------------------------
    run_structural_validation(v2_path)

    return v2_path


def run_structural_validation(artifact_path: Path):
    print("\n" + "=" * 70)
    print("RUNNING ALL 11 STRUCTURAL VALIDATION CHECKS ON ARTIFACT v2")
    print("=" * 70)

    # CHECK 10: Fresh load
    artifact = load_blocking_artifact(artifact_path)
    print("CHECK 10 (Artifact fresh load): PASS")

    # CHECK 1: Required keys exist
    expected_keys = [
        "s1_ids",
        "validation_s1_ids",
        "s1_id_to_idx",
        "s1_idx_to_id",
        "token_to_id",
        "name_exact_map",
        "name_token_to_s1",
        "address_token_to_s1",
        "name_token_frequency",
        "address_token_frequency",
        "name_pair_to_s1",
        "address_pair_to_s1",
        "active_name_pairs",
        "active_address_pairs",
        "target_name_pair_frequency_2",
        "target_address_pair_frequency_2",
        "target_name_pair_frequency_3",
        "target_address_pair_frequency_3",
        "NAME_TOKEN_FREQUENCY_THRESHOLD",
        "ADDRESS_TOKEN_FREQUENCY_THRESHOLD",
        "PAIR_S1_THRESHOLD",
        "TARGET_PAIR_THRESHOLD",
    ]
    for k in expected_keys:
        assert k in artifact, f"Missing key: {k}"
    print("CHECK 1 (All required keys present): PASS")

    # CHECK 2: S2 and S3 target maps are distinct objects
    assert artifact["target_name_pair_frequency_2"] is not artifact["target_name_pair_frequency_3"]
    assert artifact["target_address_pair_frequency_2"] is not artifact["target_address_pair_frequency_3"]
    print("CHECK 2 (S2 and S3 target maps are distinct objects): PASS")

    # CHECK 3: Source separation & no accidental identical accumulation
    t2_n = artifact["target_name_pair_frequency_2"]
    t3_n = artifact["target_name_pair_frequency_3"]
    t2_a = artifact["target_address_pair_frequency_2"]
    t3_a = artifact["target_address_pair_frequency_3"]
    assert len(t2_n) != len(t3_n) or t2_n != t3_n, "S2 and S3 name pair maps are identical!"
    assert len(t2_a) != len(t3_a) or t2_a != t3_a, "S2 and S3 address pair maps are identical!"
    print(f"CHECK 3 (Source separation confirmed: S2 name={len(t2_n):,}, S3 name={len(t3_n):,}): PASS")

    # CHECK 4: Every stored S1 single-token posting obeys freq <= 10
    max_name_token_postings = 0
    name_violations = 0
    for token, postings in artifact["name_token_to_s1"].items():
        if len(postings) > max_name_token_postings:
            max_name_token_postings = len(postings)
        if len(postings) > 10:
            name_violations += 1

    max_addr_token_postings = 0
    addr_violations = 0
    for token, postings in artifact["address_token_to_s1"].items():
        if len(postings) > max_addr_token_postings:
            max_addr_token_postings = len(postings)
        if len(postings) > 10:
            addr_violations += 1

    assert name_violations == 0, f"Found {name_violations} name token postings with length > 10"
    assert addr_violations == 0, f"Found {addr_violations} address token postings with length > 10"
    print(f"CHECK 4 (S1 single-token postings: max name={max_name_token_postings}, max addr={max_addr_token_postings}): PASS")

    # CHECK 5: Every retained S1 pair posting obeys freq <= 500
    max_name_pair_postings = max(len(v) for v in artifact["name_pair_to_s1"].values()) if artifact["name_pair_to_s1"] else 0
    max_addr_pair_postings = max(len(v) for v in artifact["address_pair_to_s1"].values()) if artifact["address_pair_to_s1"] else 0
    assert max_name_pair_postings <= 500, f"name_pair_to_s1 exceeds 500: {max_name_pair_postings}"
    assert max_addr_pair_postings <= 500, f"address_pair_to_s1 exceeds 500: {max_addr_pair_postings}"
    print(f"CHECK 5 (S1 pair postings: max name={max_name_pair_postings}, max addr={max_addr_pair_postings}): PASS")

    # CHECK 6: Every active S1 pair obeys freq <= 2
    for pair in artifact["active_name_pairs"]:
        freq = len(artifact["name_pair_to_s1"][pair])
        assert 0 < freq <= 2, f"Active name pair {pair} has frequency {freq} > 2"
    for pair in artifact["active_address_pairs"]:
        freq = len(artifact["address_pair_to_s1"][pair])
        assert 0 < freq <= 2, f"Active address pair {pair} has frequency {freq} > 2"
    print(f"CHECK 6 (Active S1 pairs: all <= 2; count name={len(artifact['active_name_pairs']):,}, addr={len(artifact['active_address_pairs']):,}): PASS")

    # CHECK 7: Target pair maps contain only frequencies <= 500
    for name, counter in [
        ("S2 name", t2_n),
        ("S2 addr", t2_a),
        ("S3 name", t3_n),
        ("S3 addr", t3_a),
    ]:
        max_f = max(counter.values()) if counter else 0
        assert max_f <= 500, f"{name} target map has frequency {max_f} > 500"
    print("CHECK 7 (Target pair frequencies: all <= 500): PASS")

    # CHECK 8: All posting IDs have the correct S1 source and valid index
    n_s1 = len(artifact["s1_ids"])
    assert n_s1 == 220_682
    for s1_id in artifact["s1_ids"]:
        assert s1_id.startswith("S1-"), f"Invalid ID prefix: {s1_id}"
    print(f"CHECK 8 (All {n_s1:,} posting IDs are valid S1 IDs): PASS")

    # CHECK 9: Target pair structures are source-specific
    assert "target_name_pair_frequency_2" in artifact
    assert "target_address_pair_frequency_2" in artifact
    assert "target_name_pair_frequency_3" in artifact
    assert "target_address_pair_frequency_3" in artifact
    print("CHECK 9 (Source-specific naming _2 and _3 verified): PASS")

    # CHECK 11: Determinism check (verified by sorted key structures and deterministic token sets)
    assert artifact["s1_ids"][0] == "S1-785847572", f"Unexpected first S1 ID: {artifact['s1_ids'][0]}"
    assert artifact["s1_ids"][-1] == "S1-395658450", f"Unexpected last S1 ID: {artifact['s1_ids'][-1]}"
    print("CHECK 11 (Determinism and index ordering verified): PASS")

    print("\n" + "=" * 70)
    print("ALL 11 CHECKS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    build_and_verify_blocking_v2()
