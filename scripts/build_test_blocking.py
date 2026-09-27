"""Build a blocking artifact from TEST S1 entities.

This is required before running run_test_inference.py.
Analogous to build_blocking_v2.py but uses test_source1/2/3.tsv.

Outputs:
  output/blocking_artifacts_test.pkl

Estimated run time: ~20-40 minutes depending on machine.
"""

import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

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
    save_blocking_artifact,
    scan_target_pair_frequencies,
)

OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR   = PROJECT_ROOT / "data" / "test"
CHUNK_SIZE = 100_000


def main():
    start = time.time()
    print("=" * 70)
    print("BUILD TEST BLOCKING ARTIFACT")
    print("=" * 70)

    # ── 1. Load all test S1 records ──────────────────────────────────────────
    s1_path = DATA_DIR / "test_source1.tsv"
    print(f"\n[1/3] Loading test S1 records from {s1_path.name} …")
    s1_rows: List[Tuple[str, str, str]] = []
    for chunk in pd.read_csv(s1_path, sep="\t", dtype=str,
                              chunksize=CHUNK_SIZE, keep_default_na=False):
        for row in chunk.itertuples(index=False):
            s1_rows.append((row.entity_id, row.business_name, row.business_address))

    print(f"  Loaded {len(s1_rows):,} test S1 records.")

    # ── 2. Build S1 blocking indexes ─────────────────────────────────────────
    print("\n[2/3] Building S1 blocking indexes …")
    artifact = build_s1_blocking_indexes(
        s1_rows,
        name_token_threshold=NAME_TOKEN_FREQUENCY_THRESHOLD,
        address_token_threshold=ADDRESS_TOKEN_FREQUENCY_THRESHOLD,
        pair_s1_threshold=PAIR_S1_THRESHOLD,
        target_pair_posting_max_frequency=TARGET_PAIR_POSTING_MAX_FREQUENCY,
    )
    print(f"  S1 entities indexed: {len(artifact['s1_ids']):,}")
    print(f"  Exact name keys:     {len(artifact['name_exact_map']):,}")
    print(f"  Name token keys:     {len(artifact['name_token_to_s1']):,}")
    print(f"  Address token keys:  {len(artifact['address_token_to_s1']):,}")
    print(f"  Name pair keys:      {len(artifact['name_pair_to_s1']):,}")
    print(f"  Address pair keys:   {len(artifact['address_pair_to_s1']):,}")

    relevant_name_pairs    = set(artifact["name_pair_to_s1"].keys())
    relevant_address_pairs = set(artifact["address_pair_to_s1"].keys())
    token_to_id            = artifact["token_to_id"]

    # ── 3. Scan target pair frequencies for S2 and S3 ───────────────────────
    for src_num, fname in [(2, "test_source2.tsv"), (3, "test_source3.tsv")]:
        src_path = DATA_DIR / fname
        print(f"\n[3/3] Scanning target pair frequencies from {fname} …")
        t_name, t_addr = scan_target_pair_frequencies(
            src_path,
            relevant_name_pairs=relevant_name_pairs,
            relevant_address_pairs=relevant_address_pairs,
            token_to_id=token_to_id,
            chunk_size=CHUNK_SIZE,
            source_label=f"Source {src_num}",
        )
        t_name_f = Counter({k: v for k, v in t_name.items() if 0 < v <= TARGET_PAIR_THRESHOLD})
        t_addr_f = Counter({k: v for k, v in t_addr.items() if 0 < v <= TARGET_PAIR_THRESHOLD})
        print(f"  Retained name pairs (<=500): {len(t_name_f):,}")
        print(f"  Retained addr pairs (<=500): {len(t_addr_f):,}")
        artifact[f"target_name_pair_frequency_{src_num}"]    = t_name_f
        artifact[f"target_address_pair_frequency_{src_num}"] = t_addr_f

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / "blocking_artifacts_test.pkl"
    print(f"\nSaving test blocking artifact to {out_path} …")
    save_blocking_artifact(artifact, out_path)
    mb = out_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved: {mb:.1f} MB  in {time.time()-start:.0f}s")
    print(f"   S1 entities: {len(artifact['s1_ids']):,}")


if __name__ == "__main__":
    main()
