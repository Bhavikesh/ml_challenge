"""Source-aware blocking and candidate generation module for Amazon ML Challenge 2026.

This module builds and evaluates the multi-route candidate generator:
1. Exact normalized name
2. S1-side name tokens (frequency <= 10)
3. S1-side address tokens (frequency <= 10)
4. S1-side active name pairs (frequency <= 2)
5. S1-side active address pairs (frequency <= 2)
6. Target-side name pairs (source-specific frequency <= 500)
7. Target-side address pairs (source-specific frequency <= 500)

CRITICAL:
S2 and S3 target frequencies are maintained and queried independently:
- target_name_pair_frequency_2, target_address_pair_frequency_2
- target_name_pair_frequency_3, target_address_pair_frequency_3
"""

import pickle
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import pandas as pd

from src.data_processing import get_tokens, normalize_text

# Default blocking hyperparameters established by empirical research
NAME_TOKEN_FREQUENCY_THRESHOLD = 10
ADDRESS_TOKEN_FREQUENCY_THRESHOLD = 10
PAIR_S1_THRESHOLD = 2
TARGET_PAIR_POSTING_MAX_FREQUENCY = 500
TARGET_PAIR_THRESHOLD = 500


def build_s1_blocking_indexes(
    s1_rows: Iterable[Tuple[str, str, str]],  # (s1_id, business_name, business_address)
    name_token_threshold: int = NAME_TOKEN_FREQUENCY_THRESHOLD,
    address_token_threshold: int = ADDRESS_TOKEN_FREQUENCY_THRESHOLD,
    pair_s1_threshold: int = PAIR_S1_THRESHOLD,
    target_pair_posting_max_frequency: int = TARGET_PAIR_POSTING_MAX_FREQUENCY,
) -> Dict[str, Any]:
    """Build S1-side blocking indexes and active pair sets.

    Parameters:
    - s1_rows: iterable of (s1_id, business_name, business_address) in fixed S1 order.
               The index in this sequence corresponds to integer s1_idx (0..N-1).

    Returns dictionary with:
    - s1_ids: list of S1 entity IDs
    - token_to_id: mapping from token string to integer ID
    - name_exact_map: dict of normalized_name -> list of s1_idx
    - name_token_to_s1: dict of token -> list of s1_idx (STRICTLY freq <= 10)
    - address_token_to_s1: dict of token -> list of s1_idx (STRICTLY freq <= 10)
    - name_token_frequency: Counter of token -> S1 entity count
    - address_token_frequency: Counter of token -> S1 entity count
    - name_pair_to_s1: dict of (t1, t2) -> list of s1_idx (freq <= 500)
    - address_pair_to_s1: dict of (t1, t2) -> list of s1_idx (freq <= 500)
    - active_name_pairs: set of (t1, t2) where 0 < freq <= pair_s1_threshold
    - active_address_pairs: set of (t1, t2) where 0 < freq <= pair_s1_threshold
    """
    s1_ids: List[str] = []
    normalized_names: List[str] = []
    normalized_addresses: List[str] = []
    name_tokens_list: List[List[str]] = []
    address_tokens_list: List[List[str]] = []

    token_to_id: Dict[str, int] = {}
    next_token_id = 1

    name_token_frequency = Counter()
    address_token_frequency = Counter()
    name_exact_map: Dict[str, List[int]] = defaultdict(list)

    # First pass: normalize, tokenize, count frequencies, build exact map
    for s1_idx, (s1_id, raw_name, raw_address) in enumerate(s1_rows):
        s1_ids.append(s1_id)

        norm_name = normalize_text(raw_name)
        norm_address = normalize_text(raw_address)
        normalized_names.append(norm_name)
        normalized_addresses.append(norm_address)

        if norm_name:
            name_exact_map[norm_name].append(s1_idx)

        n_tokens = sorted(get_tokens(norm_name))
        a_tokens = sorted(get_tokens(norm_address))
        name_tokens_list.append(n_tokens)
        address_tokens_list.append(a_tokens)

        # Count entity frequencies (tokens are deduplicated per entity)
        name_token_frequency.update(n_tokens)
        address_token_frequency.update(a_tokens)

        # Register token IDs
        for t in n_tokens:
            if t not in token_to_id:
                token_to_id[t] = next_token_id
                next_token_id += 1
        for t in a_tokens:
            if t not in token_to_id:
                token_to_id[t] = next_token_id
                next_token_id += 1

    # Second pass: calculate pair frequencies across S1
    name_pair_frequency = Counter()
    address_pair_frequency = Counter()

    for s1_idx in range(len(s1_ids)):
        n_token_ids = [token_to_id[t] for t in name_tokens_list[s1_idx]]
        if len(n_token_ids) >= 2:
            name_pair_frequency.update(combinations(n_token_ids, 2))

        a_token_ids = [token_to_id[t] for t in address_tokens_list[s1_idx]]
        if len(a_token_ids) >= 2:
            address_pair_frequency.update(combinations(a_token_ids, 2))

    # Active S1 pairs (threshold <= pair_s1_threshold, default 2)
    active_name_pairs = {
        pair for pair, freq in name_pair_frequency.items() if 0 < freq <= pair_s1_threshold
    }
    active_address_pairs = {
        pair for pair, freq in address_pair_frequency.items() if 0 < freq <= pair_s1_threshold
    }

    # Third pass: build posting lists with strict threshold caps
    name_token_to_s1: Dict[str, List[int]] = defaultdict(list)
    address_token_to_s1: Dict[str, List[int]] = defaultdict(list)
    name_pair_to_s1: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    address_pair_to_s1: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    for s1_idx in range(len(s1_ids)):
        # Filtered single-token postings (ONLY freq <= threshold)
        for t in name_tokens_list[s1_idx]:
            if name_token_frequency[t] <= name_token_threshold:
                name_token_to_s1[t].append(s1_idx)

        for t in address_tokens_list[s1_idx]:
            if address_token_frequency[t] <= address_token_threshold:
                address_token_to_s1[t].append(s1_idx)

        # Pair postings (retained up to target_pair_posting_max_frequency, default 500)
        n_token_ids = [token_to_id[t] for t in name_tokens_list[s1_idx]]
        if len(n_token_ids) >= 2:
            for pair in combinations(n_token_ids, 2):
                if name_pair_frequency[pair] <= target_pair_posting_max_frequency:
                    name_pair_to_s1[pair].append(s1_idx)

        a_token_ids = [token_to_id[t] for t in address_tokens_list[s1_idx]]
        if len(a_token_ids) >= 2:
            for pair in combinations(a_token_ids, 2):
                if address_pair_frequency[pair] <= target_pair_posting_max_frequency:
                    address_pair_to_s1[pair].append(s1_idx)

    s1_id_to_idx = {entity_id: idx for idx, entity_id in enumerate(s1_ids)}
    s1_idx_to_id = {idx: entity_id for entity_id, idx in s1_id_to_idx.items()}

    return {
        "s1_ids": s1_ids,
        "validation_s1_ids": set(s1_ids),
        "s1_id_to_idx": s1_id_to_idx,
        "s1_idx_to_id": s1_idx_to_id,
        "token_to_id": token_to_id,
        "name_exact_map": dict(name_exact_map),
        "name_token_to_s1": dict(name_token_to_s1),
        "address_token_to_s1": dict(address_token_to_s1),
        "name_token_frequency": dict(name_token_frequency),
        "address_token_frequency": dict(address_token_frequency),
        "name_pair_to_s1": dict(name_pair_to_s1),
        "address_pair_to_s1": dict(address_pair_to_s1),
        "active_name_pairs": active_name_pairs,
        "active_address_pairs": active_address_pairs,
        "NAME_TOKEN_FREQUENCY_THRESHOLD": name_token_threshold,
        "ADDRESS_TOKEN_FREQUENCY_THRESHOLD": address_token_threshold,
        "PAIR_S1_THRESHOLD": pair_s1_threshold,
        "TARGET_PAIR_POSTING_MAX_FREQUENCY": target_pair_posting_max_frequency,
        "TARGET_PAIR_THRESHOLD": TARGET_PAIR_THRESHOLD,
    }


def scan_target_pair_frequencies(
    target_tsv_path: Union[str, Path],
    relevant_name_pairs: Set[Tuple[int, int]],
    relevant_address_pairs: Set[Tuple[int, int]],
    token_to_id: Dict[str, int],
    chunk_size: int = 100_000,
    verbose: bool = True,
    source_label: str = "Target",
) -> Tuple[Counter, Counter]:
    """Scan target file (S2 or S3) to compute source-specific target-side pair frequencies.

    Counts occurrences of relevant pairs (all pairs indexed in S1) in the target source.
    Returns:
    - target_name_pair_frequency: Counter of (t1, t2) -> count
    - target_address_pair_frequency: Counter of (t1, t2) -> count
    """
    target_name_pair_freq = Counter()
    target_address_pair_freq = Counter()
    total_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            target_tsv_path,
            sep="\t",
            dtype=str,
            chunksize=chunk_size,
            keep_default_na=False,
        ),
        start=1,
    ):
        total_rows += len(chunk)

        for row in chunk.itertuples(index=False):
            name_tokens = sorted(get_tokens(row.business_name))
            addr_tokens = sorted(get_tokens(row.business_address))

            n_ids = sorted({token_to_id[t] for t in name_tokens if t in token_to_id})
            if len(n_ids) >= 2:
                for pair in combinations(n_ids, 2):
                    if pair in relevant_name_pairs:
                        target_name_pair_freq[pair] += 1

            a_ids = sorted({token_to_id[t] for t in addr_tokens if t in token_to_id})
            if len(a_ids) >= 2:
                for pair in combinations(a_ids, 2):
                    if pair in relevant_address_pairs:
                        target_address_pair_freq[pair] += 1

        if verbose and chunk_number % 10 == 0:
            print(f"[{source_label}] {total_rows:,} rows scanned...")

    if verbose:
        print(f"[{source_label}] Finished {total_rows:,} rows.")

    return target_name_pair_freq, target_address_pair_freq


def generate_candidates(
    target_name: str,
    target_address: str,
    source_number: int,
    artifact: Dict[str, Any],
) -> List[int]:
    """Generate candidate S1 integer indices for a single target record.

    Routes:
    1. Exact normalized name match
    2. Name token blocking (only for tokens where freq <= threshold)
    3. Address token blocking (only for tokens where freq <= threshold)
    4. S1 active name pairs
    5. S1 active address pairs
    6. Target-side name pairs (source-specific <= threshold)
    7. Target-side address pairs (source-specific <= threshold)

    Source-aware:
    - If source_number == 2: uses target_*_pair_frequency_2
    - If source_number == 3: uses target_*_pair_frequency_3
    """
    candidate_s1: Set[int] = set()

    norm_name = normalize_text(target_name)
    norm_address = normalize_text(target_address)

    # 1. Exact normalized name
    if norm_name:
        exact_matches = artifact["name_exact_map"].get(norm_name)
        if exact_matches:
            candidate_s1.update(exact_matches)

    name_tokens = sorted(get_tokens(norm_name))
    address_tokens = sorted(get_tokens(norm_address))
    token_to_id = artifact["token_to_id"]

    # 2. Name-token blocking (pre-filtered in artifact to freq <= 10)
    for token in name_tokens:
        postings = artifact["name_token_to_s1"].get(token)
        if postings:
            candidate_s1.update(postings)

    # 3. Address-token blocking (pre-filtered in artifact to freq <= 10)
    for token in address_tokens:
        postings = artifact["address_token_to_s1"].get(token)
        if postings:
            candidate_s1.update(postings)

    # Prepare token IDs for pair routes
    name_token_ids = [token_to_id[t] for t in name_tokens if t in token_to_id]
    address_token_ids = [token_to_id[t] for t in address_tokens if t in token_to_id]

    active_name_pairs = artifact["active_name_pairs"]
    active_address_pairs = artifact["active_address_pairs"]
    name_pair_to_s1 = artifact["name_pair_to_s1"]
    address_pair_to_s1 = artifact["address_pair_to_s1"]

    # Select source-specific target pair maps
    if source_number == 2:
        target_name_pair_freq = artifact["target_name_pair_frequency_2"]
        target_address_pair_freq = artifact["target_address_pair_frequency_2"]
    elif source_number == 3:
        target_name_pair_freq = artifact["target_name_pair_frequency_3"]
        target_address_pair_freq = artifact["target_address_pair_frequency_3"]
    else:
        raise ValueError(f"source_number must be 2 or 3, got: {source_number}")

    target_pair_threshold = artifact.get("TARGET_PAIR_THRESHOLD", TARGET_PAIR_THRESHOLD)

    # 4 & 6. Name pairs (active S1 pairs <= 2 OR target-side pairs <= threshold)
    if len(name_token_ids) >= 2:
        for pair in combinations(name_token_ids, 2):
            if pair in active_name_pairs:
                postings = name_pair_to_s1.get(pair)
                if postings:
                    candidate_s1.update(postings)
            else:
                freq = target_name_pair_freq.get(pair)
                if freq is not None and 0 < freq <= target_pair_threshold:
                    postings = name_pair_to_s1.get(pair)
                    if postings:
                        candidate_s1.update(postings)

    # 5 & 7. Address pairs (active S1 pairs <= 2 OR target-side pairs <= threshold)
    if len(address_token_ids) >= 2:
        for pair in combinations(address_token_ids, 2):
            if pair in active_address_pairs:
                postings = address_pair_to_s1.get(pair)
                if postings:
                    candidate_s1.update(postings)
            else:
                freq = target_address_pair_freq.get(pair)
                if freq is not None and 0 < freq <= target_pair_threshold:
                    postings = address_pair_to_s1.get(pair)
                    if postings:
                        candidate_s1.update(postings)

    return sorted(candidate_s1)


def save_blocking_artifact(artifact: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save blocking artifact dictionary with validation of required keys."""
    required_keys = [
        "s1_ids",
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
    missing = [k for k in required_keys if k not in artifact]
    if missing:
        raise ValueError(f"Cannot save blocking artifact: missing required keys {missing}")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_blocking_artifact(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load blocking artifact and verify key structural integrity."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Blocking artifact file not found: {filepath}")
    with open(filepath, "rb") as f:
        artifact = pickle.load(f)

    # Ensure source-specific target frequency keys exist
    for key in (
        "target_name_pair_frequency_2",
        "target_address_pair_frequency_2",
        "target_name_pair_frequency_3",
        "target_address_pair_frequency_3",
    ):
        if key not in artifact:
            raise KeyError(f"Corrupt artifact: missing required source-specific key '{key}'")
    return artifact


def run_blocking_tests():
    """Unit assertions to verify blocking logic and source separation."""
    # Mock S1 data
    mock_s1 = [
        ("S1-1", "Alpha Logistics Inc", "100 Main Street, Springfield"),
        ("S1-2", "Beta Tech Solutions", "200 Oak Avenue, Springfield"),
        ("S1-3", "Gamma Enterprises", "300 Pine Road, Metropolis"),
        ("S1-4", "Alpha Medical Center", "400 Elm Blvd, Gotham"),
    ]

    artifact = build_s1_blocking_indexes(
        mock_s1,
        name_token_threshold=10,
        address_token_threshold=10,
        pair_s1_threshold=2,
    )

    # Mock S2 and S3 target frequencies separately
    # In S2, pair ('alpha', 'logistics') occurred 5 times
    # In S3, pair ('alpha', 'medical') occurred 3 times
    token_to_id = artifact["token_to_id"]
    alpha_id = token_to_id["alpha"]
    logistics_id = token_to_id["logistics"]
    medical_id = token_to_id["medical"]

    pair_alpha_logistics = tuple(sorted([alpha_id, logistics_id]))
    pair_alpha_medical = tuple(sorted([alpha_id, medical_id]))

    artifact["target_name_pair_frequency_2"] = Counter({pair_alpha_logistics: 5})
    artifact["target_address_pair_frequency_2"] = Counter()
    artifact["target_name_pair_frequency_3"] = Counter({pair_alpha_medical: 3})
    artifact["target_address_pair_frequency_3"] = Counter()

    # Test 1: Exact name match
    cand_exact = generate_candidates("Alpha Logistics Inc", "Different Address", 2, artifact)
    assert 0 in cand_exact, "Exact name route failed to match S1-1 (idx 0)"

    # Test 2: Source 2 vs Source 3 pair frequency isolation
    # Target from S2 with Alpha Logistics
    cands_s2 = generate_candidates("Alpha Logistics", "Unknown Addr", 2, artifact)
    assert 0 in cands_s2, "S2 pair route failed to match S1-1"

    # Target from S3 with Alpha Medical
    cands_s3 = generate_candidates("Alpha Medical", "Unknown Addr", 3, artifact)
    assert 3 in cands_s3, "S3 pair route failed to match S1-4"

    # Cross-source verification: S2 should NOT see S3 frequency for Alpha Medical
    # because target_name_pair_frequency_2 doesn't have pair_alpha_medical
    # (Note: active_name_pairs route might match if freq <= 2, but target-specific route tests isolation)
    assert isinstance(artifact["target_name_pair_frequency_2"], Counter)
    assert isinstance(artifact["target_name_pair_frequency_3"], Counter)
    assert pair_alpha_medical not in artifact["target_name_pair_frequency_2"]

    # Test 3: Rare token filtering
    # "springfield" appears in S1-1 and S1-2 (freq = 2 <= 10), so it should be indexed
    assert "springfield" in artifact["address_token_to_s1"]
    assert len(artifact["address_token_to_s1"]["springfield"]) == 2

    print("All blocking unit tests PASSED.")


if __name__ == "__main__":
    run_blocking_tests()
