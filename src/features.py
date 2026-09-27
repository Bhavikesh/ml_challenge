"""Synchronized pair feature extraction module for Amazon ML Challenge 2026.

Features (16 total):
0. name_exact
1. name_ratio
2. name_token_ratio
3. name_token_jaccard
4. name_length_ratio
5. name_digit_overlap
6. address_exact
7. address_ratio
8. address_token_ratio
9. address_token_jaccard
10. address_length_ratio
11. address_digit_overlap
12. country_same
13. name_present_both
14. address_present_both
15. source_is_s3
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import numpy as np
from rapidfuzz import fuzz

# Ensure package imports work regardless of execution context
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_processing import (
    get_digit_tokens,
    get_tokens,
    normalize_country,
    normalize_text,
)

FEATURE_NAMES = [
    "name_exact",
    "name_ratio",
    "name_token_ratio",
    "name_token_jaccard",
    "name_length_ratio",
    "name_digit_overlap",
    "address_exact",
    "address_ratio",
    "address_token_ratio",
    "address_token_jaccard",
    "address_length_ratio",
    "address_digit_overlap",
    "country_same",
    "name_present_both",
    "address_present_both",
    "source_is_s3",
]

N_FEATURES = len(FEATURE_NAMES)  # 16


def safe_length_ratio(a: str, b: str) -> float:
    """Compute ratio of shorter string length to longer string length."""
    if not a or not b:
        return 0.0
    len_a, len_b = len(a), len(b)
    return min(len_a, len_b) / max(len_a, len_b)


def token_jaccard(a_tokens: Set[str], b_tokens: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not a_tokens and not b_tokens:
        return 0.0
    union_len = len(a_tokens | b_tokens)
    if union_len == 0:
        return 0.0
    return len(a_tokens & b_tokens) / union_len


class S1RecordStore:
    """In-memory pre-processed store of Source 1 records for high-throughput feature generation.

    Pre-computes and caches normalized strings, token sets, and digit sets
    indexed by integer s1_idx (0..N-1).
    """

    def __init__(self, n_records: int):
        self.n_records = n_records
        self.s1_ids: List[str] = [""] * n_records
        self.names: List[str] = [""] * n_records
        self.addresses: List[str] = [""] * n_records
        self.countries: List[str] = [""] * n_records
        self.name_tokens: List[Set[str]] = [set()] * n_records
        self.address_tokens: List[Set[str]] = [set()] * n_records
        self.name_digits: List[Set[str]] = [set()] * n_records
        self.address_digits: List[Set[str]] = [set()] * n_records

    def set_record(
        self,
        idx: int,
        entity_id: str,
        raw_name: Optional[str],
        raw_address: Optional[str],
        raw_country: Optional[str],
    ) -> None:
        """Store and pre-process an S1 record at position idx."""
        norm_name = normalize_text(raw_name)
        norm_address = normalize_text(raw_address)
        norm_country = normalize_country(raw_country)

        self.s1_ids[idx] = entity_id
        self.names[idx] = norm_name
        self.addresses[idx] = norm_address
        self.countries[idx] = norm_country
        self.name_tokens[idx] = get_tokens(norm_name)
        self.address_tokens[idx] = get_tokens(norm_address)
        self.name_digits[idx] = get_digit_tokens(norm_name)
        self.address_digits[idx] = get_digit_tokens(norm_address)


def compute_pair_features_from_raw(
    s1_name: str,
    s1_address: str,
    s1_country: str,
    target_name: str,
    target_address: str,
    target_country: str,
    source_number: int,
) -> np.ndarray:
    """Compute the 16 feature values for a single pair of raw records."""
    norm_s1_name = normalize_text(s1_name)
    norm_s1_address = normalize_text(s1_address)
    norm_s1_country = normalize_country(s1_country)

    norm_target_name = normalize_text(target_name)
    norm_target_address = normalize_text(target_address)
    norm_target_country = normalize_country(target_country)

    s1_nt = get_tokens(norm_s1_name)
    target_nt = get_tokens(norm_target_name)
    s1_at = get_tokens(norm_s1_address)
    target_at = get_tokens(norm_target_address)

    s1_nd = get_digit_tokens(norm_s1_name)
    target_nd = get_digit_tokens(norm_target_name)
    s1_ad = get_digit_tokens(norm_s1_address)
    target_ad = get_digit_tokens(norm_target_address)

    # Name features
    name_exact = 1.0 if (norm_s1_name and norm_s1_name == norm_target_name) else 0.0
    name_ratio = (
        fuzz.ratio(norm_s1_name, norm_target_name) / 100.0
        if (norm_s1_name and norm_target_name)
        else 0.0
    )
    name_token_ratio = (
        fuzz.token_set_ratio(norm_s1_name, norm_target_name) / 100.0
        if (norm_s1_name and norm_target_name)
        else 0.0
    )
    name_jaccard = token_jaccard(s1_nt, target_nt)
    name_len_ratio = safe_length_ratio(norm_s1_name, norm_target_name)
    name_digit_overlap = float(len(s1_nd & target_nd))

    # Address features
    address_exact = 1.0 if (norm_s1_address and norm_s1_address == norm_target_address) else 0.0
    address_ratio = (
        fuzz.ratio(norm_s1_address, norm_target_address) / 100.0
        if (norm_s1_address and norm_target_address)
        else 0.0
    )
    address_token_ratio = (
        fuzz.token_set_ratio(norm_s1_address, norm_target_address) / 100.0
        if (norm_s1_address and norm_target_address)
        else 0.0
    )
    address_jaccard = token_jaccard(s1_at, target_at)
    address_len_ratio = safe_length_ratio(norm_s1_address, norm_target_address)
    address_digit_overlap = float(len(s1_ad & target_ad))

    # Joint / source features
    country_same = (
        1.0 if (norm_s1_country and norm_s1_country == norm_target_country) else 0.0
    )
    name_present_both = 1.0 if (norm_s1_name and norm_target_name) else 0.0
    address_present_both = 1.0 if (norm_s1_address and norm_target_address) else 0.0
    source_is_s3 = 1.0 if source_number == 3 else 0.0

    return np.array(
        [
            name_exact,
            name_ratio,
            name_token_ratio,
            name_jaccard,
            name_len_ratio,
            name_digit_overlap,
            address_exact,
            address_ratio,
            address_token_ratio,
            address_jaccard,
            address_len_ratio,
            address_digit_overlap,
            country_same,
            name_present_both,
            address_present_both,
            source_is_s3,
        ],
        dtype=np.float32,
    )


def compute_candidate_features_batch(
    candidate_indices: Sequence[int],
    target_name: str,
    target_address: str,
    target_country: str,
    source_number: int,
    s1_store: S1RecordStore,
) -> np.ndarray:
    """Vectorized feature calculation for a batch of candidate S1 records against one target."""
    n_cands = len(candidate_indices)
    if n_cands == 0:
        return np.empty((0, N_FEATURES), dtype=np.float32)

    norm_target_name = normalize_text(target_name)
    norm_target_address = normalize_text(target_address)
    norm_target_country = normalize_country(target_country)

    target_nt = get_tokens(norm_target_name)
    target_at = get_tokens(norm_target_address)
    target_nd = get_digit_tokens(norm_target_name)
    target_ad = get_digit_tokens(norm_target_address)

    source_is_s3_val = 1.0 if source_number == 3 else 0.0

    X = np.empty((n_cands, N_FEATURES), dtype=np.float32)

    for j, s1_idx in enumerate(candidate_indices):
        s1_n = s1_store.names[s1_idx]
        s1_a = s1_store.addresses[s1_idx]
        s1_c = s1_store.countries[s1_idx]
        s1_nt = s1_store.name_tokens[s1_idx]
        s1_at = s1_store.address_tokens[s1_idx]
        s1_nd = s1_store.name_digits[s1_idx]
        s1_ad = s1_store.address_digits[s1_idx]

        # Name
        name_exact = 1.0 if (s1_n and s1_n == norm_target_name) else 0.0
        name_ratio = (
            fuzz.ratio(s1_n, norm_target_name) / 100.0
            if (s1_n and norm_target_name)
            else 0.0
        )
        name_token_ratio = (
            fuzz.token_set_ratio(s1_n, norm_target_name) / 100.0
            if (s1_n and norm_target_name)
            else 0.0
        )
        name_jaccard = token_jaccard(s1_nt, target_nt)
        name_len_ratio = safe_length_ratio(s1_n, norm_target_name)
        name_digit_overlap = float(len(s1_nd & target_nd))

        # Address
        address_exact = 1.0 if (s1_a and s1_a == norm_target_address) else 0.0
        address_ratio = (
            fuzz.ratio(s1_a, norm_target_address) / 100.0
            if (s1_a and norm_target_address)
            else 0.0
        )
        address_token_ratio = (
            fuzz.token_set_ratio(s1_a, norm_target_address) / 100.0
            if (s1_a and norm_target_address)
            else 0.0
        )
        address_jaccard = token_jaccard(s1_at, target_at)
        address_len_ratio = safe_length_ratio(s1_a, norm_target_address)
        address_digit_overlap = float(len(s1_ad & target_ad))

        # Joint / source
        country_same = 1.0 if (s1_c and s1_c == norm_target_country) else 0.0
        name_present_both = 1.0 if (s1_n and norm_target_name) else 0.0
        address_present_both = 1.0 if (s1_a and norm_target_address) else 0.0

        X[j] = [
            name_exact,
            name_ratio,
            name_token_ratio,
            name_jaccard,
            name_len_ratio,
            name_digit_overlap,
            address_exact,
            address_ratio,
            address_token_ratio,
            address_jaccard,
            address_len_ratio,
            address_digit_overlap,
            country_same,
            name_present_both,
            address_present_both,
            source_is_s3_val,
        ]

    return X


def run_features_tests():
    """Unit assertions to verify feature correctness, ranges, and edge cases."""
    # Test 1: Identical match
    f_match = compute_pair_features_from_raw(
        "Maure Williams Colombier Inc",
        "85 Wayne Avenue, Ticonderoga, NY 12883",
        "US",
        "Maure Williams Colombier Inc",
        "85 Wayne Avenue, Ticonderoga, NY 12883",
        "US",
        source_number=2,
    )
    assert f_match[0] == 1.0, "name_exact must be 1.0"
    assert f_match[1] == 1.0, "name_ratio must be 1.0"
    assert f_match[6] == 1.0, "address_exact must be 1.0"
    assert f_match[7] == 1.0, "address_ratio must be 1.0"
    assert f_match[11] == 2.0, "address_digit_overlap should count '85' and '12883'"
    assert f_match[12] == 1.0, "country_same must be 1.0"
    assert f_match[15] == 0.0, "source_is_s3 must be 0.0 for S2"

    # Test 2: S3 source flag and subtle variation
    f_var = compute_pair_features_from_raw(
        "Maure Williams Colombier Inc",
        "85 Wayne Avenue",
        "US",
        "Maure Wilblims Colombier Inc",
        "",
        "US",
        source_number=3,
    )
    assert f_var[0] == 0.0, "name_exact should be 0.0 on typo"
    assert f_var[1] > 0.90, "name_ratio should be > 0.90 on 1-char typo"
    assert f_var[14] == 0.0, "address_present_both should be 0.0 when target has no address"
    assert f_var[15] == 1.0, "source_is_s3 must be 1.0 for S3"

    # Test 3: S1RecordStore batch calculation matches single calculation
    store = S1RecordStore(2)
    store.set_record(0, "S1-1", "Alpha Inc", "100 Main St", "US")
    store.set_record(1, "S1-2", "Beta LLC", "200 Oak Ave", "France")

    X_batch = compute_candidate_features_batch(
        [0, 1],
        "Alpha Corp",
        "100 Main Street",
        "US",
        source_number=2,
        s1_store=store,
    )
    assert X_batch.shape == (2, 16)
    assert np.isfinite(X_batch).all()

    # Compare row 0 with single calculation
    single_0 = compute_pair_features_from_raw(
        "Alpha Inc", "100 Main St", "US", "Alpha Corp", "100 Main Street", "US", 2
    )
    np.testing.assert_allclose(X_batch[0], single_0, rtol=1e-5, atol=1e-5)

    print("All features unit tests PASSED.")


if __name__ == "__main__":
    run_features_tests()
