"""Data processing and canonical normalization module for Amazon ML Challenge 2026.

This module provides the single source of truth for:
- Text normalization (Unicode NFKC, casefolding, punctuation normalization, whitespace collapsing)
- Tokenization (set-based and list-based, with token length filtering >= 2)
- Digit token extraction
- Entity ID parsing and source detection
- ID split file management
"""

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

# Compiled regexes for performance
RE_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
RE_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)
RE_DIGITS = re.compile(r"\d+", flags=re.UNICODE)


def normalize_text(value: Optional[Union[str, float]]) -> str:
    """Canonical text normalization.

    Rules:
    1. None / NaN converted to empty string
    2. Cast to string
    3. Unicode NFKC normalization
    4. Casefold (more aggressive than lower() for international scripts)
    5. Convert all punctuation / non-alphanumeric (except whitespace) to spaces
    6. Collapse multiple whitespace characters into single space
    7. Strip leading and trailing whitespace
    """
    if value is None:
        return ""
    # Check for float NaN
    if isinstance(value, float) and value != value:
        return ""
    val_str = str(value)
    if not val_str:
        return ""

    val_str = unicodedata.normalize("NFKC", val_str)
    val_str = val_str.casefold()
    val_str = RE_PUNCTUATION.sub(" ", val_str)
    val_str = RE_WHITESPACE.sub(" ", val_str)
    return val_str.strip()


def normalize_country(value: Optional[Union[str, float]]) -> str:
    """Normalize country code or name.

    Keeps open-set country representation: casefolded and stripped.
    Does NOT hardcode to US/India.
    """
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value).casefold().strip()


def get_tokens(value: Optional[Union[str, float]]) -> Set[str]:
    """Extract unique token set from text with token length >= 2.

    Applies canonical normalization first.
    Returns set of tokens of length >= 2.
    """
    normalized = normalize_text(value)
    if not normalized:
        return set()
    return {token for token in normalized.split() if len(token) >= 2}


def get_tokens_list(value: Optional[Union[str, float]]) -> List[str]:
    """Extract ordered token list from text with token length >= 2.

    Applies canonical normalization first.
    Returns list of tokens of length >= 2 in original order.
    """
    normalized = normalize_text(value)
    if not normalized:
        return []
    return [token for token in normalized.split() if len(token) >= 2]


def get_digit_tokens(value: Optional[Union[str, float]]) -> Set[str]:
    """Extract set of digit sequences from text.

    Applies canonical normalization first.
    """
    normalized = normalize_text(value)
    if not normalized:
        return set()
    return set(RE_DIGITS.findall(normalized))


def parse_entity_id(entity_id: str) -> Tuple[str, int]:
    """Parse entity ID into source prefix and integer suffix.

    Example: 'S1-965667' -> ('S1', 965667)
             'S2-00047'  -> ('S2', 47)
    """
    entity_id = entity_id.strip()
    if "-" not in entity_id:
        raise ValueError(f"Invalid entity_id format (expected prefix-number): '{entity_id}'")
    prefix, num_str = entity_id.split("-", 1)
    return prefix, int(num_str)


def source_number_from_id(entity_id: str) -> int:
    """Return integer source number (1, 2, or 3) from entity ID."""
    prefix, _ = parse_entity_id(entity_id)
    if prefix == "S1":
        return 1
    elif prefix == "S2":
        return 2
    elif prefix == "S3":
        return 3
    else:
        raise ValueError(f"Unknown source prefix in entity_id: '{entity_id}'")


def load_id_list(file_path: Union[str, Path]) -> List[str]:
    """Load ordered list of entity IDs from a line-separated text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_id_set(file_path: Union[str, Path]) -> Set[str]:
    """Load set of entity IDs from a line-separated text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def build_id_mapping(ordered_ids: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build forward and reverse integer index mappings for a given list of IDs."""
    id_to_idx = {entity_id: idx for idx, entity_id in enumerate(ordered_ids)}
    idx_to_id = {idx: entity_id for entity_id, idx in id_to_idx.items()}
    return id_to_idx, idx_to_id


def run_data_processing_tests():
    """Unit assertions to verify canonical behavior and prevent regressions."""
    # 1. Normalization assertions
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""
    assert normalize_text("ABC Corp.") == "abc corp"
    assert normalize_text("Pvt.   Ltd.") == "pvt ltd"
    assert normalize_text("Café & Restaurant") == "café restaurant"
    assert normalize_text("ﬃctitious") == "ffictitious"
    assert normalize_text("Unit #402, 10th Main") == "unit 402 10th main"
    assert normalize_text("Mumbai - 400001") == "mumbai 400001"

    # 2. Tokenization length >= 2 rule
    tokens = get_tokens("A B CD EFG H I")
    assert tokens == {"cd", "efg"}  # Single letters discarded
    assert get_tokens("A") == set()
    assert get_tokens("1") == set()
    assert get_tokens("12") == {"12"}

    # 3. Country normalization (open-set)
    assert normalize_country("US") == "us"
    assert normalize_country("India ") == "india"
    assert normalize_country("France") == "france"
    assert normalize_country("FR") == "fr"
    assert normalize_country(None) == ""

    # 4. Digit tokens
    assert get_digit_tokens("Plot 123, Road 45, Sector 9") == {"123", "45", "9"}
    assert get_digit_tokens("No digits here") == set()

    # 5. Entity ID parsing
    assert parse_entity_id("S1-965667") == ("S1", 965667)
    assert parse_entity_id("S2-00047") == ("S2", 47)
    assert parse_entity_id("S3-11291185") == ("S3", 11291185)
    assert source_number_from_id("S1-123") == 1
    assert source_number_from_id("S2-456") == 2
    assert source_number_from_id("S3-789") == 3

    print("All data_processing unit tests PASSED.")


if __name__ == "__main__":
    run_data_processing_tests()
