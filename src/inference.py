"""Inference, evaluation, and submission generation module for Amazon ML Challenge 2026.

Provides:
- Streaming candidate generation and scoring for large TSV files
- Macro F0.5 per-entity evaluation matching official challenge rules
- Full threshold search and stability reporting
- Compliant export of matching_results.tsv and candidate_pairs.tsv
"""

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.blocking import generate_candidates
from src.features import S1RecordStore, compute_candidate_features_batch
from src.model import predict_probabilities


def compute_entity_f05(
    true_targets: Set[str],
    predicted_targets: Set[str],
) -> float:
    """Compute F0.5 score for a single Source 1 entity according to official challenge rules.

    Rules:
    - If true_targets is empty (singleton/no-match):
        - Score is 1.0 if predicted_targets is empty.
        - Score is 0.0 if predicted_targets is non-empty (false merge).
    - If true_targets is non-empty:
        - Score is 0.0 if predicted_targets is empty.
        - Precision = |true & pred| / |pred|
        - Recall = |true & pred| / |true|
        - F0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    n_true = len(true_targets)
    n_pred = len(predicted_targets)

    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0

    if n_pred == 0:
        return 0.0

    tp = len(true_targets & predicted_targets)
    if tp == 0:
        return 0.0

    precision = tp / float(n_pred)
    recall = tp / float(n_true)
    denom = 0.25 * precision + recall

    if denom <= 0.0:
        return 0.0

    return (1.25 * precision * recall) / denom


def evaluate_macro_f05(
    all_s1_ids: Sequence[str],
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
) -> Dict[str, Any]:
    """Calculate macro F0.5 and detailed diagnostic metrics across all S1 entities."""
    n_s1 = len(all_s1_ids)
    if n_s1 == 0:
        raise ValueError("Cannot evaluate macro F0.5 on empty S1 list.")

    f_scores = np.empty(n_s1, dtype=np.float64)
    pred_counts = np.empty(n_s1, dtype=np.int32)
    true_counts = np.empty(n_s1, dtype=np.int32)

    correct_singletons = 0
    total_singletons = 0
    non_empty_preds = 0

    for i, s1_id in enumerate(all_s1_ids):
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())

        score = compute_entity_f05(true_set, pred_set)
        f_scores[i] = score

        len_true = len(true_set)
        len_pred = len(pred_set)
        true_counts[i] = len_true
        pred_counts[i] = len_pred

        if len_pred > 0:
            non_empty_preds += 1

        if len_true == 0:
            total_singletons += 1
            if len_pred == 0:
                correct_singletons += 1

    macro_f05 = float(f_scores.mean())
    non_empty_pct = (non_empty_preds / float(n_s1)) * 100.0
    mean_preds = float(pred_counts.mean())

    return {
        "macro_f05": macro_f05,
        "n_s1": n_s1,
        "non_empty_prediction_pct": non_empty_pct,
        "mean_predictions_per_s1": mean_preds,
        "total_singletons": total_singletons,
        "correct_singletons": correct_singletons,
        "singleton_accuracy": (
            (correct_singletons / float(total_singletons) * 100.0)
            if total_singletons > 0
            else 0.0
        ),
    }


def export_submission_files(
    output_dir: Union[str, Path],
    ordered_test_s1_ids: Sequence[str],
    matching_predictions: Dict[str, Union[Set[str], List[str]]],
    candidate_predictions: Optional[Dict[str, Union[Set[str], List[str]]]] = None,
    verify_subset: bool = True,
) -> Tuple[Path, Optional[Path]]:
    """Export matching_results.tsv and candidate_pairs.tsv adhering to official specifications.

    Constraints:
    - Tab-separated (.tsv)
    - matching_results.tsv: header = ["source1_entity_id", "matched_entity_ids"]
    - candidate_pairs.tsv: header = ["source1_entity_id", "candidate_entity_ids"]
    - Every S1 entity in ordered_test_s1_ids appears exactly once in the exact order.
    - Matches must only contain S2- or S3- IDs.
    - No duplicate IDs within any row.
    - Matches must be a subset of candidate pairs if candidate_predictions is supplied.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv" if candidate_predictions is not None else None

    # 1. Write matching_results.tsv
    with open(matching_path, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in ordered_test_s1_ids:
            matches = matching_predictions.get(s1_id, [])
            # Deduplicate while preserving deterministic order
            unique_matches: List[str] = []
            seen: Set[str] = set()
            for m in matches:
                m_str = str(m).strip()
                if m_str and m_str not in seen:
                    if not (m_str.startswith("S2-") or m_str.startswith("S3-")):
                        raise ValueError(f"Invalid match ID '{m_str}' for {s1_id} (must start with S2- or S3-)")
                    seen.add(m_str)
                    unique_matches.append(m_str)

            matched_str = ",".join(unique_matches)
            f_match.write(f"{s1_id}\t{matched_str}\n")

    # 2. Write candidate_pairs.tsv if provided
    if candidate_path is not None and candidate_predictions is not None:
        with open(candidate_path, "w", encoding="utf-8") as f_cand:
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in ordered_test_s1_ids:
                cands = candidate_predictions.get(s1_id, [])
                unique_cands: List[str] = []
                seen_cands: Set[str] = set()
                for c in cands:
                    c_str = str(c).strip()
                    if c_str and c_str not in seen_cands:
                        seen_cands.add(c_str)
                        unique_cands.append(c_str)

                if verify_subset:
                    matches = matching_predictions.get(s1_id, [])
                    for m in matches:
                        if m not in seen_cands:
                            raise ValueError(
                                f"Pipeline bug: match '{m}' for S1 '{s1_id}' is not in candidate set!"
                            )

                cands_str = ",".join(unique_cands)
                f_cand.write(f"{s1_id}\t{cands_str}\n")

    return matching_path, candidate_path


def run_inference_tests():
    """Unit assertions to verify evaluation formula and submission export."""
    # Test 1: Example from official challenge description
    # S1 matches [S2-00047, S3-00812] in ground truth
    # Model predicts [S2-00047, S2-00193, S3-00812]
    # Precision = 2/3, Recall = 1.0 -> F0.5 = (1.25 * 2/3 * 1) / (0.25 * 2/3 + 1) = 0.8333 / 1.1667 = 0.7142857
    true_set = {"S2-00047", "S3-00812"}
    pred_set = {"S2-00047", "S2-00193", "S3-00812"}
    score = compute_entity_f05(true_set, pred_set)
    assert abs(score - 0.7142857) < 1e-5, f"Expected ~0.714, got {score}"

    # Test 2: Singleton behavior
    assert compute_entity_f05(set(), set()) == 1.0, "Singleton with empty prediction must score 1.0"
    assert compute_entity_f05(set(), {"S2-1"}) == 0.0, "Singleton with false merge must score 0.0"

    # Test 3: Macro average calculation
    all_s1 = ["S1-1", "S1-2", "S1-3"]
    gt = {
        "S1-1": {"S2-1", "S3-1"},  # entity with matches
        "S1-2": set(),              # singleton
        "S1-3": {"S2-2"},          # entity with match
    }
    preds = {
        "S1-1": {"S2-1", "S3-1"},  # Perfect match: 1.0
        "S1-2": set(),              # Correct empty singleton: 1.0
        "S1-3": set(),              # Missed match: 0.0
    }
    eval_res = evaluate_macro_f05(all_s1, gt, preds)
    expected_macro = (1.0 + 1.0 + 0.0) / 3.0
    assert abs(eval_res["macro_f05"] - expected_macro) < 1e-6
    assert eval_res["total_singletons"] == 1
    assert eval_res["correct_singletons"] == 1

    # Test 4: Export submission files test
    test_dir = Path("/tmp/amazon_ml_sub_test")
    m_path, c_path = export_submission_files(
        test_dir,
        ["S1-1", "S1-2"],
        matching_predictions={"S1-1": ["S2-1", "S3-1"], "S1-2": []},
        candidate_predictions={"S1-1": ["S2-1", "S3-1", "S2-99"], "S1-2": ["S2-10"]},
        verify_subset=True,
    )
    assert m_path.exists()
    assert c_path.exists()
    df_m = pd.read_csv(m_path, sep="\t", dtype=str, keep_default_na=False)
    assert list(df_m.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert len(df_m) == 2
    assert df_m.iloc[0]["matched_entity_ids"] == "S2-1,S3-1"
    assert df_m.iloc[1]["matched_entity_ids"] == ""

    print("All inference unit tests PASSED.")


if __name__ == "__main__":
    run_inference_tests()
