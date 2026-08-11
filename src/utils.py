"""Shared utility functions for RVC Evaluator modules."""

from typing import List, Dict, Any, Tuple

import numpy as np
from scipy.spatial.distance import cdist


# MCD formula scale factor: (10 * sqrt(2)) / ln(10)
MCD_SCALE = (10.0 * np.sqrt(2.0)) / np.log(10.0)


def flatten_model_entry(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a scores.json model entry to a flat dict for comparison/trend analysis.

    Input (from scores.json):
        {"model": "24", "final_score": 89.03,
         "sub_scores": {"ecapa": 100.0, ...}, ...}

    Output:
        {"model": "24", "score": 89.03, "ecapa": 100.0, ...}
    """
    entry = {"model": model_data.get("model", model_data.get("model_name", "unknown"))}

    # scores.json uses "final_score" and "sub_scores" keys
    entry["score"] = model_data.get("final_score", model_data.get("score", 0))
    subs = model_data.get("sub_scores", model_data.get("scores", {}))
    entry["ecapa"] = subs.get("ecapa", subs.get("ecapa_score", 0))
    entry["acoustic"] = subs.get("acoustic", subs.get("acoustic_score", 0))
    entry["f0"] = subs.get("f0", subs.get("f0_score", 0))
    entry["stability"] = subs.get("stability", subs.get("stability_score", 0))
    entry["artifact"] = subs.get("artifact", subs.get("artifact_score", 0))

    return entry


def dtw_align(
    features_a: np.ndarray,
    features_b: np.ndarray,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute DTW alignment between two feature sequences.

    Args:
        features_a: First feature matrix (n_frames_a, n_features).
        features_b: Second feature matrix (n_frames_b, n_features).
        metric: Distance metric for frame comparison.

    Returns:
        Tuple of (aligned_a_indices, aligned_b_indices, normalized_distance).
        Distance is divided by DTW path length.
    """
    if features_a.shape[0] == 0 or features_b.shape[0] == 0:
        return np.array([]), np.array([]), float('inf')

    dist_matrix = cdist(features_a, features_b, metric=metric)

    n, m = dist_matrix.shape
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_matrix[i - 1, j - 1]
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            )

    # Backtrack to find optimal path
    path_a = []
    path_b = []
    i, j = n, m
    while i > 0 and j > 0:
        path_a.append(i - 1)
        path_b.append(j - 1)
        min_prev = min(
            dtw_matrix[i - 1, j],
            dtw_matrix[i, j - 1],
            dtw_matrix[i - 1, j - 1],
        )
        if dtw_matrix[i - 1, j - 1] == min_prev:
            i -= 1
            j -= 1
        elif dtw_matrix[i, j - 1] == min_prev:
            j -= 1
        else:
            i -= 1

    path_a = np.array(path_a[::-1])
    path_b = np.array(path_b[::-1])

    path_distance = dtw_matrix[n, m]
    normalized_distance = path_distance / len(path_a)

    return path_a, path_b, float(normalized_distance)


def within_model_normalize(values: List[float], higher_is_better: bool = True) -> List[float]:
    """Normalize a list of values within a single model's window batch (Min-Max).

    Returns scores 0-100. NaN/None/Inf values remain NaN.

    Args:
        values: List of raw metric values (one per window).
        higher_is_better: If True, higher raw value → higher score.
                          If False, lower raw value → higher score.

    Returns:
        List of normalized scores (0-100), same length as input.
    """
    clean = [v for v in values if v is not None and not np.isnan(v) and not np.isinf(v)]
    # With ≤1 valid value, variance is zero — all valid entries score 100.0
    if len(clean) <= 1:
        return [100.0 if v is not None and not np.isnan(v) else np.nan for v in values]
    vmin, vmax = min(clean), max(clean)
    if vmax == vmin:
        return [100.0 if v is not None and not np.isnan(v) else np.nan for v in values]
    results = []
    for v in values:
        if v is None or np.isnan(v) or np.isinf(v):
            results.append(np.nan)
            continue
        norm = (v - vmin) / (vmax - vmin)
        if not higher_is_better:
            norm = 1.0 - norm
        results.append(norm * 100.0)
    return results
