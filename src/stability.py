"""
Stability Module.

Measures how consistent the model's performance is across different time windows.
Uses window-level ECAPA/Acoustic/F0 scores to compute a composite window score,
then uses the standard deviation to measure stability.

Weight: 10% of final score.
Stability_score = 1 - normalized(window_score_std)
(Lower std → higher stability score)
"""

import logging
from typing import List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_window_scores(
    ecapa_per_window: List[float],
    acoustic_per_window: List[float],
    f0_per_window: List[float],
    ecapa_weight: float = 0.40,
    acoustic_weight: float = 0.30,
    f0_weight: float = 0.15,
) -> List[float]:
    """Compute composite window-level scores from sub-metrics.

    Args:
        ecapa_per_window: ECAPA similarity per window.
        acoustic_per_window: Acoustic similarity per window.
        f0_per_window: F0 correlation per window.
        ecapa_weight: Weight for ECAPA in composite (default: 0.40).
        acoustic_weight: Weight for acoustic in composite (default: 0.30).
        f0_weight: Weight for F0 in composite (default: 0.15).

    Returns:
        List of composite window scores (one per window).
    """
    n = min(len(ecapa_per_window), len(acoustic_per_window), len(f0_per_window))
    if n == 0:
        return []

    window_scores = []
    for i in range(n):
        parts = []
        weights = []

        ecapa = ecapa_per_window[i]
        if not np.isnan(ecapa):
            parts.append(ecapa)
            weights.append(ecapa_weight)

        acoustic = acoustic_per_window[i]
        if not np.isnan(acoustic):
            parts.append(acoustic)
            weights.append(acoustic_weight)

        f0v = f0_per_window[i]
        if not np.isnan(f0v):
            parts.append(f0v)
            weights.append(f0_weight)

        if len(parts) == 0:
            window_scores.append(np.nan)
            continue

        # Normalize weights to sum to 1
        weights = np.array(weights) / sum(weights)
        window_scores.append(float(np.dot(parts, weights)))

    return window_scores


def compute_stability(
    window_scores: List[float],
) -> Dict[str, Any]:
    """Compute stability metrics from window-level scores.

    Args:
        window_scores: List of composite scores per window.

    Returns:
        Dict with window_score_mean, window_score_std, n_windows.
    """
    valid_scores = [s for s in window_scores if not np.isnan(s)]

    if len(valid_scores) == 0:
        return {
            "window_score_mean": None,
            "window_score_std": None,
            "window_score_min": None,
            "window_score_max": None,
            "window_score_range": None,
            "n_windows": len(window_scores),
            "n_valid_windows": 0,
            "status": "FAILED: No valid window scores",
        }

    mean_score = float(np.mean(valid_scores))
    std_score = float(np.std(valid_scores))

    min_score = float(np.min(valid_scores))
    max_score = float(np.max(valid_scores))

    score_range = max_score - min_score

    return {
        "window_score_mean": mean_score,
        "window_score_std": std_score,
        "window_score_min": min_score,
        "window_score_max": max_score,
        "window_score_range": score_range,
        "n_windows": len(window_scores),
        "n_valid_windows": len(valid_scores),
        "status": "OK",
    }
