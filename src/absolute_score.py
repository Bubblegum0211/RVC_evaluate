"""
Absolute Score — standalone quality rating independent of batch comparison.

Unlike Min-Max batch normalization (which maps the best model to 100 and
worst to 0), absolute scoring gives each model a fixed quality label based
on calibrated thresholds.  This means scores are comparable across different
evaluation runs — add a new model next week and its absolute score won't
change just because the batch changed.

Thresholds are calibrated against:
  - Self-baseline (original vs original, different windows)
  - Domain knowledge (MCD, F0 correlation)
  - Observed data from real RVC model evaluations

Recalibrate thresholds if your use case / dataset changes significantly.
"""

from typing import Dict, Any


# ---------------------------------------------------------------------------
# ECAPA speaker similarity (same-content comparison)
# Self-baseline (different content): ~0.41
# Good models (same content):        ~0.64–0.80
# ---------------------------------------------------------------------------
_ECAPA_TIERS = [
    (0.80, "excellent",  40),
    (0.75, "very_good",  35),
    (0.70, "good",       30),
    (0.65, "acceptable", 25),
    (0.60, "fair",       20),
    (0.55, "poor",       10),
    (0.00, "very_poor",   0),
]


# ---------------------------------------------------------------------------
# MCD (Mel Cepstral Distortion) -- lower is better
# Self-baseline (different content): ~527
# Good models (same content):        ~200-300
# ---------------------------------------------------------------------------
_MCD_TIERS = [
    (180, "excellent",  30),
    (210, "very_good",  27),
    (240, "good",       24),
    (270, "above_avg",  21),
    (300, "acceptable", 18),
    (330, "below_avg",  15),
    (380, "fair",       12),
    (450, "poor",        8),
    (600, "very_poor",   4),
    (999, "unusable",    0),
]


# ---------------------------------------------------------------------------
# F0 correlation — higher is better
# Self-baseline (different content): ~0.21
# Good models:                       ~0.97–0.99
# ---------------------------------------------------------------------------
_F0_TIERS = [
    (0.99, "excellent",  20),
    (0.98, "very_good",  18),
    (0.97, "good",       16),
    (0.95, "acceptable", 12),
    (0.90, "fair",        8),
    (0.00, "poor",        0),
]


# ---------------------------------------------------------------------------
# Stability (window_score_mean from composite window scoring)
# Ranges from 0–100 internally; good models typically 30–40+
# ---------------------------------------------------------------------------
_STABILITY_TIERS = [
    (45, "excellent",  10),
    (38, "very_good",   8),
    (30, "good",        6),
    (22, "acceptable",  4),
    (15, "fair",        2),
    ( 0, "poor",        0),
]


def _tier_lookup(value: float, tiers: list) -> tuple:
    """Return (level_label, points) for the first matching tier."""
    for threshold, label, points in tiers:
        if value >= threshold:
            return label, points
    return tiers[-1][1], tiers[-1][2]  # fallback


def evaluate_absolute_score(
    ecapa_similarity: float = 0.0,
    mcd_distance: float = 999.0,
    f0_correlation: float = 0.0,
    stability_mean: float = 0.0,
) -> Dict[str, Any]:
    """Compute an absolute quality score (0–100) for a single model.

    Args:
        ecapa_similarity: Composite ECAPA similarity (ecapa_score from ecapa.py).
        mcd_distance:    MCD distance (lower is better).
        f0_correlation:  F0 Pearson correlation with original.
        stability_mean:  Window-level composite score mean (from stability.py).

    Returns:
        Dict with:
          - score:         0–100 numeric absolute score
          - overall_level: "recommended" | "usable" | "not_recommended"
          - details:       per-dimension {value, level, points}
    """
    # --- per-dimension tier lookup ---
    ecapa_level, ecapa_points = _tier_lookup(ecapa_similarity, _ECAPA_TIERS)

    # MCD: iterate manually (lower is better, so scan ascending thresholds)
    mcd_level, mcd_points = "very_poor", 0
    if mcd_distance is not None:
        for threshold, label, pts in _MCD_TIERS:
            if mcd_distance <= threshold:
                mcd_level, mcd_points = label, pts
                break

    f0_level,  f0_points  = _tier_lookup(f0_correlation, _F0_TIERS)
    st_level,  st_points  = _tier_lookup(stability_mean, _STABILITY_TIERS)

    # --- absolute score ---
    score = ecapa_points + mcd_points + f0_points + st_points

    # --- overall level (based on absolute score, not single-dimension thresholds) ---
    if score >= 70:
        overall = "recommended"
    elif score >= 45:
        overall = "usable"
    else:
        overall = "not_recommended"

    return {
        "score": round(score, 1),
        "overall_level": overall,
        "details": {
            "ecapa":    {"value": ecapa_similarity, "level": ecapa_level, "points": ecapa_points},
            "mcd":      {"value": mcd_distance,    "level": mcd_level,   "points": mcd_points},
            "f0":       {"value": f0_correlation,  "level": f0_level,    "points": f0_points},
            "stability":{"value": stability_mean,  "level": st_level,    "points": st_points},
        },
    }
