"""
Scoring, Normalization, and Ranking Module.

Handles:
  - Min-Max normalization across the current batch
  - Missing metric handling (NaN, not 0)
  - Effective weight recalculation
  - Final score aggregation
  - Technical Rank vs Recommended Rank
"""

import logging
from typing import Dict, Any, List, Optional

import numpy as np

from absolute_score import evaluate_absolute_score

logger = logging.getLogger(__name__)


def normalize_batch(
    values: List[Optional[float]],
    higher_is_better: bool = True,
) -> List[float]:
    """Min-Max normalize a list of values across the batch.

    If all values are equal (max == min), all get 100.0.
    NaN values remain NaN.
    None values are treated as NaN.

    Args:
        values: List of metric values across all models.
        higher_is_better: If True, higher values → higher score.

    Returns:
        List of normalized scores (0–100).
    """
    clean = [v for v in values if v is not None and not np.isnan(v) and not np.isinf(v)]

    if len(clean) == 0:
        return [np.nan] * len(values)

    vmin = min(clean)
    vmax = max(clean)

    results = []
    for v in values:
        if v is None or np.isnan(v) or np.isinf(v):
            results.append(np.nan)
            continue

        if vmax == vmin:
            results.append(100.0)
        else:
            normalized = (v - vmin) / (vmax - vmin)
            if not higher_is_better:
                normalized = 1.0 - normalized
            results.append(normalized * 100.0)

    return results


def normalize_inverted(values: List[Optional[float]]) -> List[float]:
    """Normalize where lower values are better."""
    return normalize_batch(values, higher_is_better=False)


def compute_final_scores(
    all_results: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Compute final scores for all models using batch normalization.

    Steps:
    1. Collect raw metrics from all models
    2. Normalize each sub-metric across batch
    3. Compute sub-scores with configured weights
    4. Handle missing metrics with effective weight recalculation
    5. Compute final score
    6. Generate Technical Rank and Recommended Rank

    Args:
        all_results: List of per-model result dicts (with raw metrics).
        config: Full configuration.

    Returns:
        Same list with scoring/ranking fields added.
    """
    n = len(all_results)
    if n == 0:
        return all_results

    fc = config.get("final_score", {})
    weights = {
        "ecapa": fc.get("ecapa_weight", 0.40),
        "acoustic": fc.get("acoustic_weight", 0.30),
        "f0": fc.get("f0_weight", 0.15),
        "stability": fc.get("stability_weight", 0.10),
        "artifact": fc.get("artifact_weight", 0.05),
    }

    # ---- Step 1: Normalize ECAPA scores ----
    ecapa_raw = [r.get("ecapa", {}).get("ecapa_score") for r in all_results]
    ecapa_norm = normalize_batch(ecapa_raw, higher_is_better=True)

    # ---- Step 2: Acoustic normalization ----
    ac_cfg = config.get("acoustic", {})
    mcd_weight = ac_cfg.get("mcd_weight", 0.45)
    mel_weight = ac_cfg.get("mel_weight", 0.35)
    mfcc_weight = ac_cfg.get("mfcc_weight", 0.20)

    mcd_raw = [r.get("acoustic", {}).get("mcd_distance") for r in all_results]
    mel_raw = [r.get("acoustic", {}).get("mel_distance") for r in all_results]
    mfcc_raw = [r.get("acoustic", {}).get("mfcc_distance") for r in all_results]

    mcd_score = normalize_inverted(mcd_raw)
    mel_score = normalize_inverted(mel_raw)
    mfcc_score = normalize_inverted(mfcc_raw)

    # ---- Step 3: F0 normalization ----
    f0_cfg = config.get("f0", {})
    f0_corr_weight = f0_cfg.get("correlation_weight", 0.50)
    f0_dtw_weight = f0_cfg.get("dtw_weight", 0.30)
    f0_rmse_weight = f0_cfg.get("rmse_weight", 0.20)

    f0_corr_raw = [r.get("f0", {}).get("f0_correlation") for r in all_results]
    f0_rmse_raw = [r.get("f0", {}).get("f0_rmse") for r in all_results]
    f0_dtw_raw = [r.get("f0", {}).get("f0_dtw") for r in all_results]

    f0_corr_norm = normalize_batch(f0_corr_raw, higher_is_better=True)
    f0_rmse_norm = normalize_inverted(f0_rmse_raw)
    f0_dtw_norm = normalize_inverted(f0_dtw_raw)

    # ---- Step 4: Stability normalization ----
    stability_raw = [r.get("stability", {}).get("window_score_std") for r in all_results]  # std resists outlier windows better than range
    stability_norm = normalize_inverted(stability_raw)

    # ---- Step 5: Artifact normalization ----
    artifact_cfg = config.get("artifacts", {})
    art_weights = {
        "clipping_ratio": artifact_cfg.get("clipping_weight", 0.25),
        "spectral_flatness": artifact_cfg.get("spectral_flatness_weight", 0.30),
        "high_frequency_ratio": artifact_cfg.get("high_frequency_weight", 0.25),
        "harmonic_ratio": artifact_cfg.get("harmonic_anomaly_weight", 0.20),
    }

    clip_raw = [r.get("artifacts", {}).get("clipping_ratio") for r in all_results]
    sf_raw = [r.get("artifacts", {}).get("spectral_flatness") for r in all_results]
    sf_orig = [r.get("artifacts", {}).get("orig_spectral_flatness") for r in all_results]
    hf_raw = [r.get("artifacts", {}).get("high_frequency_ratio") for r in all_results]
    hf_orig = [r.get("artifacts", {}).get("orig_high_frequency_ratio") for r in all_results]
    harm_raw = [r.get("artifacts", {}).get("harmonic_ratio") for r in all_results]

    # Delta from original (closeness to reference matters for tone/timbre preservation)
    sf_delta = [abs(sf_raw[i] - sf_orig[i]) if (sf_raw[i] is not None and sf_orig[i] is not None
                and not np.isnan(sf_raw[i]) and not np.isnan(sf_orig[i])) else None
                for i in range(len(all_results))]
    hf_delta = [abs(hf_raw[i] - hf_orig[i]) if (hf_raw[i] is not None and hf_orig[i] is not None
                and not np.isnan(hf_raw[i]) and not np.isnan(hf_orig[i])) else None
                for i in range(len(all_results))]

    clip_norm = normalize_inverted(clip_raw)
    sf_norm = normalize_inverted(sf_delta)   # Lower delta from original = better
    hf_norm = normalize_inverted(hf_delta)   # Lower delta from original = better
    harm_norm = _normalize_optimal_batch(harm_raw, optimal=0.75)  # RVC synth naturally harmonic-rich (0.6-0.9)

    # ---- Step 6: Build per-model scores ----
    for i, result in enumerate(all_results):
        # Compute sub-scores
        ecapa_score = ecapa_norm[i]

        # Acoustic composite
        ac_parts = []
        ac_w = []
        if not np.isnan(mcd_score[i]):
            ac_parts.append(mcd_score[i])
            ac_w.append(mcd_weight)
        if not np.isnan(mel_score[i]):
            ac_parts.append(mel_score[i])
            ac_w.append(mel_weight)
        if not np.isnan(mfcc_score[i]):
            ac_parts.append(mfcc_score[i])
            ac_w.append(mfcc_weight)

        acoustic_score = _weighted_sum(ac_parts, ac_w)

        # F0 composite
        f0_parts = []
        f0_w = []
        if not np.isnan(f0_corr_norm[i]):
            f0_parts.append(f0_corr_norm[i])
            f0_w.append(f0_corr_weight)
        if not np.isnan(f0_dtw_norm[i]):
            f0_parts.append(f0_dtw_norm[i])
            f0_w.append(f0_dtw_weight)
        if not np.isnan(f0_rmse_norm[i]):
            f0_parts.append(f0_rmse_norm[i])
            f0_w.append(f0_rmse_weight)

        f0_score = _weighted_sum(f0_parts, f0_w)

        # Stability score
        st_score = stability_norm[i]

        # Artifact composite
        art_parts = []
        art_w = []
        if not np.isnan(clip_norm[i]):
            art_parts.append(clip_norm[i])
            art_w.append(art_weights["clipping_ratio"])
        if not np.isnan(sf_norm[i]):
            art_parts.append(sf_norm[i])
            art_w.append(art_weights["spectral_flatness"])
        if not np.isnan(hf_norm[i]):
            art_parts.append(hf_norm[i])
            art_w.append(art_weights["high_frequency_ratio"])
        if not np.isnan(harm_norm[i]):
            art_parts.append(harm_norm[i])
            art_w.append(art_weights["harmonic_ratio"])

        artifact_score = _weighted_sum(art_parts, art_w)

        # Identify missing metrics
        missing = []
        effective_weights = {}
        weight_sum = 0.0

        if ecapa_score is not None and not np.isnan(ecapa_score):
            effective_weights["ecapa"] = weights["ecapa"]
            weight_sum += weights["ecapa"]
        else:
            missing.append("ecapa")
            ecapa_score = np.nan
            effective_weights["ecapa"] = 0.0

        if acoustic_score is not None and not np.isnan(acoustic_score):
            effective_weights["acoustic"] = weights["acoustic"]
            weight_sum += weights["acoustic"]
        else:
            missing.append("acoustic")
            acoustic_score = np.nan
            effective_weights["acoustic"] = 0.0

        if f0_score is not None and not np.isnan(f0_score):
            effective_weights["f0"] = weights["f0"]
            weight_sum += weights["f0"]
        else:
            missing.append("f0")
            f0_score = np.nan
            effective_weights["f0"] = 0.0

        if st_score is not None and not np.isnan(st_score):
            effective_weights["stability"] = weights["stability"]
            weight_sum += weights["stability"]
        else:
            missing.append("stability")
            st_score = np.nan
            effective_weights["stability"] = 0.0

        if artifact_score is not None and not np.isnan(artifact_score):
            effective_weights["artifact"] = weights["artifact"]
            weight_sum += weights["artifact"]
        else:
            missing.append("artifact")
            artifact_score = np.nan
            effective_weights["artifact"] = 0.0

        # Re-normalize effective weights
        if weight_sum > 0:
            for k in effective_weights:
                effective_weights[k] /= weight_sum

        # Final score
        final_score = 0.0
        if weight_sum > 0:
            final_score = (
                effective_weights.get("ecapa", 0) * (ecapa_score if not np.isnan(ecapa_score) else 0)
                + effective_weights.get("acoustic", 0) * (acoustic_score if not np.isnan(acoustic_score) else 0)
                + effective_weights.get("f0", 0) * (f0_score if not np.isnan(f0_score) else 0)
                + effective_weights.get("stability", 0) * (st_score if not np.isnan(st_score) else 0)
                + effective_weights.get("artifact", 0) * (artifact_score if not np.isnan(artifact_score) else 0)
            )

        # Warnings
        warnings = result.get("warnings", [])

        # Status
        if missing:
            if all(k in missing for k in weights):
                status = f"MISSING:{','.join(missing)}"
            else:
                status = f"OK; MISSING:{','.join(missing)}"
        else:
            status = "OK"

        # Speech ratio and duration
        speech_ratio = result.get("speech_ratio", None)
        duration = result.get("duration", None)
        num_windows = result.get("num_windows", 0)

        # Store scores
        result["scores"] = {
            "ecapa_score": ecapa_score,
            "acoustic_score": acoustic_score,
            "f0_score": f0_score,
            "stability_score": st_score,
            "artifact_score": artifact_score,
            "final_score": final_score,
            "ecapa_mean": result.get("ecapa", {}).get("ecapa_mean"),
            "ecapa_std": result.get("ecapa", {}).get("ecapa_std"),
            "ecapa_min": result.get("ecapa", {}).get("ecapa_min"),
            "ecapa_max": result.get("ecapa", {}).get("ecapa_max"),

            "mcd_distance": result.get("acoustic", {}).get("mcd_distance"),
            "mcd_score": mcd_score[i],
            "mel_distance": result.get("acoustic", {}).get("mel_distance"),
            "mel_score": mel_score[i],
            "mfcc_distance": result.get("acoustic", {}).get("mfcc_distance"),
            "mfcc_score": mfcc_score[i],

            "f0_correlation": result.get("f0", {}).get("f0_correlation"),
            "f0_rmse": result.get("f0", {}).get("f0_rmse"),
            "f0_dtw": result.get("f0", {}).get("f0_dtw"),

            "window_score_mean": result.get("stability", {}).get("window_score_mean"),
            "window_score_std": result.get("stability", {}).get("window_score_std"),

            "clipping_ratio": result.get("artifacts", {}).get("clipping_ratio"),
            "peak": result.get("artifacts", {}).get("peak"),
            "rms": result.get("artifacts", {}).get("rms"),
            "spectral_flatness": result.get("artifacts", {}).get("spectral_flatness"),
            "high_frequency_ratio": result.get("artifacts", {}).get("high_frequency_ratio"),
            "harmonic_ratio": result.get("artifacts", {}).get("harmonic_ratio"),
            "orig_clipping_ratio": result.get("artifacts", {}).get("orig_clipping_ratio"),
            "orig_spectral_flatness": result.get("artifacts", {}).get("orig_spectral_flatness"),
            "orig_high_frequency_ratio": result.get("artifacts", {}).get("orig_high_frequency_ratio"),

            "speech_ratio": speech_ratio,
            "duration": duration,
            "num_windows": num_windows,

            "effective_weights": effective_weights,
            "missing_metrics": missing,
            "warnings": warnings,
            "status": status,
        }

        # ---- Absolute score (batch-independent quality rating) ----
        result["scores"]["absolute_score"] = evaluate_absolute_score(
            ecapa_similarity=result.get("ecapa", {}).get("ecapa_score") or 0.0,
            mcd_distance=result.get("acoustic", {}).get("mcd_distance"),
            f0_correlation=result.get("f0", {}).get("f0_correlation") or 0.0,
            stability_mean=result.get("stability", {}).get("window_score_mean") or 0.0,
        )

    # ---- Step 7: Technical Rank ----
    tech_scores = [
        (i, r["scores"]["final_score"])
        for i, r in enumerate(all_results)
    ]
    tech_scores.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -1, reverse=True)
    for rank, (idx, _) in enumerate(tech_scores, start=1):
        all_results[idx]["scores"]["technical_rank"] = rank

    # ---- Step 8: Recommended Rank ----
    # Start from technical rank, but demote models with severe warnings
    recommended = []
    for i, r in enumerate(all_results):
        w = r.get("warnings", [])
        demerits = 0
        if "SEVERE_CLIPPING" in w:
            demerits += 3
        if "HIGH_FREQUENCY_ANOMALY" in w:
            demerits += 2
        if "SEVERE_SPECTRAL_ANOMALY" in w:
            demerits += 0  # Not a real anomaly for RVC synthesis (naturally tonal)
        if "LOW_SPEECH_RATIO" in w:
            demerits += 1

        score = r["scores"]["final_score"]
        if np.isnan(score):
            score = -1

        effective = score - demerits * 5  # Demote by 5 points per demerit
        recommended.append((i, effective))

    recommended.sort(key=lambda x: x[1], reverse=True)
    for rank, (idx, _) in enumerate(recommended, start=1):
        all_results[idx]["scores"]["recommended_rank"] = rank
        all_results[idx]["scores"]["recommended_effective"] = recommended[rank - 1][1]

    return all_results


def _weighted_sum(values: List[float], weights: List[float]) -> float:
    """Compute re-normalized weighted sum, ignoring NaN values.

    Valid (value, weight) pairs are collected; weights are re-normalized
    to sum to 1. Returns NaN if no valid pairs exist.
    """
    if not values or not weights:
        return np.nan
    valid_pairs = [(v, w) for v, w in zip(values, weights) if not np.isnan(v)]
    if not valid_pairs:
        return np.nan
    total_w = sum(w for _, w in valid_pairs)
    if total_w == 0:
        return np.nan
    return sum(v * w / total_w for v, w in valid_pairs)


def _normalize_optimal_batch(values: List[Optional[float]], optimal: float) -> List[float]:
    """Batch normalize where being close to a target 'optimal' value is best.

    Distance from optimal is min-max normalized across the batch, with
    shorter distances scoring higher. Returns 0-100 scores.
    """
    clean = [v for v in values if v is not None and not np.isnan(v)]
    if len(clean) <= 1:
        return [100.0 if v is not None and not np.isnan(v) else np.nan for v in values]

    distances = [abs(v - optimal) for v in clean]
    d_min, d_max = min(distances), max(distances)

    results = []
    for v in values:
        if v is None or np.isnan(v):
            results.append(np.nan)
            continue
        d = abs(v - optimal)
        if d_max == d_min:
            results.append(100.0)
        else:
            results.append((1.0 - (d - d_min) / (d_max - d_min)) * 100.0)
    return results
