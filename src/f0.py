"""
F0 / Prosody Analysis Module.

Computes F0 (fundamental frequency) metrics for spoken voice evaluation:
  - F0 correlation (Pearson)
  - F0 RMSE
  - F0 DTW distance
  - Voiced ratio comparison

Uses voiced mask — does NOT set unvoiced frames to 0.
Includes octave error correction.

Weight: 15% of final score.
F0_score = 0.50 * correlation_score + 0.30 * DTW_score + 0.20 * RMSE_score
"""

import logging
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

try:
    from utils import within_model_normalize as _within_model_normalize, dtw_align
except ImportError:
    from .utils import within_model_normalize as _within_model_normalize, dtw_align
import librosa
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def extract_f0(
    audio: np.ndarray,
    sr: int,
    min_hz: float = 50.0,
    max_hz: float = 600.0,
    hop_length: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract F0 (pitch) and voiced flag from audio.

    Uses librosa's pyin which provides voiced probability.

    Args:
        audio: Audio array.
        sr: Sample rate.
        min_hz: Minimum F0 in Hz.
        max_hz: Maximum F0 in Hz.
        hop_length: Hop length for F0 extraction.

    Returns:
        Tuple of (f0_array, voiced_flag_array).
        f0_array: F0 values in Hz (NaN for unvoiced).
        voiced_flag: Boolean array of same length.
    """
    if len(audio) < hop_length:
        return np.array([]), np.array([])

    f0, voiced_flag, _ = librosa.pyin(
        y=audio,
        fmin=min_hz,
        fmax=max_hz,
        sr=sr,
        hop_length=hop_length,
    )

    # Handle NaN
    if f0 is None:
        return np.array([]), np.array([])

    f0 = np.nan_to_num(f0, nan=0.0)
    if voiced_flag is None:
        voiced_flag = f0 > min_hz

    return f0, voiced_flag.astype(bool)


def octave_correction(
    f0_orig: np.ndarray,
    f0_model: np.ndarray,
    voiced_orig: np.ndarray,
    voiced_model: np.ndarray,
) -> np.ndarray:
    """Apply octave error correction to model F0.

    Checks for octave doubling/halving errors in model F0 relative to original.
    For each voiced frame, tries 0.5x, 1x, 2x and picks the one closest to original.

    Args:
        f0_orig: Original F0 array.
        f0_model: Model F0 array.
        voiced_orig: Voiced flags for original.
        voiced_model: Voiced flags for model.

    Returns:
        Corrected model F0 array.
    """
    corrected = f0_model.copy()
    for i in range(len(f0_model)):
        if not voiced_model[i] or not voiced_orig[i]:
            continue
        if f0_model[i] < 1e-6 or f0_orig[i] < 1e-6:
            continue

        options = [f0_model[i] * 0.5, f0_model[i], f0_model[i] * 2.0]

        min_diff = float('inf')
        best = f0_model[i]
        for opt in options:
            if opt < 1e-6:
                continue
            diff = abs(opt - f0_orig[i])
            if diff < min_diff:
                min_diff = diff
                best = opt
        corrected[i] = best
    return corrected


def compute_f0_correlation(
    f0_orig: np.ndarray,
    f0_model: np.ndarray,
    voiced_mask: np.ndarray,
) -> Optional[float]:
    """Compute Pearson correlation of F0 on voiced frames only.

    Does NOT set unvoiced F0 to 0 before computing.

    Args:
        f0_orig: Original F0.
        f0_model: Model F0.
        voiced_mask: Combined voiced mask (both must be voiced).

    Returns:
        Pearson correlation coefficient, or None if not computable.
    """
    if np.sum(voiced_mask) < 3:
        return None

    orig_voiced = f0_orig[voiced_mask]
    model_voiced = f0_model[voiced_mask]

    # Filter valid values
    valid = (orig_voiced > 1e-6) & (model_voiced > 1e-6)
    if np.sum(valid) < 3:
        return None

    try:
        corr, _ = pearsonr(orig_voiced[valid], model_voiced[valid])
        return float(corr)
    except Exception:
        return None


def compute_f0_rmse(
    f0_orig: np.ndarray,
    f0_model: np.ndarray,
    voiced_mask: np.ndarray,
) -> Optional[float]:
    """Compute normalized F0 RMSE on voiced frames only.

    RMSE is divided by the mean F0 of the original to produce a unitless
    relative error (e.g. 0.05 = 5% error).  This prevents a constant-Hz
    error from being penalized differently at low vs high pitches.

    Args:
        f0_orig: Original F0.
        f0_model: Model F0.
        voiced_mask: Combined voiced mask.

    Returns:
        Normalized RMSE (unitless relative error), or None if not computable.
    """
    if np.sum(voiced_mask) < 1:
        return None

    orig_voiced = f0_orig[voiced_mask]
    model_voiced = f0_model[voiced_mask]

    valid = (orig_voiced > 1e-6) & (model_voiced > 1e-6)
    if np.sum(valid) < 1:
        return None

    rmse_hz = float(np.sqrt(np.mean((orig_voiced[valid] - model_voiced[valid]) ** 2)))
    mean_f0 = float(np.mean(orig_voiced[valid]))

    if mean_f0 < 1e-6:
        return None

    return rmse_hz / mean_f0


def compute_f0_dtw(
    f0_orig: np.ndarray,
    f0_model: np.ndarray,
    voiced_mask: np.ndarray,
) -> Optional[float]:
    """Compute F0 DTW distance on voiced frames.

    Args:
        f0_orig: Original F0.
        f0_model: Model F0.
        voiced_mask: Combined voiced mask.

    Returns:
        DTW distance, or None if not computable.
    """
    if np.sum(voiced_mask) < 2:
        return None

    orig_voiced = f0_orig[voiced_mask].reshape(-1, 1)
    model_voiced = f0_model[voiced_mask].reshape(-1, 1)

    valid = (orig_voiced.flatten() > 1e-6) & (model_voiced.flatten() > 1e-6)
    if np.sum(valid) < 2:
        return None

    orig_valid = orig_voiced[valid]
    model_valid = model_voiced[valid]

    if len(orig_valid) < 2 or len(model_valid) < 2:
        return None

    _, _, distance = dtw_align(orig_valid, model_valid)
    if np.isinf(distance):
        return None
    return distance


def compute_voiced_ratio(voiced: np.ndarray) -> float:
    """Compute the ratio of voiced frames.

    Args:
        voiced: Boolean array indicating voiced frames.

    Returns:
        Ratio of voiced frames (0.0 to 1.0).
    """
    if len(voiced) == 0:
        return 0.0
    return float(np.sum(voiced)) / len(voiced)


def compute_f0_scores(
    original_audio: np.ndarray,
    model_audio: np.ndarray,
    sr: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute all F0 metrics for a single model.

    Args:
        original_audio: Original audio array.
        model_audio: Model audio array.
        sr: Sample rate.
        config: Full configuration dict.

    Returns:
        Dict with f0_correlation, f0_rmse, f0_dtw, voiced_ratio_orig,
        voiced_ratio_model, and status.
    """
    f0_cfg = config.get("f0", {})
    feat_cfg = config.get("features", {})
    min_hz = f0_cfg.get("min_hz", 50.0)
    max_hz = f0_cfg.get("max_hz", 600.0)
    hop_length = feat_cfg.get("hop_length", 256)
    use_octave = f0_cfg.get("octave_correction", True)

    try:
        f0_orig, vo_orig = extract_f0(original_audio, sr, min_hz, max_hz, hop_length)
        f0_model, vo_model = extract_f0(model_audio, sr, min_hz, max_hz, hop_length)

        if len(f0_orig) == 0 or len(f0_model) == 0:
            return _f0_error_result("F0 extraction produced no frames")

        # Align lengths by minimum
        min_len = min(len(f0_orig), len(f0_model))
        f0_orig = f0_orig[:min_len]
        f0_model = f0_model[:min_len]
        vo_orig = vo_orig[:min_len]
        vo_model = vo_model[:min_len]

        # Octave correction
        if use_octave:
            f0_model = octave_correction(f0_orig, f0_model, vo_orig, vo_model)

        # Combined voiced mask
        voiced_combined = vo_orig & vo_model

        # Compute metrics
        f0_corr = compute_f0_correlation(f0_orig, f0_model, voiced_combined)
        f0_rmse = compute_f0_rmse(f0_orig, f0_model, voiced_combined)
        f0_dtw = compute_f0_dtw(f0_orig, f0_model, voiced_combined)

        vr_orig = compute_voiced_ratio(vo_orig)
        vr_model = compute_voiced_ratio(vo_model)

        # Determine status
        failed = []
        if f0_corr is None:
            failed.append("correlation")
        if f0_rmse is None:
            failed.append("rmse")
        if f0_dtw is None:
            failed.append("dtw")

        status = "OK" if not failed else f"PARTIAL: {', '.join(failed)} failed"

        return {
            "f0_correlation": f0_corr,
            "f0_rmse": f0_rmse,
            "f0_dtw": f0_dtw,
            "voiced_ratio_original": vr_orig,
            "voiced_ratio_model": vr_model,
            "status": status,
        }

    except Exception as e:
        logger.warning("F0 computation failed: %s", e)
        return _f0_error_result(str(e))


def _f0_error_result(reason: str) -> Dict[str, Any]:
    """Create an error result dict with all F0 fields set to safe defaults."""
    return {
        "f0_correlation": None,
        "f0_rmse": None,
        "f0_dtw": None,
        "voiced_ratio_original": None,
        "voiced_ratio_model": None,
        "status": f"FAILED: {reason}",
    }


def compute_per_window_f0_scores(
    orig_windows: List[np.ndarray],
    model_windows: List[np.ndarray],
    sr: int,
    config: Dict[str, Any],
) -> List[float]:
    """Compute F0 scores for each window pair independently.

    For each window: correlation + RMSE + DTW → within-model normalize → composite score.
    These per-window scores are used for stability calculation.

    Returns:
        List of F0 scores (0–100) per window, NaN for failed windows.
    """
    f0_cfg = config.get("f0", {})
    feat_cfg = config.get("features", {})
    min_hz = f0_cfg.get("min_hz", 50.0)
    max_hz = f0_cfg.get("max_hz", 600.0)
    hop_length = feat_cfg.get("hop_length", 256)
    use_octave = f0_cfg.get("octave_correction", True)

    corr_w = f0_cfg.get("correlation_weight", 0.50)
    dtw_w = f0_cfg.get("dtw_weight", 0.30)
    rmse_w = f0_cfg.get("rmse_weight", 0.20)

    n = len(orig_windows)
    corr_vals = []
    rmse_vals = []
    dtw_vals = []

    for o_audio, m_audio in zip(orig_windows, model_windows):
        try:
            f0o, voo = extract_f0(o_audio, sr, min_hz, max_hz, hop_length)
            f0m, vom = extract_f0(m_audio, sr, min_hz, max_hz, hop_length)
            if len(f0o) == 0 or len(f0m) == 0:
                corr_vals.append(np.nan); rmse_vals.append(np.nan); dtw_vals.append(np.nan)
                continue

            min_len = min(len(f0o), len(f0m))
            f0o, f0m = f0o[:min_len], f0m[:min_len]
            voo, vom = voo[:min_len], vom[:min_len]

            if use_octave:
                f0m = octave_correction(f0o, f0m, voo, vom)

            voiced = voo & vom

            corr = compute_f0_correlation(f0o, f0m, voiced)
            rmse = compute_f0_rmse(f0o, f0m, voiced)
            dtw = compute_f0_dtw(f0o, f0m, voiced)

            corr_vals.append(corr if corr is not None else np.nan)
            rmse_vals.append(rmse if rmse is not None else np.nan)
            dtw_vals.append(dtw if dtw is not None else np.nan)
        except Exception:
            corr_vals.append(np.nan); rmse_vals.append(np.nan); dtw_vals.append(np.nan)

    # Within-model normalize
    corr_norm = _within_model_normalize(corr_vals, higher_is_better=True)
    rmse_norm = _within_model_normalize(rmse_vals, higher_is_better=False)
    dtw_norm = _within_model_normalize(dtw_vals, higher_is_better=False)

    scores = []
    for i in range(n):
        parts = []
        ws = []
        if not np.isnan(corr_norm[i]):
            parts.append(corr_norm[i]); ws.append(corr_w)
        if not np.isnan(dtw_norm[i]):
            parts.append(dtw_norm[i]); ws.append(dtw_w)
        if not np.isnan(rmse_norm[i]):
            parts.append(rmse_norm[i]); ws.append(rmse_w)
        if len(parts) == 0:
            scores.append(np.nan)
        else:
            total_w = sum(ws)
            scores.append(sum(p * w / total_w for p, w in zip(parts, ws)))

    return scores
