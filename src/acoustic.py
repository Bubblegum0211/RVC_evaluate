"""
Acoustic / Timbre Analysis Module.

Computes three distinct acoustic distances using DTW alignment:
  1. MCD (Mel-Cepstral Distortion) — true cepstral distance
  2. Mel Spectral Distance — log-Mel spectrogram distance
  3. MFCC Distance — MFCC vector distance

Weight: 30% of final score.
Acoustic_score = 0.45 * MCD_score + 0.35 * Mel_score + 0.20 * MFCC_score
"""

import logging
from typing import Dict, Any, List, Tuple

try:
    from utils import within_model_normalize as _within_model_normalize, dtw_align, MCD_SCALE
except ImportError:
    from .utils import within_model_normalize as _within_model_normalize, dtw_align, MCD_SCALE

import numpy as np
import librosa
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def extract_mfcc(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 20,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    fmin: float = 50.0,
    fmax: float = 7600.0,
    include_c0: bool = False,
) -> np.ndarray:
    """Extract MFCC features.

    Args:
        audio: Audio array.
        sr: Sample rate.
        n_mfcc: Number of MFCC coefficients.
        n_fft: FFT size.
        hop_length: Hop length.
        win_length: Window length.
        fmin: Minimum frequency.
        fmax: Maximum frequency.
        include_c0: Whether to include the 0th coefficient.

    Returns:
        MFCC matrix of shape (n_frames, n_mfcc) or (n_frames, n_mfcc + 1) if include_c0.
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc + (1 if include_c0 else 0),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
    )
    if include_c0:
        return mfcc.T  # Shape: (n_frames, n_mfcc + 1)
    return mfcc[1:].T if mfcc.shape[0] > 1 else mfcc.T  # Shape: (n_frames, n_mfcc)


def extract_log_mel(
    audio: np.ndarray,
    sr: int,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    fmin: float = 50.0,
    fmax: float = 7600.0,
) -> np.ndarray:
    """Extract log-Mel spectrogram.

    Args:
        audio: Audio array.
        sr: Sample rate.
        n_mels: Number of mel bands.
        n_fft: FFT size.
        hop_length: Hop length.
        win_length: Window length.
        fmin: Minimum frequency.
        fmax: Maximum frequency.

    Returns:
        Log-Mel spectrogram of shape (n_frames, n_mels).
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    return log_mel.T


def compute_mcd(
    original_audio: np.ndarray,
    model_audio: np.ndarray,
    sr: int,
    n_mfcc: int = 24,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    fmin: float = 50.0,
    fmax: float = 7600.0,
) -> float:
    """Compute Mel-Cepstral Distortion (MCD) between two audio signals.

    MCD uses mel-frequency cepstral coefficients (MFCCs).
    Formula: MCD = (10 * sqrt(2) / ln(10)) * sqrt( Σ(c_i - c'_i)^2 )

    Uses DTW to align frames before computing distance.
    Distance is divided by DTW path length.

    Excludes c0 (the energy coefficient).

    Returns:
        MCD value (lower is better).
    """
    # Extract MFCCs (excluding c0)
    mfcc_orig = extract_mfcc(
        original_audio, sr, n_mfcc=n_mfcc,
        n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        fmin=fmin, fmax=fmax, include_c0=False,
    )
    mfcc_model = extract_mfcc(
        model_audio, sr, n_mfcc=n_mfcc,
        n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        fmin=fmin, fmax=fmax, include_c0=False,
    )

    if mfcc_orig.shape[0] < 2 or mfcc_model.shape[0] < 2:
        return float('nan')

    # DTW alignment
    path_a, path_b, _ = dtw_align(mfcc_orig, mfcc_model, metric="euclidean")

    if len(path_a) == 0:
        return float('nan')

    aligned_orig = mfcc_orig[path_a]
    aligned_model = mfcc_model[path_b]

    # MCD formula: (10 * sqrt(2) / ln(10)) * mean( sqrt( Σ(c_i - c'_i)^2 ) )
    # Per-frame squared differences
    diff = aligned_orig - aligned_model
    per_frame_dist = np.sqrt(np.sum(diff ** 2, axis=1))

    # MCD scaling factor
    mcd = MCD_SCALE * np.mean(per_frame_dist)

    return float(mcd)


def compute_mel_distance(
    original_audio: np.ndarray,
    model_audio: np.ndarray,
    sr: int,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    fmin: float = 50.0,
    fmax: float = 7600.0,
) -> float:
    """Compute Mel Spectral Distance between two audio signals.

    Compares log-Mel spectrograms with DTW alignment.
    Distance is normalized by DTW path length.

    Returns:
        Mel distance (lower is better).
    """
    mel_orig = extract_log_mel(
        original_audio, sr, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        fmin=fmin, fmax=fmax,
    )
    mel_model = extract_log_mel(
        model_audio, sr, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        fmin=fmin, fmax=fmax,
    )

    if mel_orig.shape[0] < 2 or mel_model.shape[0] < 2:
        return float('nan')

    # DTW alignment
    path_a, path_b, normalized = dtw_align(mel_orig, mel_model, metric="euclidean")

    return float(normalized)


def compute_mfcc_distance(
    original_audio: np.ndarray,
    model_audio: np.ndarray,
    sr: int,
    n_mfcc: int = 20,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    fmin: float = 50.0,
    fmax: float = 7600.0,
) -> float:
    """Compute MFCC Distance between two audio signals.

    Compares MFCC vectors (including c0) with DTW alignment.
    This is DIFFERENT from MCD — it's a simple Euclidean distance
    on MFCC vectors.

    Distance is normalized by DTW path length.

    Returns:
        MFCC distance (lower is better).
    """
    mfcc_orig = extract_mfcc(
        original_audio, sr, n_mfcc=n_mfcc,
        n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        fmin=fmin, fmax=fmax, include_c0=True,
    )
    mfcc_model = extract_mfcc(
        model_audio, sr, n_mfcc=n_mfcc,
        n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        fmin=fmin, fmax=fmax, include_c0=True,
    )

    if mfcc_orig.shape[0] < 2 or mfcc_model.shape[0] < 2:
        return float('nan')

    # DTW alignment
    path_a, path_b, normalized = dtw_align(mfcc_orig, mfcc_model, metric="euclidean")

    return float(normalized)


def compute_acoustic_scores(
    original_audio: np.ndarray,
    model_audio: np.ndarray,
    sr: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute all acoustic metrics for a single model.

    Args:
        original_audio: Original audio array.
        model_audio: Model audio array.
        sr: Sample rate.
        config: Full configuration dict.

    Returns:
        Dict with mcd_distance, mel_distance, mfcc_distance, and status.
    """
    feat_cfg = config.get("features", {})

    try:
        mcd_val = compute_mcd(
            original_audio, model_audio, sr,
            n_mfcc=24,  # Standard for MCD
            n_fft=feat_cfg.get("n_fft", 1024),
            hop_length=feat_cfg.get("hop_length", 256),
            win_length=feat_cfg.get("win_length", 1024),
            fmin=feat_cfg.get("fmin", 50.0),
            fmax=feat_cfg.get("fmax", 7600.0),
        )
    except Exception as e:
        logger.warning("MCD computation failed: %s", e)
        mcd_val = None

    try:
        mel_val = compute_mel_distance(
            original_audio, model_audio, sr,
            n_mels=feat_cfg.get("n_mels", 80),
            n_fft=feat_cfg.get("n_fft", 1024),
            hop_length=feat_cfg.get("hop_length", 256),
            win_length=feat_cfg.get("win_length", 1024),
            fmin=feat_cfg.get("fmin", 50.0),
            fmax=feat_cfg.get("fmax", 7600.0),
        )
    except Exception as e:
        logger.warning("Mel distance computation failed: %s", e)
        mel_val = None

    try:
        mfcc_val = compute_mfcc_distance(
            original_audio, model_audio, sr,
            n_mfcc=feat_cfg.get("n_mfcc", 20),
            n_fft=feat_cfg.get("n_fft", 1024),
            hop_length=feat_cfg.get("hop_length", 256),
            win_length=feat_cfg.get("win_length", 1024),
            fmin=feat_cfg.get("fmin", 50.0),
            fmax=feat_cfg.get("fmax", 7600.0),
        )
    except Exception as e:
        logger.warning("MFCC distance computation failed: %s", e)
        mfcc_val = None

    status_parts = []
    if mcd_val is None:
        status_parts.append("MCD_FAILED")
    if mel_val is None:
        status_parts.append("MEL_FAILED")
    if mfcc_val is None:
        status_parts.append("MFCC_FAILED")

    return {
        "mcd_distance": mcd_val,
        "mel_distance": mel_val,
        "mfcc_distance": mfcc_val,
        "status": "; ".join(status_parts) if status_parts else "OK",
    }


def compute_per_window_acoustic_scores(
    orig_windows: List[np.ndarray],
    model_windows: List[np.ndarray],
    sr: int,
    config: Dict[str, Any],
) -> List[float]:
    """Compute acoustic scores for each window pair independently.

    For each window: MCD + Mel + MFCC → within-model normalize → composite score.
    These per-window scores are used for stability calculation.

    Returns:
        List of acoustic scores (0–100) per window, NaN for failed windows.
    """
    feat_cfg = config.get("features", {})
    ac_cfg = config.get("acoustic", {})
    mcd_w = ac_cfg.get("mcd_weight", 0.45)
    mel_w = ac_cfg.get("mel_weight", 0.35)
    mfcc_w = ac_cfg.get("mfcc_weight", 0.20)

    n_fft = feat_cfg.get("n_fft", 1024)
    hop = feat_cfg.get("hop_length", 256)
    win_len = feat_cfg.get("win_length", 1024)
    fmin = feat_cfg.get("fmin", 50.0)
    fmax = feat_cfg.get("fmax", 7600.0)

    n = len(orig_windows)
    mcd_vals = []
    mel_vals = []
    mfcc_vals = []

    for o_audio, m_audio in zip(orig_windows, model_windows):
        try:
            mcd_vals.append(compute_mcd(o_audio, m_audio, sr, n_mfcc=24,
                                        n_fft=n_fft, hop_length=hop, win_length=win_len,
                                        fmin=fmin, fmax=fmax))
        except Exception:
            mcd_vals.append(np.nan)
        try:
            mel_vals.append(compute_mel_distance(o_audio, m_audio, sr, n_mels=feat_cfg.get("n_mels", 80),
                                                  n_fft=n_fft, hop_length=hop, win_length=win_len,
                                                  fmin=fmin, fmax=fmax))
        except Exception:
            mel_vals.append(np.nan)
        try:
            mfcc_vals.append(compute_mfcc_distance(o_audio, m_audio, sr, n_mfcc=feat_cfg.get("n_mfcc", 20),
                                                    n_fft=n_fft, hop_length=hop, win_length=win_len,
                                                    fmin=fmin, fmax=fmax))
        except Exception:
            mfcc_vals.append(np.nan)

    # Within-model normalize
    mcd_norm = _within_model_normalize(mcd_vals, higher_is_better=False)
    mel_norm = _within_model_normalize(mel_vals, higher_is_better=False)
    mfcc_norm = _within_model_normalize(mfcc_vals, higher_is_better=False)

    scores = []
    for i in range(n):
        parts = []
        ws = []
        if not np.isnan(mcd_norm[i]):
            parts.append(mcd_norm[i]); ws.append(mcd_w)
        if not np.isnan(mel_norm[i]):
            parts.append(mel_norm[i]); ws.append(mel_w)
        if not np.isnan(mfcc_norm[i]):
            parts.append(mfcc_norm[i]); ws.append(mfcc_w)
        if len(parts) == 0:
            scores.append(np.nan)
        else:
            total_w = sum(ws)
            scores.append(sum(p * w / total_w for p, w in zip(parts, ws)))

    return scores
