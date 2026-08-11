"""
Artifact / Audio Quality Detection Module.

Detects audio artifacts and quality issues using raw (un-normalized) audio:
  - Clipping ratio
  - Peak amplitude
  - RMS energy
  - Spectral flatness
  - High-frequency energy ratio
  - Harmonic / spectral anomalies

Includes ABSOLUTE threshold warnings (not just relative batch normalization).
Severe issues generate warnings that affect Recommended Rank.

Weight: 5% of final score.
"""

import logging
from typing import Dict, Any, List

import numpy as np
import librosa

logger = logging.getLogger(__name__)


# Absolute thresholds for artifact detection
CLIP_THRESHOLD = 0.01       # >1% frames clipped → WARNING
SEVERE_CLIP_THRESHOLD = 0.05  # >5% → SEVERE
HF_ANOMALY_THRESHOLD = 0.20  # >20% HF energy -> possible metallic/buzzy RVC artifact
SPECTRAL_FLATNESS_LOW = 0.02  # Below 0.02 -> overly tonal; RVC should maintain some spectral richness
LOW_SPEECH_RATIO_THRESHOLD = 0.10  # <10% speech → WARNING


def compute_clipping_ratio(
    audio: np.ndarray,
    threshold: float = 0.999,
) -> float:
    """Compute the ratio of samples that are clipped (near ±1.0).

    IMPORTANT: Must use raw audio (not normalized) to detect real clipping.

    Args:
        audio: Raw audio array (no normalization applied).
        threshold: Amplitude threshold for clipping detection.

    Returns:
        Ratio of clipped samples (0.0 to 1.0).
    """
    if len(audio) == 0:
        return 0.0

    clipped = np.sum(np.abs(audio) >= threshold)

    return float(clipped) / len(audio)


def compute_peak(audio: np.ndarray) -> float:
    """Compute peak amplitude."""
    if len(audio) == 0:
        return 0.0
    return float(np.max(np.abs(audio)))


def compute_rms(audio: np.ndarray) -> float:
    """Compute RMS energy."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))


def compute_spectral_flatness(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> float:
    """Compute spectral flatness (Wiener entropy).

    High flatness = noise-like, Low flatness = tonal.
    RVC artifacts often produce unusual spectral flatness.

    Args:
        audio: Audio array.
        sr: Sample rate.
        n_fft: FFT size.
        hop_length: Hop length.

    Returns:
        Mean spectral flatness across frames.
    """
    if len(audio) < n_fft:
        return float('nan')
    try:
        flatness = librosa.feature.spectral_flatness(
            y=audio, n_fft=n_fft, hop_length=hop_length,
        )
        return float(np.mean(flatness))
    except Exception:
        return float('nan')


def compute_high_frequency_ratio(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float = 4000.0,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> float:
    """Compute the ratio of energy in high frequencies.

    High HF ratio may indicate RVC artifacts (metallic/buzzy quality).

    Args:
        audio: Audio array.
        sr: Sample rate.
        cutoff_hz: Frequency above which is considered "high".
        n_fft: FFT size.
        hop_length: Hop length.

    Returns:
        Ratio of high-frequency energy (0.0 to 1.0).
    """
    if len(audio) < n_fft:
        return float('nan')

    try:
        stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        hf_bins = freqs >= cutoff_hz
        if not np.any(hf_bins):
            return 0.0

        total_energy = np.sum(stft ** 2)
        if total_energy < 1e-10:
            return 0.0

        hf_energy = np.sum(stft[hf_bins] ** 2)
        return float(hf_energy / total_energy)
    except Exception:
        return 0.0


def compute_harmonic_anomaly(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> float:
    """Proxy for harmonic distortion detection.

    Compares harmonic vs percussive energy ratio.
    Unusual ratios may indicate artifacts.

    Returns:
        Harmonic-to-percussive energy ratio anomaly score.
    """
    if len(audio) < n_fft:
        return float('nan')
    try:
        harmonic, percussive = librosa.effects.hpss(audio)
        h_energy = np.sum(harmonic ** 2)
        p_energy = np.sum(percussive ** 2)
        total = h_energy + p_energy
        if total < 1e-10:
            return 0.0
        return float(h_energy / total)
    except Exception:
        return 0.0


def detect_absolute_warnings(
    clipping_ratio: float,
    high_freq_ratio: float,
    spectral_flatness: float,
    speech_ratio: float,
) -> List[str]:
    """Detect absolute artifact warnings based on fixed thresholds.

    This is SEPARATE from the relative batch normalization scoring.
    Even if all models have the same issue, severe cases get warnings.

    Args:
        clipping_ratio: Clipping ratio.
        high_freq_ratio: High frequency energy ratio.
        spectral_flatness: Spectral flatness.
        speech_ratio: Overall speech ratio.

    Returns:
        List of warning strings.
    """
    warnings = []

    if clipping_ratio >= SEVERE_CLIP_THRESHOLD:
        warnings.append("SEVERE_CLIPPING")
    elif clipping_ratio >= CLIP_THRESHOLD:
        warnings.append("WARNING_CLIPPING")

    if high_freq_ratio >= HF_ANOMALY_THRESHOLD:
        warnings.append("HIGH_FREQUENCY_ANOMALY")

    if spectral_flatness < SPECTRAL_FLATNESS_LOW:
        warnings.append("SEVERE_SPECTRAL_ANOMALY")

    if speech_ratio < LOW_SPEECH_RATIO_THRESHOLD:
        warnings.append("LOW_SPEECH_RATIO")

    return warnings


def compute_artifact_metrics(
    audio: np.ndarray,
    sr: int,
) -> Dict[str, Any]:
    """Compute all artifact metrics for a single model using RAW audio.

    Args:
        audio: Raw audio array (no normalization).
        sr: Sample rate of the audio.

    Returns:
        Dict with all artifact metrics.
    """
    n_fft = 1024
    hop_length = 256

    try:
        clip = compute_clipping_ratio(audio)
    except Exception:
        clip = None

    try:
        pk = compute_peak(audio)
        rms_val = compute_rms(audio)
    except Exception:
        pk = None
        rms_val = None

    try:
        sf = compute_spectral_flatness(audio, sr, n_fft, hop_length)
    except Exception:
        sf = None

    try:
        hf = compute_high_frequency_ratio(audio, sr, n_fft=n_fft, hop_length=hop_length)
    except Exception:
        hf = None

    try:
        harm = compute_harmonic_anomaly(audio, sr, n_fft, hop_length)
    except Exception:
        harm = None

    return {
        "clipping_ratio": clip,
        "peak": pk,
        "rms": rms_val,
        "spectral_flatness": sf,
        "high_frequency_ratio": hf,
        "harmonic_ratio": harm,
        "status": "OK",
    }
