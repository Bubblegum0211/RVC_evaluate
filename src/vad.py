"""
Voice Activity Detection (VAD) module.

Detects speech segments and marks invalid (low-speech) windows.
Does NOT modify the original audio time structure.
"""

import logging
from typing import List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_speech_ratio(
    audio: np.ndarray,
    sr: int,
    energy_threshold: float = 0.01,
    top_db: float = 30.0,
) -> float:
    """Compute the ratio of speech frames in an audio window.

    Uses energy-based VAD: frames with energy above a threshold relative
    to the maximum energy are considered speech.

    Args:
        audio: Audio array.
        sr: Sample rate.
        energy_threshold: Fraction of max energy for speech detection.
        top_db: dB threshold below peak for silence detection (librosa style).

    Returns:
        Speech ratio (0.0 to 1.0).
    """
    if len(audio) == 0:
        return 0.0

    # Use librosa's split-like approach: energy-based
    ref = np.max(np.abs(audio))
    if ref < 1e-10:
        return 0.0

    # Convert top_db to linear threshold
    db_threshold = 10 ** (-top_db / 20.0)
    threshold = max(ref * db_threshold, energy_threshold)

    # Simple frame-based energy detection
    frame_length = int(sr * 0.025)  # 25ms frames
    hop_length = int(sr * 0.010)    # 10ms hop

    if frame_length == 0:
        return 0.0

    frames = _frame_audio(audio, frame_length, hop_length)
    if len(frames) == 0:
        return 0.0

    energies = np.max(np.abs(frames), axis=1)
    speech_frames = np.sum(energies > threshold)
    return speech_frames / len(frames)


def _frame_audio(
    audio: np.ndarray,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    """Split audio into overlapping frames."""
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    if n_frames <= 0:
        return np.array([])

    frames = np.zeros((n_frames, frame_length))
    for i in range(n_frames):
        start = i * hop_length
        frames[i] = audio[start:start + frame_length]
    return frames


def filter_valid_windows(
    windows: List[Dict[str, Any]],
    sr: int,
    min_speech_ratio: float = 0.05,
) -> List[Dict[str, Any]]:
    """Mark windows with speech ratio info and filter out low-speech windows.

    Args:
        windows: List of window dicts from audio.split_into_windows.
        sr: Sample rate.
        min_speech_ratio: Minimum speech ratio for a window to be valid.

    Returns:
        Same list with 'speech_ratio' and 'valid' fields added.
        The list is NOT filtered — all windows are returned with validity flags.
    """
    for w in windows:
        ratio = compute_speech_ratio(w["audio"], sr)
        w["speech_ratio"] = ratio
        w["valid"] = ratio >= min_speech_ratio

    valid_count = sum(1 for w in windows if w["valid"])
    total_count = len(windows)
    logger.info(
        "VAD: %d/%d windows valid (min_speech_ratio=%.2f)",
        valid_count, total_count, min_speech_ratio,
    )

    return windows


def get_overall_speech_ratio(windows: List[Dict[str, Any]]) -> float:
    """Compute the overall speech ratio across all windows."""
    if not windows:
        return 0.0
    ratios = [w.get("speech_ratio", 0.0) for w in windows]
    return float(np.mean(ratios))
