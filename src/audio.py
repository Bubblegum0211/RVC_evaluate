"""
Audio loading, preprocessing, and windowing module.

Provides two processing pipelines:
  - Analysis Pipeline: resample, mono, optional normalization (for ECAPA/MCD/Mel/MFCC/F0)
  - Raw Quality Pipeline: minimal processing (for artifact detection)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


def load_audio(
    path: str,
    target_sr: int = 16000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load an audio file, resample if needed, and optionally convert to mono.

    Args:
        path: Path to the WAV file.
        target_sr: Target sample rate.
        mono: If True, convert to mono.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    audio, sr = librosa.load(path, sr=target_sr, mono=mono)
    return audio, sr


def load_audio_raw(path: str) -> Tuple[np.ndarray, int]:
    """Load audio with minimal processing for raw quality analysis.

    Keeps original amplitude — no normalization.

    Args:
        path: Path to the WAV file.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Convert to mono by averaging channels
    return audio.astype(np.float32), sr


def get_duration(audio: np.ndarray, sr: int) -> float:
    """Get audio duration in seconds."""
    return len(audio) / sr


def trim_to_common_duration(
    original: np.ndarray,
    model: np.ndarray,
    sr: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trim both audios to the shorter duration (alignment_mode: min_duration).

    Args:
        original: Original audio array.
        model: Model audio array.
        sr: Sample rate.

    Returns:
        Tuple of (trimmed_original, trimmed_model).
    """
    min_len = min(len(original), len(model))
    return original[:min_len], model[:min_len]


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Peak normalize audio to [-1, 1].

    Returns original if max amplitude is 0.
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def split_into_windows(
    audio: np.ndarray,
    sr: int,
    window_seconds: float = 3.0,
    drop_incomplete: bool = True,
    min_valid_seconds: float = 2.0,
) -> List[Dict[str, Any]]:
    """Split audio into fixed-length windows.

    Args:
        audio: Audio array.
        sr: Sample rate.
        window_seconds: Window length in seconds.
        drop_incomplete: If True, drop the last incomplete window.
        min_valid_seconds: Minimum window duration to keep.

    Returns:
        List of dicts with 'audio', 'start_sample', 'end_sample', 'duration'.
    """
    window_samples = int(window_seconds * sr)
    total_samples = len(audio)
    windows = []

    for start in range(0, total_samples, window_samples):
        end = min(start + window_samples, total_samples)
        window_audio = audio[start:end]
        duration = len(window_audio) / sr

        if drop_incomplete and duration < min_valid_seconds:
            continue

        windows.append({
            "audio": window_audio,
            "start_sample": start,
            "end_sample": end,
            "duration": duration,
            "index": len(windows),
        })

    return windows


def load_model_files(models_dir: str) -> List[Path]:
    """Auto-scan models directory for .wav files (excluding original.wav).

    Args:
        models_dir: Path to the models directory.

    Returns:
        Sorted list of Path objects for model WAV files.
    """
    model_dir = Path(models_dir)
    if not model_dir.exists():
        logger.error("Models directory not found: %s", models_dir)
        return []

    wav_files = sorted(
        model_dir.glob("*.wav"),
        key=lambda p: _extract_model_number(p.stem),
    )
    logger.info("Found %d model WAV files in %s", len(wav_files), models_dir)
    return wav_files


def _extract_model_number(stem: str) -> int:
    """Extract numeric prefix from filename for sorting.

    E.g., '1', '1_renamed', '01' → 1.
    Non-numeric names (e.g. 'best_model') get infinity → sorted last.
    """
    import re
    match = re.match(r'(\d+)', stem)
    if match:
        return int(match.group(1))
    return float('inf')


def prepare_analysis_audio(
    path: str,
    sr: int,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """Prepare audio for analysis pipeline (ECAPA, MCD, F0, etc.).

    Loads, resamples, converts to mono, optionally normalizes.
    Never modifies the original WAV file on disk.
    """
    audio, actual_sr = load_audio(path, target_sr=sr, mono=True)
    if normalize:
        audio = normalize_audio(audio)
    return audio, actual_sr


def prepare_raw_audio(path: str, sr: int) -> Tuple[np.ndarray, int]:
    """Prepare audio for raw quality pipeline (clipping, peak, RMS, etc.).

    Loads with original amplitude — no normalization.
    """
    audio, actual_sr = load_audio_raw(path)
    if actual_sr != sr:
        logger.info("Resampling raw audio %s from %d Hz to %d Hz", path, actual_sr, sr)
        audio = librosa.resample(audio, orig_sr=actual_sr, target_sr=sr)
        actual_sr = sr
    return audio, actual_sr
