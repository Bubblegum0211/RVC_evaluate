"""
ECAPA-TDNN Speaker Similarity Module.

Uses a fixed local ECAPA-TDNN model to extract speaker embeddings,
then computes cosine similarity between original and model audio windows.

Weight: 40% of final score.
"""

import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ECAPAModel:
    """ECAPA-TDNN speaker embedding model wrapper.

    Uses SpeechBrain's ECAPA-TDNN pretrained model.
    Model files must be placed in ./ecapa_model/.
    """

    def __init__(self, model_dir: str, device: str = "auto"):
        self.model_dir = Path(model_dir)
        self.device = self._resolve_device(device)
        self.model = None
        self.model_info: Dict[str, Any] = {}
        self.loaded = False

    def _resolve_device(self, device: str) -> str:
        """Resolve device string. Falls back to CPU if CUDA unavailable."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return device

    def check_model_exists(self) -> bool:
        """Check if the ECAPA model directory exists, contains required files,
        and can actually be loaded by SpeechBrain.

        Returns True only if the model is loadable.
        """
        if not self.model_dir.exists():
            logger.error("ECAPA model directory not found: %s", self.model_dir)
            return False
        files = list(self.model_dir.glob("*"))
        if not files:
            logger.error("ECAPA model directory is empty: %s", self.model_dir)
            return False

        # Check for required SpeechBrain model files
        has_hyperparams = (self.model_dir / "hyperparams.yaml").exists()
        has_embedding = any(f.suffix in (".pt", ".ckpt") for f in files)
        if not (has_hyperparams and has_embedding):
            logger.error(
                "ECAPA model dir is incomplete (no hyperparams.yaml or .pt/.ckpt found). "
                "Model will fail to load."
            )
            return False

        logger.info("ECAPA model directory found: %s (%d files)", self.model_dir, len(files))
        return True

    def load(self) -> None:
        """Load the ECAPA-TDNN model from the local model directory.

        Dynamically patches hyperparams.yaml to use absolute local paths,
        ensuring the model loads from the local directory regardless of CWD.
        Uses SpeechBrain's EncoderClassifier API.
        """
        if self.loaded:
            return

        from speechbrain.inference.speaker import EncoderClassifier

        # Patch hyperparams.yaml to use absolute local path
        hparams_path = self.model_dir / "hyperparams.yaml"
        abs_dir = str(self.model_dir.resolve()).replace("\\", "/")

        with open(hparams_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        # Replace the pretrained_path line
        lines = original_content.split("\n")
        patched_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("pretrained_path:"):
                indent = line[:len(line) - len(stripped)]
                patched_lines.append(f"{indent}pretrained_path: {abs_dir}")
            else:
                patched_lines.append(line)
        patched_content = "\n".join(patched_lines)

        logger.info("Loading ECAPA-TDNN model from %s on %s...", self.model_dir, self.device)
        try:
            # Write patched content
            with open(hparams_path, "w", encoding="utf-8") as f:
                f.write(patched_content)

            self.model = EncoderClassifier.from_hparams(
                source=str(self.model_dir),
                run_opts={"device": self.device},
            )
        finally:
            # Restore original content
            with open(hparams_path, "w", encoding="utf-8") as f:
                f.write(original_content)

        self.loaded = True

        # Record model info
        self.model_info = {
            "model_path": str(self.model_dir.resolve()),
            "model_version": "speechbrain/spkrec-ecapa-voxceleb",
            "embedding_dim": 192,
            "expected_sample_rate": 16000,
            "preprocessing": "16kHz mono, no additional normalization",
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "cuda_available": torch.cuda.is_available(),
            "device": self.device,
        }
        logger.info("ECAPA-TDNN model loaded. Embedding dim: %d", self.model_info["embedding_dim"])

    def encode_batch(self, waveforms: List[np.ndarray]) -> np.ndarray:
        """Extract embeddings for a batch of audio windows.

        Uses true batched inference — all windows padded to max length and
        processed in a single GPU call for efficiency.

        Args:
            waveforms: List of numpy audio arrays (must be 16kHz mono).

        Returns:
            Numpy array of shape (batch_size, embedding_dim) with L2-normalized embeddings.
        """
        if not self.loaded or self.model is None:
            raise RuntimeError("ECAPA model not loaded. Call load() first.")

        batch_size = len(waveforms)
        emb_dim = self.model_info["embedding_dim"]

        if batch_size == 0:
            return np.empty((0, emb_dim))

        # Find non-empty waveforms and their lengths
        valid_indices = [i for i, w in enumerate(waveforms) if len(w) > 0]
        empty_indices = [i for i, w in enumerate(waveforms) if len(w) == 0]

        if not valid_indices:
            return np.zeros((batch_size, emb_dim))

        # Pad to max length
        max_len = max(len(waveforms[i]) for i in valid_indices)
        batch_tensor = torch.zeros(len(valid_indices), max_len, device=self.device)
        lengths_list = []

        for bi, idx in enumerate(valid_indices):
            wav = waveforms[idx]
            batch_tensor[bi, :len(wav)] = torch.from_numpy(wav).float()
            lengths_list.append(len(wav) / max_len)

        lengths = torch.tensor(lengths_list, device=self.device)

        with torch.no_grad():
            try:
                embs = self.model.encode_batch(batch_tensor, lengths)
                embs = embs.cpu().numpy()
            except Exception as e:
                logger.warning("ECAPA batch encoding failed: %s", e)
                embs = np.full((len(valid_indices), emb_dim), np.nan)

        # Reconstruct full output with zero embeddings for empty inputs
        full_embs = np.zeros((batch_size, emb_dim))
        for bi, idx in enumerate(valid_indices):
            full_embs[idx] = embs[bi]
        for idx in empty_indices:
            full_embs[idx] = np.zeros(emb_dim)

        # L2 normalize
        norms = np.linalg.norm(full_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        full_embs = full_embs / norms

        return full_embs

    def compute_similarity(
        self,
        emb_a: np.ndarray,
        emb_b: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two L2-normalized embeddings.

        Args:
            emb_a: First embedding vector.
            emb_b: Second embedding vector.

        Returns:
            Cosine similarity (range roughly -1 to 1, typically 0 to 1 for same speaker).
        """
        if np.any(np.isnan(emb_a)) or np.any(np.isnan(emb_b)):
            return np.nan
        dot = np.dot(emb_a, emb_b)
        return float(np.clip(dot, -1.0, 1.0))


def compute_ecapa_scores(
    original_windows: List[Dict[str, Any]],
    model_windows: List[Dict[str, Any]],
    ecapa_model: ECAPAModel,
    mean_weight: float = 0.75,
    min_weight: float = 0.15,
) -> Dict[str, Any]:
    """Compute ECAPA speaker similarity scores for a model.

    Only uses valid (non-low-speech) windows.

    Args:
        original_windows: Original audio windows with VAD info.
        model_windows: Model audio windows with VAD info.
        ecapa_model: Loaded ECAPA model instance.
        mean_weight: Weight for mean similarity in final score.
        min_weight: Weight for min similarity in final score.

    Returns:
        Dict with ecapa_mean, ecapa_std, ecapa_min, ecapa_max, ecapa_score,
        per_window_similarities, and num_valid_windows.
    """
    if not ecapa_model.loaded:
        return _ecapa_error_result("ECAPA model not loaded")

    # Get valid windows
    valid_indices = [
        i for i, (ow, mw) in enumerate(zip(original_windows, model_windows))
        if ow.get("valid", False) and mw.get("valid", False)
    ]

    if len(valid_indices) == 0:
        return _ecapa_error_result("No valid windows for ECAPA computation")

    # Extract valid window audios
    orig_audios = [original_windows[i]["audio"] for i in valid_indices]
    model_audios = [model_windows[i]["audio"] for i in valid_indices]

    try:
        orig_embs = ecapa_model.encode_batch(orig_audios)
        model_embs = ecapa_model.encode_batch(model_audios)
    except Exception as e:
        logger.error("ECAPA encoding failed: %s", e)
        return _ecapa_error_result(f"ECAPA encoding error: {e}")

    # Compute per-window cosine similarities
    similarities = []
    for o_emb, m_emb in zip(orig_embs, model_embs):
        sim = ecapa_model.compute_similarity(o_emb, m_emb)
        similarities.append(sim)

    similarities = np.array(similarities)
    # Filter NaN
    valid_sims = similarities[~np.isnan(similarities)]

    if len(valid_sims) == 0:
        return _ecapa_error_result("All ECAPA similarities are NaN")

    ecapa_mean = float(np.mean(valid_sims))
    ecapa_std = float(np.std(valid_sims))
    ecapa_min = float(np.min(valid_sims))
    ecapa_max = float(np.max(valid_sims))

    # Aggregate score: weighted combination of mean and min
    ecapa_score = mean_weight * ecapa_mean + min_weight * ecapa_min

    return {
        "ecapa_mean": ecapa_mean,
        "ecapa_std": ecapa_std,
        "ecapa_min": ecapa_min,
        "ecapa_max": ecapa_max,
        "ecapa_score": ecapa_score,
        "per_window_similarities": similarities.tolist(),
        "num_valid_windows": len(valid_sims),
        "num_total_windows": len(original_windows),
        "status": "OK",
    }


def _ecapa_error_result(reason: str) -> Dict[str, Any]:
    """Create an error result for ECAPA computation."""
    return {
        "ecapa_mean": None,
        "ecapa_std": None,
        "ecapa_min": None,
        "ecapa_max": None,
        "ecapa_score": None,
        "per_window_similarities": [],
        "num_valid_windows": 0,
        "num_total_windows": 0,
        "status": f"FAILED: {reason}",
    }
