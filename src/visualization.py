"""
Visualization Module.

Generates:
  1. Ranking bar chart — models sorted by final score
  2. Top-5 radar chart — multi-dimensional comparison
"""

import logging
from typing import List, Dict, Any

import numpy as np

# Use non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

logger = logging.getLogger(__name__)

# Configure matplotlib for Chinese support
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def plot_ranking(
    results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """Generate a horizontal bar chart of model rankings.

    Args:
        results: List of per-model result dicts with scores.
        output_path: Path to save the PNG file.
    """
    models = [r["model_name"] for r in results]
    scores = [r["scores"]["final_score"] for r in results]
    warnings_list = [r["scores"].get("warnings", []) for r in results]

    # Sort by score descending
    pairs = sorted(
        zip(scores, models, warnings_list),
        key=lambda x: x[0] if not np.isnan(x[0]) else -1,
        reverse=True,
    )
    if not pairs:
        return
    scores_sorted, models_sorted, warnings_sorted = zip(*pairs)

    fig, ax = plt.subplots(figsize=(12, max(6, len(models_sorted) * 0.4)))

    colors = []
    for w in warnings_sorted:
        if any(sev in w for sev in ["SEVERE_CLIPPING", "SEVERE_SPECTRAL_ANOMALY"]):
            colors.append("#e74c3c")
        elif any(wrn in w for wrn in ["WARNING_CLIPPING", "HIGH_FREQUENCY_ANOMALY"]):
            colors.append("#f39c12")
        else:
            colors.append("#2ecc71")

    y_pos = range(len(models_sorted))
    bars = ax.barh(y_pos, scores_sorted, color=colors, edgecolor="white", height=0.6)

    for i, (bar, score) in enumerate(zip(bars, scores_sorted)):
        if not np.isnan(score):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{score:.1f}", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models_sorted, fontsize=9)
    ax.set_xlabel("Final Score (0-100)")
    ax.set_title("RVC Model Ranking — Final Score")
    clean_scores = [s for s in scores_sorted if not np.isnan(s)]
    x_max = max(clean_scores) * 1.1 if clean_scores else 100
    ax.set_xlim(0, max(105, x_max))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Ranking chart saved to %s", output_path)


def plot_radar_top5(
    results: List[Dict[str, Any]],
    output_path: str,
    top_n: int = 5,
) -> None:
    """Generate a radar chart for the top-N models.

    Shows 5 dimensions: ECAPA, Acoustic, F0, Stability, Artifact.
    All normalized to 0-100.
    """
    tech_sorted = sorted(
        results,
        key=lambda r: r["scores"]["final_score"]
        if not np.isnan(r["scores"]["final_score"]) else -1,
        reverse=True,
    )
    top_models = tech_sorted[:top_n]

    categories = ["ECAPA", "Acoustic", "F0", "Stability", "Artifact"]
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    palette = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    for idx, model in enumerate(top_models):
        s = model["scores"]
        vals = [
            _safe_score(s.get("ecapa_score")),
            _safe_score(s.get("acoustic_score")),
            _safe_score(s.get("f0_score")),
            _safe_score(s.get("stability_score")),
            _safe_score(s.get("artifact_score")),
        ]
        vals += vals[:1]

        color = palette[idx % len(palette)]
        ax.fill(angles, vals, alpha=0.1, color=color)
        ax.plot(angles, vals, "o-", linewidth=2, label=model["model_name"], color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)
    ax.set_title("Top Models — Multi-Dimensional Comparison", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Radar chart saved to %s", output_path)


def _safe_score(val) -> float:
    """Return score or 0 if NaN."""
    if val is None:
        return 0.0
    if np.isnan(val):
        return 0.0
    return float(val)
