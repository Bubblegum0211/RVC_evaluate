"""Trend analysis for detecting overfitting across RVC training checkpoints.

Compares model versions (e.g. epoch 10, 20, 30...) to identify when
evaluation scores begin to degrade — a key overfitting signal.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any

from utils import flatten_model_entry


class TrendAnalyzer:
    """Analyze score trends across multiple model checkpoints to detect overfitting.

    Workflow:
        1. Run evaluate.py once per checkpoint to generate scores.json files.
        2. Place each run's scores.json as model_<epoch>.json in a folder.
        3. Feed that folder to TrendAnalyzer to find the best epoch and
           identify overfitting regions.
    """

    def __init__(self, overfit_patience: int = 3, min_improvement: float = 0.3):
        """Args:
            overfit_patience: Consecutive declining checkpoints before flagging overfit.
            min_improvement: Minimum score drop (relative) to count as a decline.
        """
        self.overfit_patience = overfit_patience
        self.min_improvement = min_improvement

    def load_reports(self, folder: str) -> List[Dict[str, Any]]:
        """Load all model_*.json files from folder, sorted by epoch.

        Also handles a single scores.json (from evaluate.py output) by treating
        each model entry as a separate checkpoint.
        """
        folder = Path(folder)

        # Pattern 1: model_*.json files (one per checkpoint)
        reports = []
        for file in sorted(folder.glob("model_*.json")):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                reports.extend(data)
            else:
                reports.append(data)

        # Pattern 2: Single scores.json (all models in one file)
        scores_file = folder / "scores.json"
        if not reports and scores_file.exists():
            with open(scores_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            models = data.get("models", [data])
            reports = [flatten_model_entry(m) for m in models]

        # Sort by model name (handles numeric and non-numeric names)
        def _sort_key(entry):
            name = str(entry.get("model", "0"))
            try:
                return (0, int(name), name)
            except (ValueError, TypeError):
                return (1, 0, name)
        reports.sort(key=_sort_key)
        return reports

    def find_best_epoch(self, reports: List[Dict]) -> Dict[str, Any]:
        """Return the report with the highest overall score."""
        return max(reports, key=lambda x: x["score"])

    def analyze_trend(self, reports: List[Dict]) -> List[Dict[str, Any]]:
        """Extract a time-series of key metrics across checkpoints."""
        trend = []
        for item in reports:
            entry = {
                "epoch": item["model"],
                "score": item["score"],
                "ecapa": item.get("ecapa", 0),
                "f0": item.get("f0", 0),
                "artifact": item.get("artifact", 0),
            }
            trend.append(entry)
        return trend

    def detect_overfit(self, reports: List[Dict]) -> List[str]:
        """Find checkpoints where scores have declined for > overfit_patience steps.

        Returns list of model names where overfitting was detected.
        """
        decline_count = 0
        points = []

        for i in range(1, len(reports)):
            previous = reports[i - 1]
            current = reports[i]
            score_diff = current["score"] - previous["score"]

            if score_diff < -self.min_improvement:
                decline_count += 1
            else:
                decline_count = 0

            if decline_count >= self.overfit_patience:
                points.append(str(current["model"]))

        return points

    def explain(self, reports: List[Dict]) -> Dict[str, Any]:
        """Produce a human-readable summary: best epoch + overfit detection."""
        best = self.find_best_epoch(reports)
        overfit = self.detect_overfit(reports)

        result = {
            "best_epoch": best["model"],
            "best_score": best["score"],
            "overfit_points": overfit,
            "summary": [],
        }

        result["summary"].append(
            f"最佳模型为 epoch {best['model']} 综合评分 {best['score']:.2f}"
        )

        if overfit:
            result["summary"].append(f"检测到可能过拟合区域: {overfit}")
        else:
            result["summary"].append("未发现明显过拟合趋势")

        return result

    def save(self, data: Dict[str, Any], path: str) -> None:
        """Save analysis result as JSON."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except OSError as e:
            raise RuntimeError(f"Failed to save trend analysis to {path}: {e}") from e


# ---- CLI entry point ----
def main():
    parser = argparse.ArgumentParser(
        description="RVC Trend Analyzer — detect overfitting across model checkpoints"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Folder containing model_*.json or scores.json files",
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Consecutive declining checkpoints before overfit flag (default: 3)",
    )
    parser.add_argument(
        "--min-improvement", type=float, default=0.3,
        help="Minimum score drop to count as decline (default: 0.3)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save analysis result to JSON file",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"ERROR: Folder not found: {folder}")
        sys.exit(1)

    analyzer = TrendAnalyzer(
        overfit_patience=args.patience,
        min_improvement=args.min_improvement,
    )

    reports = analyzer.load_reports(str(folder))
    if not reports:
        print(f"ERROR: No model reports found in {folder}")
        sys.exit(1)

    print(f"Loaded {len(reports)} checkpoint(s)")

    # Trend analysis
    trend = analyzer.analyze_trend(reports)
    print("\n=== Score Trend ===")
    for t in trend:
        bar = "|" * max(1, int(t["score"] / 5))
        print(f"  {t['epoch']:>6}  {t['score']:>6.2f}  {bar}")

    # Overfit detection
    result = analyzer.explain(reports)
    print(f"\n=== Summary ===")
    for line in result["summary"]:
        print(f"  {line}")

    if args.output:
        result["trend"] = trend
        analyzer.save(result, args.output)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
