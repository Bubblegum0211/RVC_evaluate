"""A/B comparison of two RVC voice models across all evaluation dimensions.

Generates a detailed side-by-side report showing per-metric differences,
winners, and natural-language analysis.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any

from utils import flatten_model_entry


class ModelComparator:
    """Compare two RVC models across all evaluation dimensions.

    Usage:
        comparator = ModelComparator()
        result = comparator.compare(model_a_data, model_b_data)
        comparator.save_report(result, "comparison.json")
    """

    # All metrics compared by this tool
    METRICS = ["score", "ecapa", "acoustic", "f0", "stability", "artifact"]

    def __init__(self, threshold: float = 1.0):
        """
        Args:
            threshold: Difference below which two models are considered "similar"
                       on a given metric. Default 1.0 works well for 0–100 scores.
        """
        self.threshold = threshold

    def load_result(self, path: str) -> Dict[str, Any]:
        """Load a scores.json file and return model entries.

        If the file contains multiple models (standard scores.json),
        returns the full dict for later model selection.
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_model(self, source, model_name: str) -> Dict[str, Any]:
        """Load a single model from a scores.json file or dict by name."""
        if isinstance(source, str):
            data = self.load_result(source)
        else:
            data = source

        models = data.get("models", [])
        for m in models:
            if m.get("model") == model_name or m.get("model_name") == model_name:
                return flatten_model_entry(m)

        # If data is already a flat entry
        if data.get("model") == model_name:
            return flatten_model_entry(data)

        raise ValueError(f"Model '{model_name}' not found in source")

    def compare(
        self, model_a: Dict[str, Any], model_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a full A/B comparison between two models.

        Args:
            model_a: Flat dict for model A (from _flatten_entry or similar).
            model_b: Flat dict for model B.

        Returns:
            Dict with difference, winner, and analysis sections.
        """
        result: Dict[str, Any] = {
            "model_a": model_a["model"],
            "model_b": model_b["model"],
            "difference": {},
            "winner": {},
            "analysis": [],
        }

        for metric in self.METRICS:
            a = model_a.get(metric, 0)
            b = model_b.get(metric, 0)
            diff = b - a

            result["difference"][metric] = round(diff, 3)

            if abs(diff) < self.threshold:
                result["winner"][metric] = "similar"
            elif diff > 0:
                result["winner"][metric] = str(model_b["model"])
            else:
                result["winner"][metric] = str(model_a["model"])

        result["analysis"] = self.generate_analysis(result["difference"])
        return result

    def generate_analysis(self, diff: Dict[str, float]) -> List[str]:
        """Produce natural-language analysis from per-metric differences."""
        text = []
        t = self.threshold

        if diff["ecapa"] > t:
            text.append("模型B音色相似度提升，说明说话人特征保持更好")
        elif diff["ecapa"] < -t:
            text.append("模型A音色相似度更高，模型B可能出现音色漂移")

        if diff["f0"] > t:
            text.append("模型B的音高跟随能力更好")
        elif diff["f0"] < -t:
            text.append("模型A的旋律/音高还原更稳定")

        if diff["artifact"] > t:
            text.append("模型B瑕疵更少")
        elif diff["artifact"] < -t:
            text.append("模型A音频质量更干净")

        if diff["stability"] > t:
            text.append("模型B训练稳定性更高")
        elif diff["stability"] < -t:
            text.append("模型A稳定性更好")

        if diff["acoustic"] > t:
            text.append("模型B声学距离更近（MCD/Mel/MFCC 表现更好）")
        elif diff["acoustic"] < -t:
            text.append("模型A声学特征更接近原始音频")

        return text

    def save_report(self, data: Dict[str, Any], path: str) -> None:
        """Save comparison report as JSON."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except OSError as e:
            raise RuntimeError(f"Failed to save comparison report to {path}: {e}") from e


# ---- CLI entry point ----
def main():
    parser = argparse.ArgumentParser(
        description="RVC Model Comparator — A/B comparison of two voice models"
    )
    parser.add_argument(
        "scores_file",
        help="Path to scores.json from an evaluate.py run",
    )
    parser.add_argument(
        "model_a",
        help="Name of Model A (e.g. '24')",
    )
    parser.add_argument(
        "model_b",
        help="Name of Model B (e.g. '25')",
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=1.0,
        help="Similarity threshold for metrics (default: 1.0)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save comparison report to JSON file",
    )
    args = parser.parse_args()

    scores_path = Path(args.scores_file)
    if not scores_path.exists():
        print(f"ERROR: File not found: {scores_path}")
        sys.exit(1)

    comparator = ModelComparator(threshold=args.threshold)

    try:
        model_a = comparator.load_model(str(scores_path), args.model_a)
        model_b = comparator.load_model(str(scores_path), args.model_b)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    result = comparator.compare(model_a, model_b)

    # Print summary
    print(f"\nA/B Comparison: {args.model_a}  vs  {args.model_b}")
    print("-" * 50)
    for metric in ModelComparator.METRICS:
        diff = result["difference"][metric]
        winner = result["winner"][metric]
        arrow = " →" if diff > 0 else ("← " if diff < 0 else " ≈")
        print(f"  {metric:<12} {arrow} {diff:+.3f}  ({winner})")

    print("\n--- Analysis ---")
    for line in result["analysis"]:
        print(f"  {line}")

    overall = result["difference"]["score"]
    if overall > 0:
        print(f"\n整体结论: 模型B ({result['model_b']}) 优于模型A ({result['model_a']})")
    elif overall < 0:
        print(f"\n整体结论: 模型A ({result['model_a']}) 优于模型B ({result['model_b']})")
    else:
        print("\n整体结论: 两个模型表现相当")

    if args.output:
        comparator.save_report(result, args.output)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
