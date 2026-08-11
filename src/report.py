"""
HTML Report Generation Module.

Generates a self-contained HTML report with:
  - Ranking table
  - Detailed scores table
  - Embedded charts (ranking.png, radar_top5.png)
  - Configuration summary
  - Warnings and status
"""

import base64
import logging
from html import escape as _html_escape
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


def generate_html_report(
    results: List[Dict[str, Any]],
    output_path: str,
    config: Dict[str, Any],
    runtime_info: Dict[str, Any],
    ranking_png: str = "",
    radar_png: str = "",
) -> None:
    """Generate a self-contained HTML report.

    Args:
        results: List of per-model result dicts.
        output_path: Path to save report.html.
        config: Configuration dict.
        runtime_info: Runtime environment info.
        ranking_png: Path to ranking chart (will be embedded).
        radar_png: Path to radar chart (will be embedded).
    """
    if not results:
        logger.warning("No results to report")
        return

    html = _build_html(results, config, runtime_info, ranking_png, radar_png)
    Path(output_path).write_text(html, encoding="utf-8")
    logger.info("HTML report saved to %s", output_path)


def _build_html(
    results: List[Dict[str, Any]],
    config: Dict[str, Any],
    runtime_info: Dict[str, Any],
    ranking_png: str,
    radar_png: str,
) -> str:
    """Build the HTML document string."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Sort by technical rank
    sorted_results = sorted(
        results, key=lambda r: r["scores"].get("technical_rank", 999)
    )

    # Build ranking table
    ranking_rows = ""
    for r in sorted_results:
        s = r["scores"]
        tech_rank = s.get("technical_rank", "-")
        rec_rank = s.get("recommended_rank", "-")
        fs = s.get("final_score", 0)
        fs_str = f"{fs:.2f}" if not np.isnan(fs) else "N/A"

        warnings = s.get("warnings", [])
        warning_str = ", ".join(warnings) if warnings else "—"
        status = s.get("status", "OK")

        status_class = "ok"
        if "SEVERE" in warning_str:
            status_class = "severe"
        elif "WARNING" in warning_str:
            status_class = "warning"

        ranking_rows += f"""
        <tr class="{status_class}">
            <td>{tech_rank}</td>
            <td>{rec_rank}</td>
            <td><strong>{_html_escape(r["model_name"])}</strong></td>
            <td class="score">{fs_str}</td>
            <td>{_val_or_na(s.get("ecapa_score"))}</td>
            <td>{_val_or_na(s.get("acoustic_score"))}</td>
            <td>{_val_or_na(s.get("f0_score"))}</td>
            <td>{_val_or_na(s.get("stability_score"))}</td>
            <td>{_val_or_na(s.get("artifact_score"))}</td>
            <td>{_val_or_na((s.get("absolute_score") or {}).get("score"))}</td>
            <td>{(s.get("absolute_score") or {}).get("overall_level", "N/A")}</td>
            <td class="warning-cell">{warning_str}</td>
            <td>{status}</td>
        </tr>"""

    # Build detail table
    detail_rows = ""
    for r in sorted_results:
        s = r["scores"]

        detail_rows += f"""
        <tr>
            <td>{_html_escape(r["model_name"])}</td>
            <td>{_val_or_na(s.get("ecapa_mean"))}</td>
            <td>{_val_or_na(s.get("ecapa_min"))}</td>
            <td>{_val_or_na(s.get("mcd_distance"))}</td>
            <td>{_val_or_na(s.get("mel_distance"))}</td>
            <td>{_val_or_na(s.get("mfcc_distance"))}</td>
            <td>{_val_or_na(s.get("f0_correlation"))}</td>
            <td>{_val_or_na(s.get("f0_rmse"))}</td>
            <td>{_val_or_na(s.get("clipping_ratio"))}</td>
            <td>{_val_or_na(s.get("high_frequency_ratio"))}</td>
            <td>{_val_or_na(s.get("speech_ratio"))}</td>
        </tr>"""

    # Config table
    weights = config.get("final_score", {})
    config_rows = f"""
        <tr><td>ECAPA Weight</td><td>{weights.get("ecapa_weight", 0.40)}</td></tr>
        <tr><td>Acoustic Weight</td><td>{weights.get("acoustic_weight", 0.30)}</td></tr>
        <tr><td>F0 Weight</td><td>{weights.get("f0_weight", 0.15)}</td></tr>
        <tr><td>Stability Weight</td><td>{weights.get("stability_weight", 0.10)}</td></tr>
        <tr><td>Artifact Weight</td><td>{weights.get("artifact_weight", 0.05)}</td></tr>
        <tr><td>Sample Rate</td><td>{config.get("audio", {}).get("sample_rate", 16000)} Hz</td></tr>
        <tr><td>Window Size</td><td>{config.get("window", {}).get("seconds", 3.0)}s</td></tr>
        <tr><td>Device</td><td>{runtime_info.get("device", "N/A")}</td></tr>
        <tr><td>Python</td><td>{runtime_info.get("python_version", "N/A")}</td></tr>
        <tr><td>PyTorch</td><td>{runtime_info.get("pytorch_version", "N/A")}</td></tr>
        <tr><td>CUDA</td><td>{runtime_info.get("cuda_version", "N/A")}</td></tr>
    """

    # Embed images
    ranking_img_tag = ""
    radar_img_tag = ""
    if ranking_png and Path(ranking_png).exists():
        with open(ranking_png, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ranking_img_tag = f'<img src="data:image/png;base64,{b64}" alt="Ranking Chart" style="max-width:100%;">'
    if radar_png and Path(radar_png).exists():
        with open(radar_png, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        radar_img_tag = f'<img src="data:image/png;base64,{b64}" alt="Radar Chart" style="max-width:100%;">'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RVC Model Evaluation Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f5f7fa; color: #333; }}
.container {{ max-width: 1400px; margin: 0 auto; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #2c3e50; margin-top: 30px; }}
.timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 4px; overflow: hidden; }}
th {{ background: #3498db; color: white; padding: 10px 8px; text-align: left; font-size: 0.85em; }}
td {{ padding: 8px; border-bottom: 1px solid #ecf0f1; font-size: 0.85em; }}
tr:hover {{ background: #f8f9fa; }}
.score {{ font-weight: bold; font-size: 1.1em; }}
tr.ok {{ }}
tr.warning {{ background: #fff9e6; }}
tr.severe {{ background: #ffe6e6; }}
.warning-cell {{ color: #e67e22; font-size: 0.8em; max-width: 200px; }}
.charts {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }}
.chart-box {{ flex: 1; min-width: 400px; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.chart-box h3 {{ margin-top: 0; }}
.tabs {{ overflow-x: auto; }}
.note {{ background: #eaf2f8; padding: 12px; border-radius: 4px; margin: 15px 0; font-size: 0.9em; }}
</style>
</head>
<body>
<div class="container">
<h1>RVC Voice Model Evaluation Report</h1>
<p class="timestamp">Generated: {now}</p>

<div class="note">
<strong>Note:</strong> Scores are relative rankings within the current batch (0-100). Higher scores = closer to Original. Technical Rank is purely mathematical; Recommended Rank considers artifact warnings. Absolute Score is an independent quality rating (0-100) not affected by batch size.
</div>

<h2>1. Ranking</h2>
<div class="tabs">
<table>
<thead>
<tr>
    <th>Tech Rank</th>
    <th>Rec Rank</th>
    <th>Model</th>
    <th>Final Score</th>
    <th>ECAPA</th>
    <th>Acoustic</th>
    <th>F0</th>
    <th>Stability</th>
    <th>Artifact</th>
    <th>Abs Score</th>
    <th>Abs Level</th>
    <th>Warnings</th>
    <th>Status</th>
</tr>
</thead>
<tbody>
{ranking_rows}
</tbody>
</table>
</div>

<h2>2. Charts</h2>
<div class="charts">
<div class="chart-box">
<h3>Ranking</h3>
{ranking_img_tag}
</div>
<div class="chart-box">
<h3>Top Models Radar</h3>
{radar_img_tag}
</div>
</div>

<h2>3. Detailed Metrics</h2>
<div class="tabs">
<table>
<thead>
<tr>
    <th>Model</th>
    <th>ECAPA Mean</th>
    <th>ECAPA Min</th>
    <th>MCD</th>
    <th>Mel Dist</th>
    <th>MFCC Dist</th>
    <th>F0 Corr</th>
    <th>F0 RMSE</th>
    <th>Clipping</th>
    <th>HF Ratio</th>
    <th>Speech Ratio</th>
</tr>
</thead>
<tbody>
{detail_rows}
</tbody>
</table>
</div>

<h2>4. Configuration</h2>
<table>
<thead><tr><th>Parameter</th><th>Value</th></tr></thead>
<tbody>{config_rows}</tbody>
</table>

<p style="text-align:center; color:#95a5a6; margin-top:40px; font-size:0.85em;">
RVC Voice Model Evaluator — Generated {now}
</p>
</div>
</body>
</html>"""


def _val_or_na(val, precision=3) -> str:
    """Format a value or return 'N/A'."""
    if val is None:
        return "N/A"
    if isinstance(val, float) and np.isnan(val):
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    return str(val)
