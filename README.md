# RVC Voice Model Evaluator

> Objective evaluation and scoring tool for [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) voice conversion models — find the model that sounds closest to the original speaker.

<p align="center">
  <strong>ECAPA-TDNN Speaker Similarity</strong> &bull;
  <strong>MCD / Mel / MFCC Acoustic Distance</strong> &bull;
  <strong>F0 Prosody Analysis</strong> &bull;
  <strong>Artifact Detection</strong> &bull;
  <strong>Stability Scoring</strong>
</p>

---

## Overview

When you train multiple RVC models (or the same model at different epochs), how do you know which one actually sounds the most like the original speaker? Listening to each one is subjective and time-consuming. This tool provides a fully automated, multi-dimensional evaluation pipeline that compares RVC-generated audio against an original reference recording and produces a ranked score report.

**Key philosophy:** The goal is not to find the "cleanest" audio, but to find the model whose voice is **closest to the original speaker**. A model can sound great but have completely lost the speaker's identity — this tool catches that.

### Evaluation Pipeline

```
original.wav ──┐
               ├─► VAD ──► Windowing ──┬─► ECAPA Speaker Similarity (40%)
models/*.wav ──┘                       ├─► Acoustic Distance (30%)
                                       │    ├─ MCD (Mel Cepstral Distortion)
                                       │    ├─ Mel Spectral Distance
                                       │    └─ MFCC Distance
                                       ├─► F0 / Prosody (15%)
                                       │    ├─ F0 Correlation
                                       │    ├─ F0 DTW Distance
                                       │    └─ F0 RMSE (Normalized)
                                       ├─► Stability (10%)
                                       └─► Artifact Detection (5%)
                                            ├─ Clipping Ratio
                                            ├─ Spectral Flatness (Delta from Original)
                                            ├─ High-Frequency Ratio (Delta from Original)
                                            └─ Harmonic Anomaly
```

All acoustic comparisons use **DTW (Dynamic Time Warping)** alignment to handle minor timing differences between original and generated audio.

---

## Features

- **ECAPA-TDNN Speaker Verification** — Extracts speaker embeddings via SpeechBrain's ECAPA-TDNN model and computes cosine similarity per window. Score = 0.75 × mean + 0.25 × min to penalize inconsistent segments.
- **True MCD (Mel Cepstral Distortion)** — 24-dimensional MFCC (excluding c0 energy), DTW-aligned, with proper `10√2/ln10` scaling. Not a cheap Euclidean approximation.
- **Multi-Metric Acoustic Distance** — MCD + Log-Mel Spectral Distance + MFCC Distance, each DTW-aligned and weighted.
- **F0 Prosody Analysis** — Pearson correlation, DTW distance, and normalized RMSE (relative to mean F0) on voiced frames only, with octave-error correction.
- **Artifact Detection** — Clipping, spectral flatness, high-frequency anomaly, and harmonic ratio detection. Uses **delta-from-original** scoring so a model is judged relative to the reference, not just on absolute values.
- **Absolute Quality Scoring** — In addition to batch-relative ranking, each model receives an absolute 0-100 score with a quality label (`recommended` / `usable` / `not_recommended`), calibrated against self-baseline measurements.
- **Self-Baseline Test** — Splits the original audio in half and evaluates it against itself, establishing the "natural variation ceiling" that no model can realistically surpass.
- **A/B Comparison** — Side-by-side comparison of any two models with natural-language analysis.
- **Overfit Detection** — Track score trends across training checkpoints to identify when a model starts overfitting.
- **Rich Output** — CSV, Excel (4 sheets), JSON, HTML report with embedded charts, ranking bar chart, Top-5 radar chart, Top-N audio files for A/B listening, and a concatenated comparison audio file.

---

## Quick Start

### 1. Install Dependencies

```bash
# Recommended: create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

**Key Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.1.2 | Deep learning framework |
| torchaudio | 2.1.2 | Audio processing |
| speechbrain | 1.0.2 | ECAPA-TDNN speaker model |
| librosa | 0.10.2 | Audio feature extraction |
| scipy | 1.11.4 | DTW / statistics |
| numpy | 1.26.4 | Numerical computation |
| matplotlib | 3.8.2 | Chart generation |
| openpyxl | 3.1.2 | Excel output |
| soundfile | 0.12.1 | Audio I/O |
| pyyaml | 6.0.1 | Config parsing |

### 2. Prepare ECAPA-TDNN Model

Download the ECAPA-TDNN model from [SpeechBrain's HuggingFace](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) and place the files in the `ecapa_model/` directory:

```
ecapa_model/
├── hyperparams.yaml
├── label_encoder.txt
├── embed_model.ckpt
└── ...
```

The program uses a **fixed local model** — no runtime downloads.

### 3. Prepare Audio Files

```
project-root/
├── original.wav        # Original speaker reference (≥ 10s recommended)
└── models/
    ├── 1.wav           # RVC model output #1
    ├── 2.wav           # RVC model output #2
    ├── ...
    └── N.wav           # RVC model output #N
```

### 4. Run

```bash
# Default: uses config.yaml settings
python evaluate.py

# Specify paths
python evaluate.py --original ./original.wav --input ./models

# Use GPU
python evaluate.py --device cuda

# CPU mode
python evaluate.py --device cpu

# Self-baseline test
python evaluate.py --self-test

# A/B compare two models after evaluation
python evaluate.py --compare 1 2

# Overfit trend analysis
python evaluate.py --trend-dir ./results

# Windows quick launch
run_evaluate.bat
```

---

## Scoring System

### Weights

| Module | Weight | Description |
|--------|-------:|-------------|
| ECAPA Speaker Similarity | 40% | Speaker identity preservation |
| Acoustic / Timbre | 30% | MCD (45%) + Mel (35%) + MFCC (20%) |
| F0 / Prosody | 15% | Pitch contour matching |
| Stability | 10% | Cross-window consistency (std) |
| Artifact / Quality | 5% | Clipping, spectral anomalies, HF artifacts |
| **Total** | **100%** | |

### Batch Scoring vs. Absolute Scoring

The tool provides two independent scoring systems:

| | Batch Score (Final Score) | Absolute Score |
|---|---|---|
| **Method** | Min-Max normalization across current batch | Calibrated threshold tiers |
| **Range** | 0-100 (relative) | 0-100 (absolute) |
| **Comparable across runs?** | No — depends on batch | Yes — fixed thresholds |
| **Use case** | "Which model is best in this batch?" | "Is this model good enough to use?" |
| **Levels** | Technical Rank vs. Recommended Rank | `recommended` / `usable` / `not_recommended` |

### Technical Rank vs. Recommended Rank

- **Technical Rank** — Pure mathematical score ranking.
- **Recommended Rank** — Adjusted for quality warnings (severe clipping, high-frequency anomalies). A model with good scores but severe artifacts gets demoted.

### Warnings

| Warning | Trigger | Score Impact |
|---------|---------|-------------|
| `SEVERE_CLIPPING` | >5% frames clipped | -5 points |
| `WARNING_CLIPPING` | >1% frames clipped | None (informational) |
| `HIGH_FREQUENCY_ANOMALY` | >20% HF energy ratio | -10 points |
| `SEVERE_SPECTRAL_ANOMALY` | Spectral flatness < 0.02 | 0 (normal for RVC) |
| `LOW_SPEECH_RATIO` | <10% speech detected | -5 points |

> **Note:** `SEVERE_SPECTRAL_ANOMALY` is expected for RVC synthesis — RVC audio is naturally tonal. It carries 0 demerit points and is retained only as an informational marker.

---

## Configuration

All parameters are in [`config.yaml`](config.yaml). Key sections:

```yaml
# Audio
audio:
  sample_rate: 16000
  normalize_for_analysis: true   # Only affects analysis pipeline, NOT artifact detection

# Windowing
window:
  seconds: 5.0
  alignment_mode: "min_duration"

# ECAPA
ecapa:
  mean_weight: 0.75
  min_weight: 0.15               # Penalizes segments with low similarity

# Acoustic
acoustic:
  mcd_weight: 0.45
  mel_weight: 0.35
  mfcc_weight: 0.20
  use_dtw: true

# F0
f0:
  min_hz: 50
  max_hz: 600
  octave_correction: true

# Runtime
runtime:
  device: "auto"                 # "auto" | "cuda" | "cpu"
  batch_size: 16
```

---

## Output Files

```
results/
├── self_baseline/
│   └── self_baseline.json          # Self-test reference values
└── 20260808_183244/
    ├── ranking.csv                 # Summary ranking
    ├── detailed_scores.csv         # All raw metrics
    ├── ranking.xlsx                # Excel (4 sheets: Ranking / Detailed / Config / Human Eval)
    ├── scores.json                 # Structured JSON data
    ├── report.html                 # Self-contained HTML report with charts
    ├── ranking.png                 # Ranking bar chart
    ├── radar_top5.png              # Top-5 radar chart
    ├── comparison.wav              # AB comparison audio (original + top-N snippets)
    ├── evaluation.log              # Full log
    └── top5/                       # Top-N audio for quick listening
        ├── 01_model1.wav
        ├── 02_model2.wav
        └── ...
```

---

## Project Structure

```
RVC-Evaluator/
├── evaluate.py                 # Main entry point
├── config.yaml                 # All configurable parameters
├── requirements.txt            # Python dependencies
├── run_evaluate.bat            # Windows quick launcher
│
├── src/
│   ├── audio.py                # Audio loading, windowing, alignment
│   ├── vad.py                  # Voice Activity Detection (energy-based)
│   ├── ecapa.py                # ECAPA-TDNN speaker similarity (40%)
│   ├── acoustic.py             # MCD / Mel / MFCC + DTW alignment (30%)
│   ├── f0.py                   # F0 extraction, correlation, DTW, RMSE (15%)
│   ├── stability.py            # Cross-window consistency scoring (10%)
│   ├── artifacts.py            # Clipping, spectral flatness, HF anomaly (5%)
│   ├── absolute_score.py       # Batch-independent calibrated scoring
│   ├── scoring.py              # Normalization, weighting, ranking
│   ├── comparison.py           # A/B model comparison
│   ├── trend_analysis.py       # Overfit detection across checkpoints
│   ├── visualization.py        # Matplotlib charts
│   ├── report.py               # HTML report generation
│   └── utils.py                # Shared utilities
│
├── tests/
│   └── test_basic.py           # Unit tests (48 tests)
│
├── ecapa_model/                # ECAPA-TDNN model files (user-provided)
├── original.wav                # Reference audio (user-provided)
├── models/                     # RVC model outputs (user-provided)
└── results/                    # Evaluation outputs (auto-generated)
```

---

## MCD Implementation Details

This tool implements **true Mel Cepstral Distortion**, not a simplified approximation:

| Parameter | Value |
|-----------|-------|
| Coefficients | 24 MFCC (c1-c24) |
| c0 (energy) | Excluded |
| Formula | `MCD = (10√2 / ln10) × mean(√(Σ(cᵢ - c'ᵢ)²))` |
| Unit | dB (logarithmic scale, lower = more similar) |
| Alignment | DTW with Euclidean distance metric |
| Normalization | Distance divided by DTW path length |

**Typical MCD ranges (calibrated):**

| MCD | Quality |
|-----|---------|
| ≤ 180 | Excellent |
| 180-240 | Very Good |
| 240-300 | Good |
| 300-380 | Acceptable |
| > 450 | Poor |

---

## Self-Baseline Test

```bash
python evaluate.py --self-test
```

Splits `original.wav` into two halves and evaluates them against each other. This establishes baseline values for each metric — the "natural variation ceiling" that represents the speaker's own consistency. Use it to:

- Understand how close your models are to the theoretical best
- Calibrate absolute score thresholds for your specific speaker
- Verify that the evaluation pipeline is working correctly

---

## A/B Comparison

```bash
# After evaluation, compare two models
python evaluate.py --compare 1 2
```

Produces a side-by-side report showing per-metric differences, winners, and natural-language analysis:

```
==================================================
A/B Comparison: 1  vs  2
==================================================
  score         ← -3.250  (1)
  ecapa         ← -2.100  (1)
  acoustic       → +1.500  (2)
  ...
```

---

## Overfit Detection

```bash
python evaluate.py --trend-dir ./results
```

Analyzes score trends across multiple evaluation runs to detect when a model starts overfitting:

```
=== Score Trend ===
  epoch 10   65.32  |||||||||||||||
  epoch 20   72.18  |||||||||||||||||
  epoch 30   75.44  ||||||||||||||||||
  epoch 40   73.21  |||||||||||||||||     ← declining
  epoch 50   70.89  ||||||||||||||||      ← declining
```

---

## FAQ

**Q: The result shows `SEVERE_SPECTRAL_ANOMALY` — is something wrong?**

No. RVC-synthesized audio is naturally tonal (low spectral flatness). This warning is expected and carries **0 score penalty**. It's kept only as an informational marker.

**Q: ECAPA model not found?**

Place SpeechBrain ECAPA-TDNN model files in the `ecapa_model/` directory. See [Step 2](#2-prepare-ecapa-tdnn-model).

**Q: CUDA out of memory?**

Reduce `batch_size` in `config.yaml` (try 8 or 4), or use `--device cpu`.

**Q: Final score is 90 but the model doesn't sound great?**

Final score is **relative to the batch**. A 90 just means it's the best among the models you provided. Use the **absolute score** and **self-baseline** to judge whether it's actually good enough in absolute terms.

**Q: One metric failed (F0 FAILED / MCD FAILED)?**

The program won't crash. The failed metric is marked as NaN and its weight is redistributed to the remaining metrics. Check the log for the specific error.

**Q: How many models can I evaluate at once?**

No hard limit. The tool auto-scans `models/*.wav`. More models = better batch normalization, but longer runtime.

---

## Testing

```bash
python -m pytest tests/ -v
```

48 unit tests covering scoring, normalization, artifact detection, MCD computation, F0 analysis, and absolute score calibration.

---

## Tech Stack

- **Python 3.10+**
- **PyTorch 2.1** — ECAPA-TDNN inference
- **SpeechBrain 1.0** — Pretrained speaker verification model
- **librosa** — Audio feature extraction (MFCC, Mel, STFT, HPSS)
- **SciPy** — DTW dynamic programming
- **Matplotlib** — Chart generation
- **openpyxl** — Excel report generation

---

## License

MIT License — feel free to use, modify, and distribute.

---

## Acknowledgments

- [SpeechBrain](https://github.com/speechbrain/speechbrain) — ECAPA-TDNN speaker verification
- [RVC Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — Voice conversion framework
- [librosa](https://librosa.org/) — Audio analysis library
