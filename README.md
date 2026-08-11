# RVC 语音模型评估器

> 针对 [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 语音转换模型的客观评估与打分工具 —— 找出声音最接近原始说话人的模型。

<p align="center">
  <strong>ECAPA-TDNN 说话人相似度</strong> &bull;
  <strong>MCD / Mel / MFCC 声学距离</strong> &bull;
  <strong>F0 韵律分析</strong> &bull;
  <strong>伪影检测</strong> &bull;
  <strong>稳定性评分</strong>
</p>

---

## 概述

当你训练了多个 RVC 模型（或同一模型的不同 epoch），如何知道哪个最像原始说话人？逐个试听既主观又耗时。本工具提供一套全自动、多维度的评估流程，将 RVC 生成的音频与原始参考录音进行对比，生成排名评分报告。

**核心理念：** 目标不是找"最干净"的音频，而是找声音**最接近原始说话人**的模型。一个模型可能听起来很好，但说话人特征已经完全丢失 —— 本工具就是用来发现这个问题的。

### 评估流程

```
original.wav ──┐
               ├─► VAD ──► 分窗 ──┬─► ECAPA 说话人相似度 (40%)
models/*.wav ──┘                   ├─► 声学距离 (30%)
                                   │    ├─ MCD（Mel 倒谱失真）
                                   │    ├─ Mel 频谱距离
                                   │    └─ MFCC 距离
                                   ├─► F0 / 韵律 (15%)
                                   │    ├─ F0 相关性
                                   │    ├─ F0 DTW 距离
                                   │    └─ F0 RMSE（归一化）
                                   ├─► 稳定性 (10%)
                                   └─► 伪影检测 (5%)
                                        ├─ 削波率
                                        ├─ 频谱平坦度（与原始音频的差值）
                                        ├─ 高频能量比（与原始音频的差值）
                                        └─ 谐波异常
```

所有声学对比均使用 **DTW（动态时间规整）** 对齐，以处理原始音频与生成音频之间的微小时间差异。

---

## 功能特性

- **ECAPA-TDNN 说话人验证** — 通过 SpeechBrain 的 ECAPA-TDNN 模型提取说话人嵌入，逐窗计算余弦相似度。评分 = 0.75 × 均值 + 0.25 × 最小值，以惩罚不一致的片段。
- **真正的 MCD（Mel 倒谱失真）** — 24 维 MFCC（排除 c0 能量项），DTW 对齐，使用标准 `10√2/ln10` 系数。不是简单的欧氏距离近似。
- **多维度声学距离** — MCD + Log-Mel 频谱距离 + MFCC 距离，各自 DTW 对齐后加权融合。
- **F0 韵律分析** — 在浊音帧上计算 Pearson 相关系数、DTW 距离和归一化 RMSE（相对于平均 F0），支持八度误差校正。
- **伪影检测** — 削波、频谱平坦度、高频异常、谐波比检测。使用**与原始音频的差值**评分，确保模型是相对于参考音频被评判，而非仅看绝对值。
- **绝对质量评分** — 除批次相对排名外，每个模型还获得 0-100 的绝对评分和质量标签（`recommended` / `usable` / `not_recommended`），基于自基线测量校准。
- **自基线测试** — 将原始音频分为两半互相评估，建立"自然变化上限"——任何模型都不可能超过这个上限。
- **A/B 对比** — 任意两个模型的并排比较，附带自然语言分析。
- **过拟合检测** — 跨训练 checkpoint 追踪评分趋势，识别模型何时开始过拟合。
- **丰富的输出** — CSV、Excel（4 个工作表）、JSON、内嵌图表的 HTML 报告、排名柱状图、Top-5 雷达图、Top-N 音频文件（方便 A/B 试听）、拼接对比音频。

---

## 快速开始

### 0. 一键安装运行（Windows，推荐）

双击 `setup_and_run.bat`，自动完成：
- 检测 Python 并创建 `.venv` 虚拟环境
- 从 `requirements.txt` 安装所有依赖
- 下载 ECAPA-TDNN 说话人模型（约 55MB）
- 执行评估

适用于任何已安装 **Python 3.9+** 的 Windows 机器，无需手动操作。

### 1. 手动安装依赖（备选方案）

```bash
# 推荐先创建虚拟环境
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

**核心依赖：**

| 包 | 版本 | 用途 |
|----|------|------|
| torch | 2.1.2 | 深度学习框架 |
| torchaudio | 2.1.2 | 音频处理 |
| speechbrain | 1.0.2 | ECAPA-TDNN 说话人模型 |
| librosa | 0.10.2 | 音频特征提取 |
| scipy | 1.11.4 | DTW / 统计计算 |
| numpy | 1.26.4 | 数值计算 |
| matplotlib | 3.8.2 | 图表生成 |
| openpyxl | 3.1.2 | Excel 输出 |
| soundfile | 0.12.1 | 音频读写 |
| pyyaml | 6.0.1 | 配置解析 |

### 2. 准备 ECAPA-TDNN 模型

从 [SpeechBrain HuggingFace](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) 下载 ECAPA-TDNN 模型，放入 `ecapa_model/` 目录：

```
ecapa_model/
├── hyperparams.yaml
├── label_encoder.txt
├── embed_model.ckpt
└── ...
```

程序使用**固定本地模型**，运行时不联网下载。

> 如果使用一键安装脚本（步骤 0），此步骤会自动完成。

### 3. 准备音频文件

```
项目根目录/
├── original.wav        # 原始说话人参考音频（建议 ≥ 10 秒）
└── models/
    ├── 1.wav           # RVC 模型输出 #1
    ├── 2.wav           # RVC 模型输出 #2
    ├── ...
    └── N.wav           # RVC 模型输出 #N
```

### 4. 运行评估

```bash
# 默认：使用 config.yaml 设置
python evaluate.py

# 指定路径
python evaluate.py --original ./original.wav --input ./models

# 使用 GPU
python evaluate.py --device cuda

# CPU 模式
python evaluate.py --device cpu

# 自基线测试
python evaluate.py --self-test

# A/B 对比两个模型（评估完成后）
python evaluate.py --compare 1 2

# 过拟合趋势分析
python evaluate.py --trend-dir ./results

# Windows 快速启动
run_evaluate.bat
```

---

## 评分体系

### 权重分配

| 模块 | 权重 | 说明 |
|------|-----:|------|
| ECAPA 说话人相似度 | 40% | 说话人特征保留程度 |
| 声学/音色 | 30% | MCD (45%) + Mel (35%) + MFCC (20%) |
| F0/韵律 | 15% | 音高轨迹匹配度 |
| 稳定性 | 10% | 跨窗口一致性（标准差） |
| 伪影/质量 | 5% | 削波、频谱异常、高频伪影 |
| **总计** | **100%** | |

### 批次评分 vs. 绝对评分

本工具提供两套独立的评分系统：

| | 批次评分（Final Score） | 绝对评分（Absolute Score） |
|---|---|---|
| **方法** | 当前批次内 Min-Max 归一化 | 校准阈值分级 |
| **范围** | 0-100（相对） | 0-100（绝对） |
| **可跨批次比较？** | 否 — 取决于批次 | 是 — 固定阈值 |
| **用途** | "这批模型里哪个最好？" | "这个模型够用吗？" |
| **级别** | 技术排名 vs. 推荐排名 | `recommended` / `usable` / `not_recommended` |

### 技术排名 vs. 推荐排名

- **技术排名** — 纯数学评分排名。
- **推荐排名** — 根据质量警告（严重削波、高频异常等）调整。评分好但有严重伪影的模型会被降级。

### 警告系统

| 警告 | 触发条件 | 扣分 |
|------|---------|------|
| `SEVERE_CLIPPING` | >5% 帧削波 | -5 分 |
| `WARNING_CLIPPING` | >1% 帧削波 | 无（仅提示） |
| `HIGH_FREQUENCY_ANOMALY` | >20% 高频能量比 | -10 分 |
| `SEVERE_SPECTRAL_ANOMALY` | 频谱平坦度 < 0.02 | 0 分（RVC 正常现象） |
| `LOW_SPEECH_RATIO` | <10% 语音检测率 | -5 分 |

> **说明：** `SEVERE_SPECTRAL_ANOMALY` 对 RVC 合成语音是正常现象 —— RVC 音频天然偏纯音化。此警告扣分为 0，仅作为信息标记保留。

---

## 配置说明

所有参数在 [`config.yaml`](config.yaml) 中配置，主要部分：

```yaml
# 音频
audio:
  sample_rate: 16000
  normalize_for_analysis: true   # 仅影响分析流程，不影响伪影检测

# 分窗
window:
  seconds: 5.0
  alignment_mode: "min_duration"

# ECAPA
ecapa:
  mean_weight: 0.75
  min_weight: 0.15               # 惩罚低相似度片段

# 声学
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

# 运行时
runtime:
  device: "auto"                 # "auto" | "cuda" | "cpu"
  batch_size: 16
```

---

## 输出文件

```
results/
├── self_baseline/
│   └── self_baseline.json          # 自基线参考值
└── 20260808_183244/
    ├── ranking.csv                 # 排名摘要
    ├── detailed_scores.csv         # 所有原始指标
    ├── ranking.xlsx                # Excel（4 个工作表：排名/详情/配置/人工评价）
    ├── scores.json                 # 结构化 JSON 数据
    ├── report.html                 # 自包含 HTML 报告（含图表）
    ├── ranking.png                 # 排名柱状图
    ├── radar_top5.png              # Top-5 雷达图
    ├── comparison.wav              # A/B 对比音频（原始 + Top-N 片段）
    ├── evaluation.log              # 完整日志
    └── top5/                       # Top-N 音频（快速试听）
        ├── 01_model1.wav
        ├── 02_model2.wav
        └── ...
```

---

## 项目结构

```
RVC-Evaluator/
├── evaluate.py                 # 主入口
├── config.yaml                 # 所有可配置参数
├── requirements.txt            # Python 依赖
├── run_evaluate.bat            # Windows 快速启动
├── setup_and_run.bat           # Windows 一键安装+运行
│
├── src/
│   ├── audio.py                # 音频加载、分窗、对齐
│   ├── vad.py                  # 语音活动检测（基于能量）
│   ├── ecapa.py                # ECAPA-TDNN 说话人相似度 (40%)
│   ├── acoustic.py             # MCD / Mel / MFCC + DTW 对齐 (30%)
│   ├── f0.py                   # F0 提取、相关性、DTW、RMSE (15%)
│   ├── stability.py            # 跨窗口一致性评分 (10%)
│   ├── artifacts.py            # 削波、频谱平坦度、高频异常 (5%)
│   ├── absolute_score.py       # 独立于批次的校准评分
│   ├── scoring.py              # 归一化、加权、排名
│   ├── comparison.py           # A/B 模型对比
│   ├── trend_analysis.py       # 跨 checkpoint 过拟合检测
│   ├── visualization.py        # Matplotlib 图表
│   ├── report.py               # HTML 报告生成
│   └── utils.py                # 共享工具函数
│
├── tests/
│   └── test_basic.py           # 单元测试（48 项）
│
├── ecapa_model/                # ECAPA-TDNN 模型文件（用户提供）
├── original.wav                # 参考音频（用户提供）
├── models/                     # RVC 模型输出（用户提供）
└── results/                    # 评估输出（自动生成）
```

---

## MCD 实现细节

本工具实现的是**真正的 Mel 倒谱失真**，不是简化近似：

| 参数 | 值 |
|------|-----|
| 系数 | 24 维 MFCC（c1-c24） |
| c0（能量） | 排除 |
| 公式 | `MCD = (10√2 / ln10) × mean(√(Σ(cᵢ - c'ᵢ)²))` |
| 单位 | dB（对数刻度，越低越相似） |
| 对齐 | DTW，欧氏距离度量 |
| 归一化 | 距离除以 DTW 路径长度 |

**典型 MCD 范围（已校准）：**

| MCD | 质量 |
|-----|------|
| ≤ 180 | 优秀 |
| 180-240 | 很好 |
| 240-300 | 良好 |
| 300-380 | 可接受 |
| > 450 | 差 |

---

## 自基线测试

```bash
python evaluate.py --self-test
```

将 `original.wav` 分成两半互相评估，建立各项指标的基线值 —— 即说话人自身一致性的"自然变化上限"。可用于：

- 了解你的模型距离理论最优有多远
- 为特定说话人校准绝对评分阈值
- 验证评估流程是否正常工作

---

## A/B 对比

```bash
# 评估完成后，对比两个模型
python evaluate.py --compare 1 2
```

生成并排报告，展示各指标差异、胜者和自然语言分析：

```
==================================================
A/B 对比: 1  vs  2
==================================================
  总分           ← -3.250  (1 胜)
  ECAPA          ← -2.100  (1 胜)
  声学            → +1.500  (2 胜)
  ...
```

---

## 过拟合检测

```bash
python evaluate.py --trend-dir ./results
```

分析多次评估运行的评分趋势，检测模型何时开始过拟合：

```
=== 评分趋势 ===
  epoch 10   65.32  |||||||||||||||
  epoch 20   72.18  |||||||||||||||||
  epoch 30   75.44  ||||||||||||||||||
  epoch 40   73.21  |||||||||||||||||     ← 下降中
  epoch 50   70.89  ||||||||||||||||      ← 下降中
```

---

## 常见问题

**Q: 结果显示 `SEVERE_SPECTRAL_ANOMALY` — 是有问题吗？**

没有问题。RVC 合成音频天然偏纯音化（频谱平坦度低），此警告是正常现象，**扣分为 0**，仅作为信息标记保留。

**Q: ECAPA 模型找不到？**

将 SpeechBrain ECAPA-TDNN 模型文件放入 `ecapa_model/` 目录，详见[步骤 2](#2-准备-ecapa-tdnn-模型)。使用一键安装脚本可自动下载。

**Q: CUDA 显存不足？**

在 `config.yaml` 中减小 `batch_size`（试试 8 或 4），或使用 `--device cpu`。

**Q: 最终评分 90 分但模型听起来不好？**

最终评分是**相对于批次**的。90 分只说明它是你提供的模型里最好的。请结合**绝对评分**和**自基线**判断是否真正够用。

**Q: 某个指标失败（F0 FAILED / MCD FAILED）？**

程序不会崩溃。失败的指标标记为 NaN，其权重会重新分配给其余指标。查看日志了解具体错误。

**Q: 一次可以评估多少个模型？**

没有硬性上限。工具会自动扫描 `models/*.wav`。模型越多，批次归一化越准确，但运行时间越长。

---

## 测试

```bash
python -m pytest tests/ -v
```

包含 48 项单元测试，覆盖评分、归一化、伪影检测、MCD 计算、F0 分析和绝对评分校准。

---

## 技术栈

- **Python 3.10+**
- **PyTorch 2.1** — ECAPA-TDNN 推理
- **SpeechBrain 1.0** — 预训练说话人验证模型
- **librosa** — 音频特征提取（MFCC、Mel、STFT、HPSS）
- **SciPy** — DTW 动态规划
- **Matplotlib** — 图表生成
- **openpyxl** — Excel 报告生成

---

## 许可证

MIT License — 自由使用、修改和分发。

---

## 致谢

- [SpeechBrain](https://github.com/speechbrain/speechbrain) — ECAPA-TDNN 说话人验证
- [RVC Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — 语音转换框架
- [librosa](https://librosa.org/) — 音频分析库

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
