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
