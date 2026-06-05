# ChartArena <!-- omit in toc -->

**A Comprehensive Bilingual Benchmark for General Chart Parsing across Families, Scenarios, and Formats**

<p align="center">
  <a href="README_zh.md">中文版</a> •
  <a href="https://arxiv.org/abs/2606.01348">Paper</a> •
  <a href="https://github.com/pspdada/ChartArena">GitHub Repo</a> •
  <a href="https://huggingface.co/datasets/psp-dada/ChartArena">HuggingFace Dataset</a> •
  <a href="#">ModelScope Dataset</a>
</p>

## News <!-- omit in toc -->

- [2026.06.01] 📖 Code and data are released!

## Overview <!-- omit in toc -->

**ChartArena** is a comprehensive bilingual benchmark for evaluating the **chart parsing** capabilities of vision-language models, spanning the full difficulty spectrum of charts encountered in practice. It covers **eight chart families**: both numeric charts (bar, line, pie, radar, box plot, combination) and diagrammatic structures (flowchart, mind map), each presented across **three visual scenarios** (digital renderings, printed photos, and hand-drawn photos) and **two languages** (Chinese and English).

To enable fair comparison across models that produce mutually incompatible output formats, ChartArena adopts a **format-agnostic evaluation protocol**: heterogeneous predictions are normalized into two canonical semantic spaces: a triple view for numeric charts and a directed graph view for diagrammatic charts, and scored with structure-aware metrics.

<table align="center">
    <p align="center">
      <img src="/docs/figures/ChartArena_overview.jpg" width="80%" />
    </p>
</table>

## Contents <!-- omit in toc -->

- [Benchmark Statistics](#benchmark-statistics)
- [Leaderboard](#leaderboard)
- [Task Definitions](#task-definitions)
- [Getting Started](#getting-started)
- [Citation](#citation)
- [License](#license)

## Benchmark Statistics

| Item             | Details                                                               |
| ---------------- | --------------------------------------------------------------------- |
| Chart Families   | 8 (bar, line, pie, radar, box plot, combination, flowchart, mind map) |
| Chart Categories | Numeric charts, mind maps, flowcharts                                 |
| Visual Scenarios | 3 (digital rendering, printed photo, hand-drawn photo)                |
| Languages        | Bilingual (Chinese and English)                                       |

## Leaderboard

We evaluate 26 models across three categories: general-purpose MLLMs, document parsing MLLMs, and expert chart understanding models. Results are reported as **mAP$_{high}$** per chart family, with separate **EN** (English) and **ZH** (Chinese) scores each averaged over three visual scenarios. Within each category, **bold** marks the best result per column.

<details>
<summary>Full leaderboard (click to expand)</summary>

### General-Purpose MLLMs <!-- omit in toc -->

| Model                       |  Date   | Bar (EN) | Bar (ZH) | Line (EN) | Line (ZH) | Pie (EN) | Pie (ZH) | Radar (EN) | Radar (ZH) | Box (EN) | Box (ZH) | Combo (EN) | Combo (ZH) | Flow (EN) | Flow (ZH) | Mind (EN) | Mind (ZH) | Avg (EN) | Avg (ZH) |
| :-------------------------- | :-----: | :------: | :------: | :-------: | :-------: | :------: | :------: | :--------: | :--------: | :------: | :------: | :--------: | :--------: | :-------: | :-------: | :-------: | :-------: | :------: | :------: |
| GPT-4o                      | 2024.05 |   21.6   |   36.3   |   27.5    |   52.9    |   76.7   |   74.2   |    9.7     |    24.9    |   19.1   |   9.6    |    9.9     |    40.7    |   49.8    |   27.1    |   64.0    |   24.8    |   34.8   |   36.3   |
| GPT-5                       | 2025.08 |   35.1   |   52.3   |   48.1    |   65.1    |   81.1   |   78.9   |  **32.0**  |    41.5    |   19.8   |   12.8   |    14.2    |    46.5    |   58.1    |   35.3    |   76.6    |   33.5    |   45.6   |   45.8   |
| Qwen2.5-VL-7B-Instruct      | 2025.02 |   15.2   |   36.9   |   17.9    |   39.9    |   63.4   |   73.1   |    8.3     |    19.1    |   0.9    |   2.8    |    6.0     |    40.6    |   29.7    |   23.2    |   45.4    |   29.9    |   23.3   |   33.2   |
| Qwen2.5-VL-72B-Instruct     | 2025.02 |   27.1   |   53.3   |   38.2    |   66.7    |   73.5   |   77.0   |    10.9    |    38.5    |   15.0   |   15.3   |    14.3    |    50.5    |   50.1    |   43.6    |   63.8    |   55.0    |   36.6   |   50.0   |
| InternVL3.5-8B              | 2025.08 |   20.9   |   49.4   |   34.1    |   49.9    |   63.9   |   72.6   |    12.6    |    35.7    |   4.3    |   10.7   |    7.6     |    41.2    |   31.5    |   24.3    |   47.0    |   32.2    |   27.7   |   39.5   |
| InternVL3.5-241B-A28B       | 2025.08 |   27.5   |   57.2   |   41.3    |   55.7    |   77.7   |   83.3   |    15.2    |    41.4    |   18.7   |   21.6   |    17.7    |    47.8    |   43.8    |   36.6    |   62.6    |   45.5    |   38.0   |   48.6   |
| Qwen3VL-8B-Instruct         | 2025.10 |   33.9   |   63.4   |   43.1    |   67.9    |   78.6   |   88.3   |    16.8    |    52.1    |   35.7   |   30.4   |    14.2    |    51.9    |   50.0    |   41.5    |   75.2    |   62.6    |   43.4   |   57.3   |
| Qwen3VL-235B-A22B-Ins.      | 2025.10 |   44.5   |   71.9   |   57.1    |   77.1    |   85.8   |   87.9   |    24.6    |    52.4    | **54.8** |   55.1   |    29.1    |    60.8    |   57.9    |   49.8    |   79.4    |   73.7    |   54.2   |   66.1   |
| Qwen3.5-35B-A3B             | 2026.02 |   48.0   |   68.1   |   60.4    |   77.6    |   89.7   |   88.7   |    25.2    |    57.9    |   50.1   |   50.6   |    35.2    |    62.1    |   62.5    |   56.5    |   77.1    |   75.6    |   56.0   |   67.1   |
| GLM-4.5V                    | 2025.07 |   33.5   |   61.4   |   51.7    |   70.5    |   81.2   |   83.1   |    19.7    |    43.1    |   32.4   |   37.4   |    21.2    |    52.5    |   44.7    |   39.6    |   66.2    |   43.7    |   43.8   |   53.9   |
| Seed-1.8 (non-thinking)     | 2025.12 |   29.1   |   59.7   |   46.0    |   72.5    |   84.7   |   88.0   |    22.0    |    45.9    |   16.1   |   17.5   |    15.0    |    59.7    |   47.8    |   50.3    |   76.5    |   69.1    |   42.2   |   57.8   |
| Seed-2.0 Pro (non-thinking) | 2026.02 |   40.3   |   73.3   |   56.5    |   80.7    |   91.5   |   90.5   |    21.3    |    54.7    |   44.5   | **55.2** |    32.4    |    62.2    |   62.6    |   61.3    |   83.1    | **85.8**  |   54.0   |   70.5   |
| Kimi K2.5 (non-thinking)    | 2026.02 |   45.2   |   70.3   |   60.9    |   79.8    |   87.2   |   86.7   |    30.2    |    59.7    |   40.6   |   47.6   |    33.6    |    63.6    |   59.9    |   57.9    |   80.8    |   79.4    |   54.8   |   68.1   |
| MiMo-V2-Omni                | 2026.03 |   31.1   |   56.9   |   41.5    |   66.4    |   87.0   |   85.8   |    19.7    |    46.1    |   19.1   |   30.3   |    19.4    |    54.7    |   57.1    |   51.0    |   76.6    |   64.6    |   43.9   |   57.0   |
| Gemini 2.5 Pro              | 2025.03 |   46.0   |   76.5   |   56.5    |   77.6    |   88.6   |   87.3   |    17.5    |    53.0    |   10.2   |   22.1   |    28.7    |    57.6    |   62.1    |   57.8    |   71.7    |   67.1    |   47.7   |   62.4   |
| Gemini 3.1 Pro              | 2026.02 | **57.9** | **78.7** | **67.0**  | **85.3**  | **92.5** | **95.1** |    31.8    |  **62.7**  |   32.5   |   45.2   |  **39.7**  |  **70.3**  | **65.6**  | **63.1**  | **86.8**  |   85.2    | **59.2** | **73.2** |

### Document Parsing MLLMs <!-- omit in toc -->

| Model             |  Date   | Bar (EN) | Bar (ZH) | Line (EN) | Line (ZH) | Pie (EN) | Pie (ZH) | Radar (EN) | Radar (ZH) | Box (EN) | Box (ZH) | Combo (EN) | Combo (ZH) | Flow (EN) | Flow (ZH) | Mind (EN) | Mind (ZH) | Avg (EN) | Avg (ZH) |
| :---------------- | :-----: | :------: | :------: | :-------: | :-------: | :------: | :------: | :--------: | :--------: | :------: | :------: | :--------: | :--------: | :-------: | :-------: | :-------: | :-------: | :------: | :------: |
| dots.mocr (3B)    | 2025.07 |   28.3   |   40.9   |   41.8    |   60.1    |   68.8   | **78.3** |  **20.3**  |  **43.1**  |   24.1   |   16.0   |  **26.9**  |    47.1    |   26.2    |   20.6    |   28.7    |   19.6    |   33.1   |   40.7   |
| PaddleOCR-VL (1B) | 2025.10 |   31.8   |   49.3   |   43.0    |   51.6    |   57.5   |   75.2   |    14.4    |    29.0    |   11.7   |   20.7   |    21.3    |  **54.0**  |    --     |    --     |    --     |    --     |   23.9   |   35.8   |
| HunyuanOCR (1B)   | 2025.11 | **33.0** | **60.0** | **49.5**  | **68.2**  | **71.0** |   74.8   |    19.0    |    41.1    | **43.9** | **45.2** |    20.1    |    50.8    | **39.9**  | **35.9**  | **55.0**  | **46.6**  | **41.4** | **52.8** |

### Expert Chart Understanding Models <!-- omit in toc -->

| Model           |  Date   | Bar (EN) | Bar (ZH) | Line (EN) | Line (ZH) | Pie (EN) | Pie (ZH) | Radar (EN) | Radar (ZH) | Box (EN) | Box (ZH) | Combo (EN) | Combo (ZH) | Flow (EN) | Flow (ZH) | Mind (EN) | Mind (ZH) | Avg (EN) | Avg (ZH) |
| :-------------- | :-----: | :------: | :------: | :-------: | :-------: | :------: | :------: | :--------: | :--------: | :------: | :------: | :--------: | :--------: | :-------: | :-------: | :-------: | :-------: | :------: | :------: |
| ChartAst (13B)  | 2024.01 |   5.2    |    --    |    4.2    |    --     |   0.3    |    --    |    1.5     |     --     |   0.3    |    --    |    0.0     |     --     |    --     |    --     |    --     |    --     |   1.4    |    --    |
| ChartVLM (8.3B) | 2024.02 |   11.2   |   5.3    |   11.5    |    4.3    |   12.9   |   8.2    |    2.1     |    5.0     |   0.7    |   0.4    |    4.1     |    4.4     |    --     |    --     |    --     |    --     |   5.3    |   3.5    |
| TinyChart (3B)  | 2024.04 |   6.1    |   6.3    |    9.7    |    3.2    |   5.7    |   5.4    |    0.5     |    3.4     |   0.2    |   1.3    |    0.7     |    4.2     |    --     |    --     |    --     |    --     |   2.9    |   3.0    |
| ChartMoE (8B)   | 2024.09 |   18.7   |   24.4   |   14.7    |   22.3    |   15.0   |   48.5   |    3.7     |    16.1    |   2.7    |   1.6    |    5.1     |    19.5    |    4.0    |    --     |    4.1    |    --     |   8.5    |   16.7   |
| ChartCoder (7B) | 2025.01 |   23.2   |   12.6   |   22.0    |   19.6    |   34.3   |   16.7   |    5.5     |    13.9    |   5.4    |   11.4   |    3.7     |    5.1     |    5.6    |    --     |    1.0    |    --     |   12.6   |   9.9    |
| RRVF (7B)       | 2025.07 | **35.8** | **66.5** | **41.5**  | **54.3**  | **51.6** | **75.3** |    16.6    |    40.3    | **14.7** | **14.1** |  **23.5**  |  **61.2**  | **36.4**  | **32.4**  | **68.4**  | **63.8**  | **36.0** | **51.0** |
| MSRL (7B)       | 2025.08 |   32.7   |   45.2   |   35.2    |   34.3    |   41.2   |   67.9   |  **25.9**  |  **48.0**  |   11.2   |   13.0   |    16.7    |    35.2    |   23.2    |   12.4    |   31.0    |   18.8    |   27.1   |   34.3   |

</details>

## Task Definitions

ChartArena groups charts into three categories, each with a default extraction task:

| Chart Category             | Examples                                 | Default Task |
| -------------------------- | ---------------------------------------- | ------------ |
| Numerical charts           | Bar / Line / Pie / Radar / Box / Combo … | SE_MD        |
| Mind maps (logic diagrams) | Tree / hierarchy diagrams                | SE_MD        |
| Flowcharts                 | Process / workflow diagrams              | SE_MERMAID   |

<details>
<summary>The eleven extraction tasks (click to expand)</summary>

| Task         | Output Format         | Description                                                         |
| ------------ | --------------------- | ------------------------------------------------------------------- |
| SE_MD        | Markdown table / list | Numerical charts → Markdown table; mind maps → Markdown nested list |
| SE_JSON      | JSON                  | Structured JSON with `title` and `values`                           |
| SE_CSV       | CSV                   | Comma-separated values                                              |
| SE_CODE      | Python (matplotlib)   | Reproduce the chart as executable Python code                       |
| SE_SVG       | SVG                   | Reproduce the chart as SVG markup                                   |
| SE_MERMAID   | Mermaid               | Flowchart as Mermaid diagram syntax                                 |
| SE_GRAPHVIZ  | Graphviz DOT          | Flowchart as DOT language                                           |
| SE_PLANTUML  | PlantUML              | Flowchart as PlantUML syntax                                        |
| SE_DIAGRAMS  | diagrams.net XML      | Flowchart as draw.io XML                                            |
| SE_D2        | D2                    | Flowchart as D2 diagram language                                    |
| SE_CYTOSCAPE | Cytoscape JSON        | Flowchart as Cytoscape.js JSON                                      |

</details>

Scoring metrics: **mAP** (map_strict / map_slight / map_high) and **EM** (exact match).

## Getting Started

### 1. Setup <!-- omit in toc -->

```bash
git clone <this-repo>
cd ChartArena
pip install -r requirements.txt
# Optional: only if you plan to use --api_type local_vllm
pip install vllm
```

### 2. Download benchmark data <!-- omit in toc -->

The dataset (jsonl + images) is released as a single archive. Place the files under `data/`:

```
data/
├── ChartArena.jsonl
└── images/
    ├── bar/...
    ├── line/...
    ├── pie/...
    └── ...
```

Each line of the jsonl looks like:

```json
{
  "img_path": "images/xxx.png",
  "chart_type": "柱状图",
  "img_type": "电子印刷",
  "lang_type": "中文",
  "anno": "..."
}
```

`img_path` is a relative path from the `data/` directory and is used as the unique key throughout the pipeline.

### 3. Inference <!-- omit in toc -->

Two backends are supported via `--api_type`: `openai_compat` for any OpenAI-compatible HTTP service (local or cloud), and `local_vllm` for in-process model loading. Inference supports **resume** — re-running the same command skips already-completed samples.

<details>
<summary>Backend details and task selection (click to expand)</summary>

#### (a) `openai_compat` — any OpenAI-compatible HTTP service <!-- omit in toc -->

Works with locally-served models (`vllm serve`, `sglang`, `lmdeploy`) **or** public APIs that speak the OpenAI Chat Completions protocol (OpenAI, Gemini, Claude, Together, …).

```bash
python infer.py \
    --api_type openai_compat \
    --model_name Qwen2.5-VL-72B-Instruct \
    --base_url http://127.0.0.1:8000/v1 \
    --api_key EMPTY \
    --max_workers 64
```

#### (b) `local_vllm` — in-process vLLM, give it a model path <!-- omit in toc -->

No need to start a server first. The script loads the checkpoint directly with `vllm.LLM`.

```bash
python infer.py \
    --api_type local_vllm \
    --model_path /path/to/Qwen2.5-VL-72B-Instruct \
    --tensor_parallel_size 4 \
    --max_model_len 32768
```

#### Task selection <!-- omit in toc -->

By default, each chart category runs one task. You can override with `--task_data`, `--task_logic`, `--task_flowchart` (each accepts one or more task names):

```bash
# Run SE_MD and SE_JSON for numerical charts, SE_MERMAID for flowcharts
python infer.py --api_type openai_compat --model_name ... --base_url ... \
    --task_data SE_MD SE_JSON \
    --task_flowchart SE_MERMAID
```

#### Output <!-- omit in toc -->

Each run writes one jsonl file:

```
infer_outputs/<model_tag>/results.jsonl
```

`<model_tag>` defaults to `--model_name` / basename of `--model_path`. You can override it with `--output_tag`.

</details>

### 4. Judging <!-- omit in toc -->

```bash
# Score all models under infer_outputs/
python judge.py

# Score specific models only
python judge.py --models Qwen2.5-VL-72B-Instruct gemini-2.5-pro

# Force re-score a specific task (e.g. after a scoring algorithm update)
python judge.py --force_rejudge SE_MERMAID
```

Outputs to `judge_outputs/<model_tag>/results.jsonl`. The judge step is purely rule-based and fast.

### 5. Analysis report <!-- omit in toc -->

```bash
python analyze.py
# → judge_outputs/results_analysis.xlsx
```

Scores are shown to 3 decimal places (e.g. `0.873`).

<details>
<summary>Workbook contents (click to expand)</summary>

- **Task overview** (Sheet 1) — per-model average score for each task
- **Per-task sheets** — model × source file breakdown for each task
- **By chart type** (`by_chart_type/`) — one Excel per task, one sheet per chart type
- **Detailed breakdown** (`detail_by_category/`) — per-model per-task breakdown by `(chart_type, img_type, lang_type)`

</details>

### 6. End-to-end example <!-- omit in toc -->

```bash
# 1. Run inference
python infer.py \
    --api_type openai_compat \
    --model_name Qwen2.5-VL-72B-Instruct \
    --base_url http://127.0.0.1:8000/v1

# 2. Score
python judge.py

# 3. Generate Excel report
python analyze.py
```

### 7. Repo layout <!-- omit in toc -->

The repository is organized around a three-stage pipeline (inference → judging → analysis), with pluggable API backends, per-format scoring modules, and shared metric utilities.

<details>
<summary>Full directory tree (click to expand)</summary>

```
ChartArena/
├── README.md / README_zh.md
├── requirements.txt
├── data/                        # ← download benchmark data here
├── infer_outputs/               # inference results (auto-created)
├── judge_outputs/               # scoring results  (auto-created)
├── apis/
│   ├── base.py                  # APIBase abstract class
│   ├── openai_compat.py         # OpenAI-compatible client
│   └── local_vllm.py            # in-process vLLM
├── methods/
│   ├── prompts.py               # prompt templates
│   ├── context.py               # context building utilities
│   ├── normalize.py             # output normalization
│   ├── scoring.py               # scoring entry points
│   └── parsers/                 # per-format output parsers
├── metrics/
│   ├── SCRM.py                  # core MAP / EM metric
│   ├── tree_eval.py             # Markdown list evaluation
│   ├── mermaid_eval.py          # Mermaid diagram evaluation
│   ├── flowchart_common.py      # flowchart multi-format evaluation
│   └── dsl_parsers/             # DSL-specific parser utilities
├── utils/
│   ├── io.py                    # ResultWriter (thread-safe incremental writer)
│   ├── signal_utils.py          # graceful Ctrl+C shutdown
│   └── image_utils.py           # base64 encoding for OpenAI-compat
├── infer.py                     # entry: inference
├── judge.py                     # entry: rule-based scoring
└── analyze.py                   # entry: Excel analysis report
```

</details>

## Citation

```bibtex
@article{peng2026chartarena,
  title   = {{ChartArena}: Benchmarking Chart Parsing across Languages, Scenarios, and Formats},
  author  = {Peng, Shangpin and Li, Gengluo and Wan, Xingyu and Zhang, Chengquan and Feng, Hao and Wu, Binghong and Shen, Huawen and Wang, Weinong and Cai, Ziyi and Tian, Zhuotao and others},
  journal = {arXiv preprint arXiv:2606.01348},
  year    = {2026}
}
```

## License

This benchmark is released for **research purposes only**.
