# ChartArena <!-- omit in toc -->

**面向图表族系、视觉场景与输出格式的综合性双语通用图表解析基准**

<p align="center">
  <a href="README.md">English</a> •
  <a href="https://arxiv.org/abs/2606.01348">论文</a> •
  <a href="https://github.com/pspdada/ChartArena">GitHub 仓库</a> •
  <a href="https://huggingface.co/datasets/psp-dada/ChartArena">HuggingFace 数据集</a> •
  <a href="https://modelscope.cn/datasets/pspdada/ChartArena">ModelScope 数据集</a>
</p>

## 新闻 <!-- omit in toc -->

- [2026.07.07] 🚀 我们新增了评测模型，覆盖更大范围的参数量与模型族系，并同步更新了 arXiv 论文，欢迎查看！
- [2026.06.01] 📖 代码和数据已发布！

## 概览 <!-- omit in toc -->

**ChartArena** 是一个面向视觉语言模型**图表解析（chart parsing）**能力评测的综合性双语基准，覆盖了实际场景中图表的完整难度谱系。它涵盖**八类图表族系**：既包括数值类图表（柱状图、折线图、饼图、雷达图、箱线图、组合图），也包括结构类图表（流程图、思维导图），每一类都呈现在**三种视觉场景**（电子渲染、印刷照片、手绘照片）和**两种语言**（中文与英文）之下。

为了在输出格式互不兼容的模型之间实现公平比较，ChartArena 采用了一套**格式无关的评测协议**：将异构的预测结果归一化到两个标准语义空间：数值类图表的三元组视图（triple view）与结构类图表的有向图视图（directed graph view），并使用结构感知的指标进行评分。

<table align="center">
    <p align="center">
      <img src="/docs/figures/ChartArena_overview.jpg" width="80%" />
    </p>
</table>

### 任务覆盖对比 <!-- omit in toc -->

与现有的图表解析基准相比，**ChartArena** 在图表类型、视觉场景和语言三个维度上提供了最全面的覆盖，从而支持对图表解析进行真实且全面的评测。

| Benchmark             |  Date   |  Size  | Bar | Line | Pie | Radar | Box Plot | Comb. | Flowchart | Mind Map | Digital | Printed | Hand-drawn | English | Chinese |
| :-------------------- | :-----: | :----: | :-: | :--: | :-: | :---: | :------: | :---: | :-------: | :------: | :-----: | :-----: | :--------: | :-----: | :-----: |
| PlotQA-SE             | 2019.09 | 33,657 |  ✓  |  ✓   |     |       |          |       |           |          |    ✓    |         |            |    ✓    |         |
| ChartQA-SE            | 2022.03 | 1,509  |  ✓  |  ✓   |  ✓  |       |          |       |           |          |    ✓    |         |            |    ✓    |         |
| MMC-Bench             | 2023.11 | 1,063  |  ✓  |  ✓   |  ✓  |   ✓   |          |       |           |          |    ✓    |         |            |    ✓    |         |
| ChartX-SE             | 2024.02 | 1,152  |  ✓  |  ✓   |  ✓  |   ✓   |    ✓     |       |           |          |    ✓    |         |            |    ✓    |         |
| ChartY                | 2024.04 | 6,048  |  ✓  |  ✓   |  ✓  |       |          |   ✓   |           |          |    ✓    |         |            |    ✓    |    ✓    |
| VG-DCU                | 2024.04 | 3,044  |  ✓  |  ✓   |  ✓  |       |    ✓     |   ✓   |           |          |    ✓    |         |            |    ✓    |         |
| ChartP-Bench          | 2026.02 | 1,200  |  ✓  |  ✓   |     |       |          |       |           |          |    ✓    |         |            |    ✓    |         |
| ParseBench            | 2026.04 | 1,039  |  ✓  |  ✓   |  ✓  |       |          |   ✓   |           |          |    ✓    |         |            |    ✓    |         |
| ExChart-Bench         | 2026.04 | 3,600  |  ✓  |  ✓   |  ✓  |   ✓   |          |       |           |          |    ✓    |         |            |    ✓    |         |
| **ChartArena (ours)** | 2026.05 | 2,400  |  ✓  |  ✓   |  ✓  |   ✓   |    ✓     |   ✓   |     ✓     |    ✓     |    ✓    |    ✓    |     ✓      |    ✓    |    ✓    |

## 目录 <!-- omit in toc -->

- [基准统计](#基准统计)
- [排行榜](#排行榜)
- [任务定义](#任务定义)
- [快速开始](#快速开始)
- [引用](#引用)
- [许可证](#许可证)

## 基准统计

| 项目     | 详情                                                                   |
| -------- | ---------------------------------------------------------------------- |
| 图表族系 | 8 类（柱状图、折线图、饼图、雷达图、箱线图、组合图、流程图、思维导图） |
| 图表大类 | 数值类图表、思维导图、流程图                                           |
| 视觉场景 | 3 种（电子渲染、印刷照片、手绘照片）                                   |
| 语言     | 双语（中文与英文）                                                     |

## 排行榜

我们评测了 26 个模型，涵盖通用多模态大模型、文档解析专用模型和图表专家模型三类。以 **mAP$_{high}$** 为主要指标，分别报告 **EN**（英文）和 **ZH**（中文）得分，每个得分均在三种视觉场景上取平均。各类别内**加粗**表示该列最优结果。

<details>
<summary>完整排行榜（点击展开）</summary>

### 通用多模态大模型（General-Purpose MLLMs） <!-- omit in toc -->

| Model                       |  Date   | Bar (EN) | Bar (ZH) | Line (EN) | Line (ZH) | Pie (EN) | Pie (ZH) | Radar (EN) | Radar (ZH) | Box (EN) | Box (ZH) | Combo (EN) | Combo (ZH) | Flow (EN) | Flow (ZH) | Mind (EN) | Mind (ZH) | Avg (EN) | Avg (ZH) |
| :-------------------------- | :-----: | :------: | :------: | :-------: | :-------: | :------: | :------: | :--------: | :--------: | :------: | :------: | :--------: | :--------: | :-------: | :-------: | :-------: | :-------: | :------: | :------: |
| GPT-4o                      | 2024.05 |   21.6   |   36.3   |   27.5    |   52.9    |   76.7   |   74.2   |    9.7     |    24.9    |   19.1   |   9.6    |    9.9     |    40.7    |   49.8    |   27.1    |   64.0    |   24.8    |   34.8   |   36.3   |
| GPT-5                       | 2025.08 |   35.1   |   52.3   |   48.1    |   65.1    |   81.1   |   78.9   |  **32.0**  |    41.5    |   19.8   |   12.8   |    14.2    |    46.5    |   58.1    |   35.3    |   76.6    |   33.5    |   45.6   |   45.8   |
| InternVL3.5-8B              | 2025.08 |   22.7   |   52.6   |   34.4    |   53.7    |   65.8   |   73.8   |    14.0    |    34.7    |   5.6    |   9.5    |    11.3    |    42.1    |   32.6    |   23.8    |   48.3    |   31.8    |   29.3   |   40.2   |
| InternVL3.5-241B-A28B       | 2025.08 |   27.5   |   57.2   |   41.3    |   55.7    |   77.7   |   83.3   |    15.2    |    41.4    |   18.7   |   21.6   |    17.7    |    47.8    |   43.8    |   36.6    |   62.6    |   45.5    |   38.0   |   48.6   |
| Qwen2.5-VL-7B-Instruct      | 2025.02 |   15.2   |   36.9   |   17.9    |   39.9    |   63.4   |   73.1   |    8.3     |    19.1    |   0.9    |   2.8    |    6.0     |    40.6    |   29.7    |   23.2    |   45.4    |   29.9    |   23.3   |   33.2   |
| Qwen2.5-VL-72B-Instruct     | 2025.02 |   27.1   |   53.3   |   38.2    |   66.7    |   73.5   |   77.0   |    10.9    |    38.5    |   15.0   |   15.3   |    14.3    |    50.5    |   50.1    |   43.6    |   63.8    |   55.0    |   36.6   |   50.0   |
| Qwen3-VL-8B-Instruct        | 2025.10 |   27.5   |   58.6   |   35.5    |   61.1    |   77.3   |   84.7   |    16.8    |    42.6    |   11.6   |   12.1   |    13.2    |    47.9    |   50.0    |   41.5    |   66.4    |   54.6    |   37.3   |   50.4   |
| Qwen3-VL-235B-A22B-Instruct | 2025.10 |   38.4   |   67.9   |   52.3    |   73.8    |   82.6   |   85.5   |    23.2    |    52.4    |   14.1   |   14.1   |    29.1    |    58.2    |   57.9    |   49.8    |   70.8    |   65.2    |   46.0   |   58.4   |
| Qwen3.5-35B-A3B (thinking)  | 2026.02 |   46.2   |   65.3   |   60.3    |   77.6    |   89.7   |   88.4   |    25.2    |    57.8    |   42.2   |   50.6   |    31.5    |    56.9    |   62.5    |   56.5    |   75.1    |   70.9    |   54.1   |   65.5   |
| GLM-4.5V                    | 2025.07 |   33.5   |   61.4   |   51.7    |   70.5    |   81.2   |   83.1   |    19.7    |    43.1    |   32.4   |   37.4   |    21.2    |    52.5    |   44.7    |   39.6    |   66.2    |   43.7    |   43.8   |   53.9   |
| Seed-1.8 (non-thinking)     | 2025.12 |   29.1   |   59.7   |   46.0    |   72.5    |   84.7   |   88.0   |    22.0    |    45.9    |   16.1   |   17.5   |    15.0    |    59.7    |   47.8    |   50.3    |   76.5    |   69.1    |   42.2   |   57.8   |
| Seed-2.0 Pro (non-thinking) | 2026.02 |   40.3   |   73.3   |   56.5    |   80.7    |   91.5   |   90.5   |    21.3    |    54.7    | **44.5** | **55.2** |    32.4    |    62.2    |   62.6    |   61.3    |   83.1    | **85.8**  |   54.0   |   70.5   |
| Kimi K2.5 (non-thinking)    | 2026.02 |   45.2   |   70.3   |   60.9    |   79.8    |   87.2   |   86.7   |    30.2    |    59.7    |   40.6   |   47.6   |    33.6    |    63.6    |   59.9    |   57.9    |   80.8    |   79.4    |   54.8   |   68.1   |
| MiMo-V2-Omni                | 2026.03 |   31.1   |   56.9   |   41.5    |   66.4    |   87.0   |   85.8   |    19.7    |    46.1    |   19.1   |   30.3   |    19.4    |    54.7    |   57.1    |   51.0    |   76.6    |   64.6    |   43.9   |   57.0   |
| Gemini 2.5 Pro              | 2025.03 |   46.0   |   76.5   |   56.5    |   77.6    |   88.6   |   87.3   |    17.5    |    53.0    |   10.2   |   22.1   |    28.7    |    57.6    |   62.1    |   57.8    |   71.7    |   67.1    |   47.7   |   62.4   |
| Gemini 3.1 Pro              | 2026.02 | **57.9** | **78.7** | **67.0**  | **85.3**  | **92.5** | **95.1** |    31.8    |  **62.7**  |   32.5   |   45.2   |  **39.7**  |  **70.3**  | **65.6**  | **63.1**  | **86.8**  |   85.2    | **59.2** | **73.2** |

### 文档解析多模态大模型（Document Parsing MLLMs） <!-- omit in toc -->

| Model             |  Date   | Bar (EN) | Bar (ZH) | Line (EN) | Line (ZH) | Pie (EN) | Pie (ZH) | Radar (EN) | Radar (ZH) | Box (EN) | Box (ZH) | Combo (EN) | Combo (ZH) | Flow (EN) | Flow (ZH) | Mind (EN) | Mind (ZH) | Avg (EN) | Avg (ZH) |
| :---------------- | :-----: | :------: | :------: | :-------: | :-------: | :------: | :------: | :--------: | :--------: | :------: | :------: | :--------: | :--------: | :-------: | :-------: | :-------: | :-------: | :------: | :------: |
| dots.mocr (3B)    | 2025.07 |   28.3   |   40.9   |   41.8    |   60.1    |   68.8   | **78.3** |  **20.3**  |  **43.1**  |   24.1   |   16.0   |  **26.9**  |    47.1    |   26.2    |   20.6    |   28.7    |   19.6    |   33.1   |   40.7   |
| PaddleOCR-VL (1B) | 2025.10 |   31.8   |   49.3   |   43.0    |   51.6    |   57.5   |   75.2   |    14.4    |    29.0    |   11.7   |   20.7   |    21.3    |  **54.0**  |    --     |    --     |    --     |    --     |   23.9   |   35.8   |
| HunyuanOCR (1B)   | 2025.11 | **33.0** | **60.0** | **49.5**  | **68.2**  | **71.0** |   74.8   |    19.0    |    41.1    | **43.9** | **45.2** |    20.1    |    50.8    | **39.9**  | **35.9**  | **55.0**  | **46.6**  | **41.4** | **52.8** |

### 图表专家模型（Expert Chart Understanding Models） <!-- omit in toc -->

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

## 任务定义

ChartArena 将图表分为三大类，每类对应一个默认提取任务：

| 图表类别               | 示例                                                | 默认任务   |
| ---------------------- | --------------------------------------------------- | ---------- |
| 数值类图表             | 柱状图 / 折线图 / 饼图 / 雷达图 / 箱线图 / 组合图 … | SE_MD      |
| 思维导图（逻辑结构图） | 树状图 / 层级图                                     | SE_MD      |
| 流程图                 | 流程 / 工作流图                                     | SE_MERMAID |

<details>
<summary>十一个提取任务（点击展开）</summary>

| 任务         | 输出格式             | 说明                                                     |
| ------------ | -------------------- | -------------------------------------------------------- |
| SE_MD        | Markdown 表格 / 列表 | 数值类图表 → Markdown 表格；思维导图 → Markdown 多级列表 |
| SE_JSON      | JSON                 | 包含 `title` 和 `values` 的结构化 JSON                   |
| SE_CSV       | CSV                  | 逗号分隔值                                               |
| SE_CODE      | Python（matplotlib） | 将图表还原为可执行的 Python 代码                         |
| SE_SVG       | SVG                  | 将图表还原为 SVG 标记语言                                |
| SE_MERMAID   | Mermaid              | 流程图 → Mermaid 图表语法                                |
| SE_GRAPHVIZ  | Graphviz DOT         | 流程图 → DOT 语言                                        |
| SE_PLANTUML  | PlantUML             | 流程图 → PlantUML 语法                                   |
| SE_DIAGRAMS  | diagrams.net XML     | 流程图 → draw.io XML                                     |
| SE_D2        | D2                   | 流程图 → D2 图表语言                                     |
| SE_CYTOSCAPE | Cytoscape JSON       | 流程图 → Cytoscape.js JSON                               |

</details>

评分指标：**mAP**（map_strict / map_slight / map_high）和 **EM**（精确匹配）。

## 快速开始

### 1. 环境安装 <!-- omit in toc -->

```bash
git clone <this-repo>
cd ChartArena
pip install -r requirements.txt
# 可选：仅当使用 --api_type local_vllm 时需要
pip install vllm
```

### 2. 下载评测数据 <!-- omit in toc -->

数据（jsonl + 图片）以单一压缩包发布。请将其解压到 `data/` 目录下：

```
data/
├── ChartArena.jsonl
└── images/...
```

jsonl 每一行的格式如下：

```json
{
  "img_path": "images/xxx.png",
  "chart_type": "柱状图",
  "img_type": "电子印刷",
  "lang_type": "中文",
  "anno": "..."
}
```

`img_path` 是相对于 `data/` 目录的相对路径，在整个流程中作为唯一主键使用。

### 3. 推理（Inference） <!-- omit in toc -->

通过 `--api_type` 切换两种后端：`openai_compat` 支持任意 OpenAI 兼容 HTTP 服务（本地部署或公有云接口），`local_vllm` 支持进程内直接加载本地权重。推理支持**断点续推**，中断后重新运行同一命令会自动跳过已完成的样本。

<details>
<summary>后端详情与任务选择（点击展开）</summary>

#### (a) `openai_compat` — 任意 OpenAI 兼容 HTTP 服务 <!-- omit in toc -->

适用于 `vllm serve` / `sglang` / `lmdeploy` 等本地服务，也适用于符合 OpenAI Chat Completions 协议的公有云接口（OpenAI、Gemini、Claude、Together 等）。

```bash
python infer.py \
    --api_type openai_compat \
    --model_name Qwen2.5-VL-72B-Instruct \
    --base_url http://127.0.0.1:8000/v1 \
    --api_key EMPTY \
    --max_workers 64
```

#### (b) `local_vllm` — 进程内加载 vLLM，直接给本地权重路径 <!-- omit in toc -->

不需要先启动服务，脚本会通过 `vllm.LLM` 在进程内加载本地 checkpoint。

```bash
python infer.py \
    --api_type local_vllm \
    --model_path /path/to/Qwen2.5-VL-72B-Instruct \
    --tensor_parallel_size 4 \
    --max_model_len 32768
```

#### 任务选择 <!-- omit in toc -->

默认每类图表跑一个任务。可通过 `--task_data`、`--task_logic`、`--task_flowchart` 覆盖（每个参数支持多个任务名）：

```bash
# 数值类图表同时跑 SE_MD 和 SE_JSON，流程图跑 SE_MERMAID
python infer.py --api_type openai_compat --model_name ... --base_url ... \
    --task_data SE_MD SE_JSON \
    --task_flowchart SE_MERMAID
```

#### 输出位置 <!-- omit in toc -->

每次运行会写入一个 jsonl：

```
infer_outputs/<model_tag>/results.jsonl
```

`<model_tag>` 默认取 `--model_name` / `--model_path` 的 basename，也可以用 `--output_tag` 显式覆盖。

</details>

### 4. 评分（Judging） <!-- omit in toc -->

```bash
# 评分 infer_outputs/ 下的全部模型
python judge.py

# 只评分指定模型
python judge.py --models Qwen2.5-VL-72B-Instruct gemini-2.5-pro

# 强制重评某个任务（评分算法升级后刷新历史结果）
python judge.py --force_rejudge SE_MERMAID
```

输出到 `judge_outputs/<model_tag>/results.jsonl`。评分阶段为纯规则计算，速度很快。

### 5. 分析报表（Analysis） <!-- omit in toc -->

```bash
python analyze.py
# → judge_outputs/results_analysis.xlsx
```

分数保留 3 位小数（例如 `0.873`）。

<details>
<summary>Excel 内容（点击展开）</summary>

- **任务总览**（第一个 Sheet）— 每个模型在每个 task 的总平均分
- **各 task 分表** — 模型 × 数据文件的评分明细
- **按图表类型拆分**（`by_chart_type/`）— 每个 task 一个 Excel，每个 chart_type 一个 Sheet
- **详细分类结果**（`detail_by_category/`）— 按 `(chart_type, img_type, lang_type)` 细分的每模型每任务结果

</details>

### 6. 完整流程示例 <!-- omit in toc -->

```bash
# 1. 推理
python infer.py \
    --api_type openai_compat \
    --model_name Qwen2.5-VL-72B-Instruct \
    --base_url http://127.0.0.1:8000/v1

# 2. 评分
python judge.py

# 3. 生成 Excel 报表
python analyze.py
```

### 7. 代码结构 <!-- omit in toc -->

代码库围绕三阶段流程（推理 → 评分 → 分析）组织，包含可插拔的 API 后端、各格式独立的评分模块以及共享的指标工具。

<details>
<summary>完整目录结构（点击展开）</summary>

```
ChartArena/
├── README.md / README_zh.md
├── requirements.txt
├── data/                        # ← 数据下载到这里
├── infer_outputs/               # 推理结果（自动创建）
├── judge_outputs/               # 评分结果（自动创建）
├── apis/
│   ├── base.py                  # APIBase 抽象基类
│   ├── openai_compat.py         # OpenAI 兼容客户端
│   └── local_vllm.py            # 进程内 vLLM
├── methods/
│   ├── prompts.py               # prompt 模板
│   ├── context.py               # 上下文构建工具
│   ├── normalize.py             # 输出归一化
│   ├── scoring.py               # 评分入口
│   └── parsers/                 # 各格式输出解析器
├── metrics/
│   ├── SCRM.py                  # 核心 MAP / EM 指标
│   ├── tree_eval.py             # Markdown 列表评估
│   ├── mermaid_eval.py          # Mermaid 图表评估
│   ├── flowchart_common.py      # 流程图多格式评估
│   └── dsl_parsers/             # DSL 专项解析工具
├── utils/
│   ├── io.py                    # ResultWriter（线程安全增量写入器）
│   ├── signal_utils.py          # 友好响应 Ctrl+C
│   └── image_utils.py           # OpenAI 兼容 API 的 base64 编码
├── infer.py                     # 入口：推理
├── judge.py                     # 入口：规则评分
└── analyze.py                   # 入口：Excel 分析报表
```

</details>

## 引用

```bibtex
@article{peng2026chartarena,
  title   = {{ChartArena}: Benchmarking Chart Parsing across Languages, Scenarios, and Formats},
  author  = {Peng, Shangpin and Li, Gengluo and Wan, Xingyu and Zhang, Chengquan and Feng, Hao and Wu, Binghong and Shen, Huawen and Wang, Weinong and Cai, Ziyi and Tian, Zhuotao and Hu, Han and Ma, Can and Zhou, Yu},
  journal = {arXiv preprint arXiv:2606.01348},
  year    = {2026}
}
```

## 许可证

本基准仅供**学术研究使用**。
