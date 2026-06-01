"""ChartArena benchmark 评分入口。

读取 infer.py 的推理结果，使用规则评分（无需 VLM API），
将评分结果保存到 judge_outputs/ 目录。

用法:
    # 评分所有模型
    python judge.py

    # 评分指定模型
    python judge.py --models Qwen2.5-VL-72B-Instruct gemini-2.5-pro

    # 强制重新评测某个任务（评分算法升级后刷新历史结果）
    python judge.py --force_rejudge SE_MERMAID

输出:
    judge_outputs/<model_tag>/results.jsonl
"""

import argparse
import concurrent.futures
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from methods.prompts import FLOWCHART_TASKS  # noqa: E402
from methods.scoring import JUDGE_FUNC  # noqa: E402
from utils.io import ResultWriter  # noqa: E402
from utils.signal_utils import ABORT_EVENT, install_signal_handlers_once  # noqa: E402

DEFAULT_INFER_DIR = REPO_ROOT / "infer_outputs"
DEFAULT_JUDGE_DIR = REPO_ROOT / "judge_outputs"

DEFAULT_TASKS = [
    "SE_MD",
    "SE_JSON",
    "SE_CSV",
    "SE_CODE",
    "SE_SVG",
    "SE_MERMAID",
    "SE_GRAPHVIZ",
    "SE_PLANTUML",
    "SE_DIAGRAMS",
    "SE_D2",
    "SE_CYTOSCAPE",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ChartArena 规则评分")
    p.add_argument("--infer_dir", type=str, default=str(DEFAULT_INFER_DIR), help="infer_outputs 目录")
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_JUDGE_DIR), help="judge_outputs 目录")
    p.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="只评分指定模型；不传则扫描 infer_dir 下所有子目录",
    )
    p.add_argument("--tasks", type=str, nargs="+", default=DEFAULT_TASKS, help="需要评分的任务列表")
    p.add_argument(
        "--force_rejudge",
        type=str,
        nargs="*",
        default=[],
        help="强制重评的 task 名称列表（评分算法升级后刷新历史结果）",
    )
    p.add_argument(
        "--skip_missing",
        action="store_true",
        default=False,
        help="跳过缺少推理/提取结果的任务（默认 False，即缺失时记零分）",
    )
    p.add_argument("--max_workers", type=int, default=64)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--max_rows", type=int, default=-1, help="每个文件最多处理的行数（调试用）")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def _applicable_tasks(chart_type: str, tasks: set[str]) -> set[str]:
    """根据 chart_type 筛出适用的任务子集。

    - 流程图：只保留 FLOWCHART_TASKS（SE_MERMAID / SE_GRAPHVIZ / SE_PLANTUML 等）
    - 其他图表：排除 FLOWCHART_TASKS
    """
    if (chart_type or "") == "流程图":
        return {t for t in tasks if t in FLOWCHART_TASKS}
    return {t for t in tasks if t not in FLOWCHART_TASKS}


def read_processed_judge(
    output_file: str,
    tasks: set[str],
    force_rejudge: set[str],
) -> tuple[dict[str, dict], set[str]]:
    """读取已评分结果，返回 (已处理数据字典, 需要补评的 image_path 集合)。"""
    processed: dict[str, dict] = {}
    needs: set[str] = set()
    if not os.path.isfile(output_file):
        return processed, needs
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = item.get("img_path", "")
                if not key:
                    continue
                judge_results = item.get("judge_results", {}) or {}
                # 强制重评：从已有结果中剔除对应 task
                if force_rejudge:
                    judge_results = {k: v for k, v in judge_results.items() if k not in force_rejudge}
                    item["judge_results"] = judge_results
                completed = set(judge_results.keys())
                expected = _applicable_tasks(item.get("chart_type", ""), tasks)
                if not expected.issubset(completed):
                    needs.add(key)
                processed[key] = item
    except Exception as e:
        print(f"读取已评分数据时出错: {e}")
    return processed, needs


def judge_one_row(
    row: dict,
    tasks: set[str],
    existing: dict,
    skip_missing: bool,
) -> dict | None:
    """对单条数据的所有未完成任务进行评分。"""
    img_path = row.get("img_path", "")
    judge_results = dict(existing)
    infer_results = row.get("infer_results", {}) or {}

    if not infer_results:
        return None

    chart_type = str(row.get("chart_type", "") or "").strip()
    completed = set(judge_results.keys())
    tasks_to_judge = _applicable_tasks(chart_type, tasks) - completed

    if not tasks_to_judge:
        return None

    for task_name in tasks_to_judge:
        if task_name not in infer_results:
            if skip_missing:
                continue
            judge_results[task_name] = {"score": {"score": 0.0}, "error": "no_infer"}
            continue

        task_result: dict = infer_results[task_name]
        extract_value = task_result.get("extract")
        if extract_value is None:
            answer = task_result.get("answer", "")
            if not answer:
                if skip_missing:
                    continue
                judge_results[task_name] = {"score": {"score": 0.0}, "error": "no_extract"}
                continue
            extract_value = {"extracted_table": answer}

        judge_func: Callable = JUDGE_FUNC.get(task_name)
        if not judge_func:
            continue

        try:
            score = judge_func(extract_value, row)
            judge_results[task_name] = score
        except Exception as e:
            print(f"  任务 '{task_name}' 评分失败 [{img_path}]: {e}")
            judge_results[task_name] = {"score": {"score": 0.0}, "error": str(e)}

    result = dict(row)
    result["judge_results"] = judge_results
    return result


def judge_one_model(
    model_tag: str,
    infer_file: Path,
    output_file: Path,
    tasks: set[str],
    force_rejudge: set[str],
    max_workers: int,
    save_interval: int,
    skip_missing: bool,
    max_rows: int,
    debug: bool,
) -> tuple[int, int, int]:
    """对单个模型的 infer 结果跑评分。返回 (total, judged, skipped)。"""
    rows: list[dict] = []
    with open(infer_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if max_rows > 0:
        rows = rows[:max_rows]
    if debug:
        rows = rows[: min(5, len(rows))]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    processed, needs = read_processed_judge(str(output_file), tasks, force_rejudge)
    writer = ResultWriter(str(output_file), processed, save_interval=save_interval)

    # 待评分列表
    pending: list[tuple[dict, dict]] = []
    fully_done = 0
    for row in rows:
        key = row.get("img_path", "")
        existing_judge = processed.get(key, {}).get("judge_results", {})
        expected = _applicable_tasks(row.get("chart_type", ""), tasks)
        if expected.issubset(set(existing_judge.keys())):
            fully_done += 1
            continue
        pending.append((row, existing_judge))

    print(f"  [{model_tag}] total={len(rows)}, 完全完成={fully_done}, 待评分={len(pending)}")
    if not pending:
        writer.finalize()
        return len(rows), 0, 0

    judged = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(judge_one_row, row, tasks, existing, skip_missing): row for row, existing in pending}
        pbar = tqdm.tqdm(total=len(futures), desc=f"judge[{model_tag}]")
        for fut in concurrent.futures.as_completed(futures):
            if ABORT_EVENT.is_set():
                break
            try:
                result = fut.result()
                if result:
                    writer.update_and_save(result)
                    judged += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"\n评分失败: {e}")
                traceback.print_exc()
                skipped += 1
            pbar.update(1)
        pbar.close()
    writer.finalize()
    return len(rows), judged, skipped


def compute_summary(output_file: str) -> dict:
    """计算单个输出文件的汇总评分（按 task 拆分）。"""
    if not os.path.isfile(output_file):
        return {}
    scores: dict[str, list[float]] = {}
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                for task_name, task_judge in (item.get("judge_results") or {}).items():
                    score_dict = task_judge.get("score", {})
                    if isinstance(score_dict, dict) and "map_strict" in score_dict:
                        scores.setdefault(task_name, []).append(score_dict["map_strict"])
    except Exception as e:
        print(f"读取评分结果时出错: {e}")
    return {t: sum(v) / len(v) for t, v in scores.items() if v}


def main() -> None:
    args = parse_args()

    infer_dir = Path(args.infer_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not infer_dir.is_dir():
        raise SystemExit(f"infer 目录不存在: {infer_dir}")

    print("=" * 72)
    print("ChartArena Judging")
    print("=" * 72)
    print(f"infer_dir  : {infer_dir}")
    print(f"output_dir : {output_dir}")

    # 确定要评分的任务集合
    tasks = {t for t in args.tasks if t in JUDGE_FUNC}
    force_rejudge = set(args.force_rejudge or [])
    tasks |= {t for t in force_rejudge if t in JUDGE_FUNC}
    print(f"任务 ({len(tasks)}): {sorted(tasks)}")

    if not tasks:
        raise SystemExit("没有有效的评分任务")

    # 确定模型列表
    if args.models:
        model_tags = args.models
    else:
        model_tags = sorted(d.name for d in infer_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    print(f"模型数量: {len(model_tags)} -> {model_tags}\n")

    install_signal_handlers_once()

    summary: list[tuple[str, int, int, int]] = []
    for tag in model_tags:
        if ABORT_EVENT.is_set():
            break
        infer_file = infer_dir / tag / "results.jsonl"
        if not infer_file.is_file():
            print(f"[{tag}] 跳过：找不到 {infer_file}")
            continue
        output_file = output_dir / tag / "results.jsonl"
        total, judged, skipped = judge_one_model(
            tag,
            infer_file,
            output_file,
            tasks,
            force_rejudge,
            max_workers=args.max_workers,
            save_interval=args.save_interval,
            skip_missing=args.skip_missing,
            max_rows=args.max_rows,
            debug=args.debug,
        )
        summary.append((tag, total, judged, skipped))

        # 打印该模型的汇总分数
        scores = compute_summary(str(output_file))
        if scores:
            score_str = "  ".join(f"{t}: {v:.4f}" for t, v in sorted(scores.items()))
            print(f"  [{tag}] {score_str}")

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    for tag, total, judged, skipped in summary:
        print(f"  {tag:40s}  total={total:6d}  judged={judged:6d}  skipped={skipped:6d}")
    print(f"\n✅ judge 全部完成，结果在 {output_dir}")

    if ABORT_EVENT.is_set():
        sys.exit(130)


if __name__ == "__main__":
    main()
