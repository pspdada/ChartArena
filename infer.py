"""ChartArena benchmark 推理入口。

对 benchmark 中的每张图表调用 VLM，将结构化提取结果保存到 infer_outputs/ 目录。

用法:
    # 1) 本地 OpenAI 兼容服务（vllm serve / sglang / lmdeploy 等）
    python infer.py --api_type openai_compat \\
        --model_name Qwen2.5-VL-72B-Instruct \\
        --base_url http://127.0.0.1:8000/v1

    # 2) 进程内 vLLM（直接加载本地权重）
    python infer.py --api_type local_vllm \\
        --model_path /path/to/checkpoint \\
        --tensor_parallel_size 4
输出:
    infer_outputs/<model_tag>/results.jsonl
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from apis import API_TYPES, get_api  # noqa: E402
from methods.prompts import INFER_EXTRACT_FUNC, get_prompts_for_api  # noqa: E402
from utils.io import ResultWriter  # noqa: E402
from utils.signal_utils import ABORT_EVENT, install_signal_handlers_once  # noqa: E402

# ============================================================
# 默认配置
# ============================================================
DEFAULT_DATA_FILE = REPO_ROOT / "data" / "ChartArena.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "infer_outputs"

# 三类图表对应的默认任务
# - 数值类图表（柱状图、折线图、饼图等）：SE_MD（Markdown 表格）
# - 思维导图（逻辑结构图）：SE_MD（Markdown 多级列表）
# - 流程图：SE_MERMAID
DEFAULT_TASK_DATA = "SE_MD"  # 数值类图表任务
DEFAULT_TASK_LOGIC = "SE_MD"  # 思维导图任务
DEFAULT_TASK_FLOWCHART = "SE_MERMAID"  # 流程图任务


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ChartArena 推理入口")

    # API 选择
    p.add_argument(
        "--api_type",
        choices=API_TYPES,
        required=True,
        help="local_vllm: 进程内 vllm.LLM; openai_compat: 标准 OpenAI 协议",
    )

    # openai_compat 参数
    p.add_argument("--model_name", type=str, default=None, help="openai_compat 调用时使用的 model 字段")
    p.add_argument("--base_url", type=str, default=None, help="openai_compat 服务地址，例如 http://127.0.0.1:8000/v1")
    p.add_argument("--api_key", type=str, default="EMPTY")

    # local_vllm 参数
    p.add_argument("--model_path", type=str, default=None, help="local_vllm: 本地模型权重路径")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    # 数据 / 输出
    p.add_argument(
        "--data_file",
        type=str,
        default=str(DEFAULT_DATA_FILE),
        help=f"benchmark jsonl 路径，默认 {DEFAULT_DATA_FILE}",
    )
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument(
        "--output_tag",
        type=str,
        default=None,
        help="结果子目录名，默认从 model_name / model_path / api_name 推断",
    )

    # 任务参数（每类图表可指定一个或多个任务）
    p.add_argument(
        "--task_data",
        type=str,
        nargs="+",
        default=[DEFAULT_TASK_DATA],
        help=f"数值类图表（柱状图/折线图/饼图等）的任务，默认 {DEFAULT_TASK_DATA}",
    )
    p.add_argument(
        "--task_logic",
        type=str,
        nargs="+",
        default=[DEFAULT_TASK_LOGIC],
        help=f"思维导图（逻辑结构图）的任务，默认 {DEFAULT_TASK_LOGIC}",
    )
    p.add_argument(
        "--task_flowchart",
        type=str,
        nargs="+",
        default=[DEFAULT_TASK_FLOWCHART],
        help=f"流程图的任务，默认 {DEFAULT_TASK_FLOWCHART}",
    )

    # 推理参数
    p.add_argument("--max_workers", type=int, default=64)
    p.add_argument("--max_try", type=int, default=3)
    p.add_argument("--max_rows", type=int, default=-1)
    p.add_argument("--save_interval", type=int, default=50)
    p.add_argument("--debug", action="store_true")

    return p.parse_args()


def build_api(args: argparse.Namespace):
    """根据 api_type 构造 API 实例。"""
    if args.api_type == "openai_compat":
        if not args.model_name or not args.base_url:
            raise SystemExit("--api_type openai_compat 需要同时提供 --model_name 与 --base_url")
        return get_api(
            "openai_compat",
            model_name=args.model_name,
            base_url=args.base_url,
            api_key=args.api_key,
            max_try=args.max_try,
        )
    elif args.api_type == "local_vllm":
        if not args.model_path:
            raise SystemExit("--api_type local_vllm 需要提供 --model_path")
        return get_api(
            "local_vllm",
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_try=args.max_try,
        )
    else:
        raise ValueError(f"不支持的 api_type: {args.api_type}")


def derive_output_tag(args: argparse.Namespace) -> str:
    """推断输出子目录名。"""
    if args.output_tag:
        return args.output_tag
    if args.api_type == "openai_compat" and args.model_name:
        return args.model_name
    if args.api_type == "local_vllm" and args.model_path:
        return Path(args.model_path).name
    return "default"


def resolve_image_path(row: dict, data_dir: Path) -> str:
    """将 jsonl 中的相对路径拼成绝对路径。"""
    rel = row.get("img_path", "")
    if not rel:
        return ""
    if os.path.isabs(rel):
        return rel
    return str(data_dir / rel)


def tasks_for_row(row: dict, task_data: list[str], task_logic: list[str], task_flowchart: list[str]) -> list[str]:
    """根据 chart_type 决定该样本应跑的任务列表。

    - 流程图 → task_flowchart（SE_MERMAID 等）
    - 思维导图 → task_logic（SE_MD 等）
    - 其他数值类图表 → task_data（SE_MD 等）
    """
    chart_type = str(row.get("chart_type", "") or "").strip()
    if chart_type == "流程图":
        return task_flowchart
    if chart_type == "思维导图":
        return task_logic
    return task_data


def process_one_row(
    api_instance,
    row: dict,
    abs_img_path: str,
    task_data: list[str],
    task_logic: list[str],
    task_flowchart: list[str],
    existing: dict,
    max_retries: int,
) -> dict | None:
    """对单条样本跑所有未完成的任务。返回新 row（包含合并后的 infer_results）。"""
    if not abs_img_path or not os.path.exists(abs_img_path):
        print(f"警告：图片不存在 {abs_img_path}")
        return None

    chart_type = str(row.get("chart_type", "") or "").strip()
    file_tasks = tasks_for_row(row, task_data, task_logic, task_flowchart)
    pending = [t for t in file_tasks if t not in existing]
    if not pending:
        return None

    infer_results = dict(existing)

    for task_name in pending:
        # 根据 chart_type 获取对应的 prompt
        prompts = get_prompts_for_api([task_name], chart_type=chart_type)
        prompt_text = prompts.get(task_name)
        if not prompt_text:
            continue

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                ok, thinking, answer = api_instance(abs_img_path, prompt_text)
                if not ok or answer is None:
                    raise RuntimeError("API 调用失败或返回空结果")

                extract_fn = INFER_EXTRACT_FUNC.get(task_name)
                extract_ok, extracted = False, None
                if extract_fn is not None:
                    try:
                        extract_ok, extracted = extract_fn(answer)
                    except Exception as e:
                        print(f"  任务 '{task_name}' 提取异常: {e}")
                        extract_ok = False

                if extract_fn is not None and not extract_ok and attempt < max_retries:
                    print(f"  任务 '{task_name}' 提取失败，重试 {attempt}/{max_retries}")
                    time.sleep(2)
                    continue

                rec = {"thinking": thinking or "", "answer": answer}
                if extract_ok:
                    rec["extract"] = extracted
                infer_results[task_name] = rec
                break
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    print(f"  任务 '{task_name}' 失败 ({attempt}/{max_retries}): {last_error}")
                    time.sleep(2)
                else:
                    infer_results[task_name] = {"thinking": "", "answer": "", "error": last_error}

    result = dict(row)
    result["infer_results"] = infer_results
    return result


def main() -> None:
    args = parse_args()

    data_file = Path(args.data_file).resolve()
    if not data_file.is_file():
        raise SystemExit(f"benchmark 文件不存在: {data_file}")
    data_dir = data_file.parent

    output_tag = derive_output_tag(args)
    output_dir = Path(args.output_dir).resolve() / output_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.jsonl"

    print("=" * 72)
    print("ChartArena Inference")
    print("=" * 72)
    print(f"api_type      : {args.api_type}")
    print(f"output_tag    : {output_tag}")
    print(f"data_file     : {data_file}")
    print(f"output_file   : {output_file}")
    print(f"max_workers   : {args.max_workers}")
    print(f"max_rows      : {args.max_rows if args.max_rows > 0 else 'all'}")
    print(f"task_data     : {args.task_data}")
    print(f"task_logic    : {args.task_logic}")
    print(f"task_flowchart: {args.task_flowchart}")

    # 读 jsonl
    rows: list[dict] = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    if args.debug:
        rows = rows[: min(5, len(rows))]
    print(f"loaded {len(rows)} rows")

    # 初始化 API
    print("\n初始化 API...")
    api_instance = build_api(args)
    print("API 就绪")

    # 历史结果（增量）：读取已落盘的结果，按 image_path 建索引
    processed: dict[str, dict] = {}
    if output_file.is_file():
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
                    if key:
                        processed[key] = item
        except Exception as e:
            print(f"读取历史结果时出错: {e}")
    print(f"历史结果: 已写入 {len(processed)} 条")

    # 待处理列表（按 chart_type 路由后判断是否已完成）
    pending: list[tuple[dict, str, dict]] = []
    fully_done = 0
    for row in rows:
        rel = row.get("img_path", "")
        if not rel:
            continue
        abs_img = resolve_image_path(row, data_dir)
        existing_infer = processed.get(rel, {}).get("infer_results", {})
        file_tasks = set(tasks_for_row(row, args.task_data, args.task_logic, args.task_flowchart))
        if file_tasks.issubset(set(existing_infer.keys())):
            fully_done += 1
            continue
        pending.append((row, abs_img, existing_infer))

    print(f"完全完成: {fully_done}, 待处理: {len(pending)}\n")
    if not pending:
        print("没有需要处理的数据")
        return

    install_signal_handlers_once()
    writer = ResultWriter(str(output_file), processed, save_interval=args.save_interval)

    executor = ThreadPoolExecutor(max_workers=args.max_workers)
    aborted = False
    try:
        futures = {
            executor.submit(
                process_one_row,
                api_instance,
                row,
                abs_img,
                args.task_data,
                args.task_logic,
                args.task_flowchart,
                existing,
                args.max_try,
            ): row
            for row, abs_img, existing in pending
        }
        pbar = tqdm.tqdm(total=len(futures), desc="inference")
        for fut in concurrent.futures.as_completed(futures):
            if ABORT_EVENT.is_set():
                aborted = True
                break
            try:
                result = fut.result()
                if result:
                    writer.update_and_save(result)
            except Exception as e:
                print(f"\n处理失败: {e}")
                traceback.print_exc()
            pbar.update(1)
        pbar.close()
        if aborted:
            for f in futures:
                if not f.done():
                    f.cancel()
    finally:
        if ABORT_EVENT.is_set():
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True)

    print("\n落盘最终结果...")
    writer.finalize()
    print(f"✅ 推理完成: {output_file}")
    if ABORT_EVENT.is_set():
        sys.exit(130)


if __name__ == "__main__":
    main()
