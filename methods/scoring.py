"""评分结果构造 + _judge_by_task + JUDGE_FUNC 注册表。"""

from typing import Any, Callable

from metrics.flowchart_common import flowchart_eval_multi, parse_flowchart
from metrics.mermaid_eval import is_mermaid
from metrics.SCRM import csv_eval
from metrics.tree_eval import (
    is_markdown_list,
    normalize_tree,
    parse_markdown_list,
    tree_eval,
    tree_to_paths,
)

from .context import set_judge_context
from .normalize import (
    _strip_ref_header_for_svg,
    normalize_prediction_for_data,
    normalize_prediction_for_flowchart,
    normalize_prediction_for_logic,
    normalize_to_csv,
)


def _graph_ir_to_serializable(graph) -> dict:
    """将图 IR（nodes, edges, labeled_edges）序列化为可 JSON 存储的字典。

    只保存 label 级别的信息（用 label 替换 node_id），便于人工阅读和调试。
    格式：
        {
            "nodes": ["label1", "label2", ...],
            "edges": [["src_label", "dst_label", "edge_label"], ...]
        }
    """
    nodes, _edges, labeled_edges = graph
    node_labels = list(nodes.values())
    edge_list = [[nodes.get(s, s), nodes.get(d, d), lab or ""] for s, d, lab in labeled_edges]
    return {"nodes": node_labels, "edges": edge_list}


def _tree_paths_to_serializable(paths: list) -> list:
    """将路径集合序列化为可 JSON 存储的列表。

    格式：[["root", "child", "grandchild"], ...]
    """
    return [list(p) for p in paths]


# task → pred_dsl 映射（ref 统一为 Mermaid，由人工标注）
_TASK_TO_PRED_DSL: dict[str, str] = {
    "SE_MERMAID": "mermaid",
    "SE_GRAPHVIZ": "dot",
    "SE_PLANTUML": "plantuml",
    "SE_DIAGRAMS": "diagrams",
    "SE_D2": "d2",
    "SE_CYTOSCAPE": "cytoscape",
}


def _transpose_csv(csv_str: str, separator: str = " \\t ", delimiter: str = " \\n ") -> str:
    """转置内部 CSV 格式的表格（行列互换）。

    内部 CSV 格式：列用 ' \\t ' 分隔，行用 ' \\n ' 分隔。
    转置后行变列、列变行。

    仅当表格是"矩形"（所有行列数一致）时才转置，否则返回空字符串。
    """
    if not csv_str or not csv_str.strip():
        return ""

    rows = csv_str.split(delimiter)
    if len(rows) < 2:
        return ""

    # 解析为二维数组
    matrix = [row.split(separator) for row in rows]

    # 检查是否矩形
    ncols = len(matrix[0])
    if ncols < 2:
        return ""
    if not all(len(row) == ncols for row in matrix):
        # 尝试修复：截断到最小列数
        min_cols = min(len(row) for row in matrix)
        if min_cols < 2:
            return ""
        matrix = [row[:min_cols] for row in matrix]
        ncols = min_cols

    nrows = len(matrix)

    # 转置
    transposed = []
    for col_idx in range(ncols):
        new_row = [matrix[row_idx][col_idx] for row_idx in range(nrows)]
        transposed.append(new_row)

    # 重新组装为内部 CSV 格式
    return delimiter.join(separator.join(row) for row in transposed)


def _zero_score_dict(**extra) -> dict:
    """返回全零的评分字典"""
    d = {
        "em": 0.0,
        "map_strict": 0.0,
        "map_slight": 0.0,
        "map_high": 0.0,
        "ap_50_strict": 0.0,
        "ap_75_strict": 0.0,
        "ap_90_strict": 0.0,
        "ap_50_slight": 0.0,
        "ap_75_slight": 0.0,
        "ap_90_slight": 0.0,
        "ap_50_high": 0.0,
        "ap_75_high": 0.0,
        "ap_90_high": 0.0,
    }
    d.update(extra)
    return d


def _build_score_dict(scores_tuple: tuple, eval_logs: list[str] | None = None, **extra) -> dict:
    """将 csv_eval / tree_eval 返回的 13 元组转为字典"""
    (
        em,
        map_strict,
        map_slight,
        map_high,
        ap_50_strict,
        ap_75_strict,
        ap_90_strict,
        ap_50_slight,
        ap_75_slight,
        ap_90_slight,
        ap_50_high,
        ap_75_high,
        ap_90_high,
    ) = scores_tuple
    d = {
        "em": em,
        "map_strict": map_strict,
        "map_slight": map_slight,
        "map_high": map_high,
        "ap_50_strict": ap_50_strict,
        "ap_75_strict": ap_75_strict,
        "ap_90_strict": ap_90_strict,
        "ap_50_slight": ap_50_slight,
        "ap_75_slight": ap_75_slight,
        "ap_90_slight": ap_90_slight,
        "ap_50_high": ap_50_high,
        "ap_75_high": ap_75_high,
        "ap_90_high": ap_90_high,
    }
    d.update(extra)
    if eval_logs:
        d["eval_logs"] = eval_logs
    return d


# ============================================================
# 核心评分：根据 task / reference 格式路由到 csv_eval / tree_eval / flowchart_eval_multi
# ============================================================


def _judge_by_task(prediction: str, reference: str, task: str, easy: int = 1) -> dict:
    """根据任务类型 task 对单条数据评分。

    - 流程图类 task（SE_MERMAID / SE_GRAPHVIZ / SE_PLANTUML / SE_DIAGRAMS）
      直接按 task 路由到 flowchart_eval_multi（pred_dsl 按 task 映射，ref_dsl 固定 mermaid）；
    - 其他 task 按 reference 格式决定走 tree_eval / flowchart_eval_multi / csv_eval：
        * Markdown 无序列表 → tree_eval
        * Mermaid 流程图   → flowchart_eval_multi（mermaid ↔ mermaid）
        * 其他（表格/CSV）  → csv_eval
    - 然后按 task 类型把 prediction 转换到对应格式。
    """
    if not prediction:
        prediction = ""
    if not reference:
        reference = ""

    # ---- 分支 0: SE_MERMAID / SE_GRAPHVIZ / SE_PLANTUML / SE_DIAGRAMS → 统一多 DSL flowchart 分支 ----
    # pred 按各自 DSL 解析为图 IR，ref 统一按 mermaid 解析（人工标注保证）；
    # 解析失败 → 空图 → 自然 0 分 + parse_failed 日志。
    # SE_MERMAID 已经 A/B 验证（455 条 OmniChart flowchart 数据字节级一致），
    # 迁移到此统一路径无回归。
    if task in _TASK_TO_PRED_DSL:
        pred_dsl = _TASK_TO_PRED_DSL[task]
        if not prediction.strip() or not reference.strip():
            return _zero_score_dict(
                eval_mode="flowchart", task=task, pred_text=prediction, ref_text=reference, pred_dsl=pred_dsl
            )
        # 归一化 pred（如管道分隔表格 → mermaid）
        actual_pred = normalize_prediction_for_flowchart(prediction, task)
        # 如果归一化后变成了 mermaid 格式，更新 pred_dsl
        actual_dsl = pred_dsl
        if pred_dsl != "mermaid" and is_mermaid(actual_pred):
            actual_dsl = "mermaid"
        scores, eval_logs = flowchart_eval_multi(
            [actual_pred], [reference], easy=easy, pred_dsl=actual_dsl, ref_dsl="mermaid"
        )
        # 解析图 IR，保存为可读结构（便于调试和可解析性）
        pred_graph = parse_flowchart(actual_pred, dsl=actual_dsl)
        ref_graph = parse_flowchart(reference, dsl="mermaid")
        best_result = _build_score_dict(
            scores,
            eval_logs,
            eval_mode="flowchart",
            task=task,
            pred_text=actual_pred,
            ref_text=reference,
            pred_dsl=pred_dsl,
            pred_graph=_graph_ir_to_serializable(pred_graph),
            ref_graph=_graph_ir_to_serializable(ref_graph),
        )
        # Fallback：如果分数较低且 pred 是从管道表格转换的，尝试合并 cell 版本
        if best_result.get("map_high", 0) < 0.5 and actual_pred != prediction.strip():
            from .normalize import _pipe_table_to_mermaid

            merged_pred = _pipe_table_to_mermaid(prediction.strip(), merge_cells=True)
            if merged_pred and is_mermaid(merged_pred):
                scores2, eval_logs2 = flowchart_eval_multi(
                    [merged_pred], [reference], easy=easy, pred_dsl="mermaid", ref_dsl="mermaid"
                )
                merged_pred_graph = parse_flowchart(merged_pred, dsl="mermaid")
                alt_result = _build_score_dict(
                    scores2,
                    eval_logs2,
                    eval_mode="flowchart",
                    task=task,
                    pred_text=merged_pred,
                    ref_text=reference,
                    pred_dsl=pred_dsl,
                    pred_graph=_graph_ir_to_serializable(merged_pred_graph),
                    ref_graph=_graph_ir_to_serializable(ref_graph),
                )
                if alt_result.get("map_high", 0) > best_result.get("map_high", 0):
                    best_result = alt_result
        return best_result

    # ---- 分支 1: Markdown 无序列表（思维导图） ----
    if is_markdown_list(reference):
        if not prediction.strip() or not reference.strip():
            return _zero_score_dict(eval_mode="tree", pred_text=prediction, ref_text=reference)
        actual_pred = normalize_prediction_for_logic(prediction, task)
        scores, eval_logs = tree_eval([actual_pred], [reference], easy=easy)
        # 解析路径集合，保存为可读结构（便于调试和可解析性）
        pred_roots = normalize_tree(parse_markdown_list(actual_pred))
        ref_roots = normalize_tree(parse_markdown_list(reference))
        pred_paths = tree_to_paths(pred_roots)
        ref_paths = tree_to_paths(ref_roots)
        best_result = _build_score_dict(
            scores,
            eval_logs,
            eval_mode="tree",
            task=task,
            pred_text=actual_pred,
            ref_text=reference,
            pred_paths=_tree_paths_to_serializable(pred_paths),
            ref_paths=_tree_paths_to_serializable(ref_paths),
        )
        # 对于管道分隔格式的输出，尝试 pipe 方法作为备选，取分数更高的结果
        # 这避免了 pipe 方法在某些样本上产生噪声节点导致回退
        if "|" in prediction and best_result.get("map_high", 0) < 0.5:
            from .normalize import _pipe_table_to_markdown_list

            pipe_pred = _pipe_table_to_markdown_list(prediction)
            if pipe_pred and is_markdown_list(pipe_pred) and pipe_pred != actual_pred:
                scores2, eval_logs2 = tree_eval([pipe_pred], [reference], easy=easy)
                pipe_pred_roots = normalize_tree(parse_markdown_list(pipe_pred))
                pipe_pred_paths = tree_to_paths(pipe_pred_roots)
                alt_result = _build_score_dict(
                    scores2,
                    eval_logs2,
                    eval_mode="tree",
                    task=task,
                    pred_text=pipe_pred,
                    ref_text=reference,
                    pred_paths=_tree_paths_to_serializable(pipe_pred_paths),
                    ref_paths=_tree_paths_to_serializable(ref_paths),
                )
                if alt_result.get("map_high", 0) > best_result.get("map_high", 0):
                    best_result = alt_result
        return best_result

    # ---- 分支 2: Mermaid 流程图（兜底路径：当非流程图类 task 碰到 mermaid reference 时进入） ----
    if is_mermaid(reference):
        if not prediction.strip() or not reference.strip():
            return _zero_score_dict(eval_mode="flowchart", pred_text=prediction, ref_text=reference)
        # 对 SE_CODE 等非 mermaid-native task，先把 pred 归一化成 mermaid（例如 networkx 代码→mermaid），
        # 避免下游按 mermaid 解析直接 parse 失败 0 分。
        actual_pred = normalize_prediction_for_flowchart(prediction, task)
        scores, eval_logs = flowchart_eval_multi(
            [actual_pred], [reference], easy=easy, pred_dsl="mermaid", ref_dsl="mermaid"
        )
        pred_graph = parse_flowchart(actual_pred, dsl="mermaid")
        ref_graph = parse_flowchart(reference, dsl="mermaid")
        return _build_score_dict(
            scores,
            eval_logs,
            eval_mode="flowchart",
            task=task,
            pred_text=actual_pred,
            ref_text=reference,
            pred_graph=_graph_ir_to_serializable(pred_graph),
            ref_graph=_graph_ir_to_serializable(ref_graph),
        )

    # ---- 分支 3: 表格 / CSV（数值类） ----
    pred_csv = normalize_prediction_for_data(prediction, task)
    ref_csv = normalize_to_csv(reference)

    if not pred_csv or not ref_csv:
        return _zero_score_dict(eval_mode="csv", task=task, pred_csv=pred_csv, ref_csv=ref_csv)

    # SE_SVG 专属处理：SVG 模态无法自然承载列名，强制将 ref 表头剥离，
    # 让 pred / ref 都按无表头模式（列索引无关）对齐，仅比较 "行标签 + 数值" 的集合。
    if task == "SE_SVG":
        ref_csv = _strip_ref_header_for_svg(ref_csv)

    scores, eval_logs = csv_eval([pred_csv], [ref_csv], easy=easy)
    best_result = _build_score_dict(scores, eval_logs, eval_mode="csv", task=task, pred_csv=pred_csv, ref_csv=ref_csv)

    # 转置 Fallback：雷达图等图表类型中，模型可能输出行列转置的表格。
    # 当正常评分较低时，尝试转置 pred_csv 后重新评分，取更高分。
    if best_result.get("map_high", 0) < 0.5 and task != "SE_SVG":
        transposed_csv = _transpose_csv(pred_csv)
        if transposed_csv and transposed_csv != pred_csv:
            scores2, eval_logs2 = csv_eval([transposed_csv], [ref_csv], easy=easy)
            alt_result = _build_score_dict(
                scores2, eval_logs2, eval_mode="csv", task=task, pred_csv=transposed_csv, ref_csv=ref_csv
            )
            if alt_result.get("map_high", 0) > best_result.get("map_high", 0):
                best_result = alt_result

    return best_result


# ============================================================
# 入口函数：每个 task 一个，绑定到 JUDGE_FUNC
# ============================================================


def _extract_prediction(extract_value: Any, task_result: dict | None = None) -> str:
    """从 extract 或 answer 中获取预测文本"""
    if isinstance(extract_value, dict):
        return extract_value.get("extracted_table", "") or ""
    if isinstance(extract_value, str):
        return extract_value
    if task_result and isinstance(task_result, dict):
        return task_result.get("answer", "") or ""
    return ""


def _make_judge_entry(task: str) -> Callable:
    """根据 task 生成评分入口函数"""

    def _judge_func(extract_value: Any, row: dict) -> dict:
        prediction = _extract_prediction(extract_value)
        reference = row.get("anno", "")

        if not prediction:
            prediction = ""
        if not reference:
            return {"score": _zero_score_dict(task=task), "error": "no_reference"}

        # 把 chart_type 放进线程局部上下文，供 _infer_col_labels 等函数读取
        set_judge_context(chart_type=row.get("chart_type", ""))
        try:
            score = _judge_by_task(prediction, reference, task=task, easy=1)
        finally:
            set_judge_context(chart_type="")
        return {"score": score}

    _judge_func.__name__ = f"judge_{task.lower()}"
    _judge_func.__doc__ = f"{task} 评分入口函数"
    return _judge_func


# ============================================================
# 评分函数注册表
# ============================================================

JUDGE_FUNC: dict[str, Callable] = {
    "SE_MD": _make_judge_entry("SE_MD"),
    # 带 hint 的 Markdown 任务，评分逻辑与 SE_MD 完全一致。
    "SE_MD_w_HINT": _make_judge_entry("SE_MD"),
    "SE_JSON": _make_judge_entry("SE_JSON"),
    "SE_CSV": _make_judge_entry("SE_CSV"),
    "SE_CODE": _make_judge_entry("SE_CODE"),
    "SE_SVG": _make_judge_entry("SE_SVG"),
    "SE_MERMAID": _make_judge_entry("SE_MERMAID"),
    "SE_GRAPHVIZ": _make_judge_entry("SE_GRAPHVIZ"),
    "SE_PLANTUML": _make_judge_entry("SE_PLANTUML"),
    "SE_DIAGRAMS": _make_judge_entry("SE_DIAGRAMS"),
    "SE_D2": _make_judge_entry("SE_D2"),
    "SE_CYTOSCAPE": _make_judge_entry("SE_CYTOSCAPE"),
}


if __name__ == "__main__":
    # 简单自测
    pred_md = "| 类别   | 得分 |\n| ---- | -- |\n| 财富   | 8  |\n| 家庭   | 6  |"
    ref_md = "|  |  |\n| --- | --- |\n| 财富 | 8 |\n| 家庭 | 6 |"
    print("[SE_MD]", _judge_by_task(pred_md, ref_md, task="SE_MD"))

    pred_csv = "类别,得分\n财富,8\n家庭,6"
    print("[SE_CSV]", _judge_by_task(pred_csv, ref_md, task="SE_CSV"))

    pred_json = '{"title": "test", "values": {"得分": {"财富": 8, "家庭": 6}}}'
    print("[SE_JSON]", _judge_by_task(pred_json, ref_md, task="SE_JSON"))
