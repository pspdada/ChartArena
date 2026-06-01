"""流程图多 DSL 解析器包。

各 DSL 解析器：
    dot.py        — Graphviz DOT
    plantuml.py   — PlantUML
    diagrams.py   — mingrammer Diagrams（Python DSL）
    d2.py         — D2 (terrastruct)
    cytoscape.py  — Cytoscape.js JSON

公共工具、路由函数和评估入口定义在本文件中。
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..mermaid_eval import (
    _edge_similarity,
    _hungarian_matching_score,
    _node_similarity,
    flowchart_similarity,
    is_mermaid,
    parse_mermaid,
)
from ._types import DSLType, Graph, _empty_graph, strip_code_fence
from .cytoscape import is_cytoscape, parse_cytoscape
from .d2 import is_d2, parse_d2
from .diagrams import is_diagrams, parse_diagrams
from .dot import is_dot, parse_dot
from .plantuml import is_plantuml, parse_plantuml

# ============================================================
# 路由：自动嗅探 / 显式 hint
# ============================================================


def parse_flowchart(text: str, dsl: DSLType | None = None) -> Graph:
    """统一入口：按 DSL 类型解析为图 IR。

    Args:
        text: 源码文本
        dsl:  显式指定的 DSL 类型；None 时按 is_mermaid / is_dot / is_plantuml /
              is_diagrams / is_cytoscape / is_d2 顺序嗅探（mermaid 优先）。

    Returns:
        Graph IR（解析失败时返回空图）。
    """
    if not text or not text.strip():
        return _empty_graph()

    dispatch = {
        "mermaid": parse_mermaid,
        "dot": parse_dot,
        "plantuml": parse_plantuml,
        "diagrams": parse_diagrams,
        "d2": parse_d2,
        "cytoscape": parse_cytoscape,
    }
    if dsl and dsl in dispatch:
        return dispatch[dsl](text)

    # 自动嗅探
    if is_mermaid(text):
        return parse_mermaid(text)
    if is_dot(text):
        return parse_dot(text)
    if is_plantuml(text):
        return parse_plantuml(text)
    if is_diagrams(text):
        return parse_diagrams(text)
    if is_cytoscape(text):
        return parse_cytoscape(text)
    if is_d2(text):
        return parse_d2(text)
    return _empty_graph()


# ============================================================
# 评估入口
# ============================================================


def _graph_to_label_edges(
    graph: Graph,
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """把 IR 里的 node_id 替换为 label，便于与 flowchart_similarity 的内部逻辑一致。"""
    nodes, _, labeled_edges = graph
    node_labels = list(nodes.values())
    edge_labels = [(nodes.get(s, s), nodes.get(d, d), lab or "") for s, d, lab in labeled_edges]
    return node_labels, edge_labels


def flowchart_similarity_graph(
    pred_graph: Graph,
    ref_graph: Graph,
    edge_weight: float = 0.6,
    node_weight: float = 0.4,
    sim_threshold: float = 0.0,
) -> float:
    """基于图 IR 的相似度计算（算法与 mermaid_eval.flowchart_similarity 完全一致）。

    空图（解析失败）总是得 0。
    """
    if not pred_graph[0] or not ref_graph[0]:
        return 0.0

    pred_node_labels, pred_edge_labels = _graph_to_label_edges(pred_graph)
    ref_node_labels, ref_edge_labels = _graph_to_label_edges(ref_graph)

    edge_score = _hungarian_matching_score(pred_edge_labels, ref_edge_labels, _edge_similarity, sim_threshold)
    node_score = _hungarian_matching_score(pred_node_labels, ref_node_labels, _node_similarity, sim_threshold)

    if not pred_edge_labels and not ref_edge_labels:
        return node_score
    return edge_weight * edge_score + node_weight * node_score


def flowchart_eval_multi(
    predictions: list[str],
    references: list[str],
    easy: Literal[0, 1],
    pred_dsl: DSLType = "mermaid",
    ref_dsl: DSLType = "mermaid",
) -> tuple[tuple, list[str]]:
    """通用流程图评估：pred / ref 各自按指定 DSL 解析为图 IR 后算分。

    返回结构与 mermaid_eval.flowchart_eval 完全一致：(scores_tuple_13, eval_logs)。
    eval_logs 会记录 parse_failed=True 标志（当 pred_graph 为空图时）。
    """
    import logging

    logger = logging.getLogger(__name__)

    eval_logs: list[str] = []

    tolerance_params: dict[str, float] = {
        "strict": 1.0 if easy == 1 else 0.95,
        "slight": 0.85 if easy == 1 else 0.75,
        "high": 0.6 if easy == 1 else 0.5,
    }

    # 预解析，避免重复解析
    pred_graphs: list[Graph] = []
    ref_graphs: list[Graph] = []
    for idx, (p_text, r_text) in enumerate(zip(predictions, references)):
        pg = parse_flowchart(p_text, dsl=pred_dsl)
        rg = parse_flowchart(r_text, dsl=ref_dsl)
        pred_graphs.append(pg)
        ref_graphs.append(rg)
        if not pg[0]:
            eval_logs.append(
                f"[flowchart_eval_multi] idx={idx} pred_dsl={pred_dsl} parse_failed=True "
                f"pred_text_len={len(p_text or '')}"
            )

    def _compute_sim_list(sim_threshold: float) -> list[float]:
        return [
            flowchart_similarity_graph(pg, rg, sim_threshold=sim_threshold) for pg, rg in zip(pred_graphs, ref_graphs)
        ]

    sim_lists = {tol: _compute_sim_list(th) for tol, th in tolerance_params.items()}

    def _get_ap(sim_list: list[float], sim_threshold: float) -> float:
        if not sim_list:
            return 0.0
        return len([s for s in sim_list if s >= sim_threshold]) / len(sim_list)

    map_strict = map_slight = map_high = 0.0
    for sim_threshold in np.arange(0.5, 1, 0.05):
        map_strict += _get_ap(sim_lists["strict"], sim_threshold) / 10
        map_slight += _get_ap(sim_lists["slight"], sim_threshold) / 10
        map_high += _get_ap(sim_lists["high"], sim_threshold) / 10

    em = _get_ap(sim_lists["strict"], 1.0)
    ap_50_strict = _get_ap(sim_lists["strict"], 0.5)
    ap_75_strict = _get_ap(sim_lists["strict"], 0.75)
    ap_90_strict = _get_ap(sim_lists["strict"], 0.90)
    ap_50_slight = _get_ap(sim_lists["slight"], 0.5)
    ap_75_slight = _get_ap(sim_lists["slight"], 0.75)
    ap_90_slight = _get_ap(sim_lists["slight"], 0.90)
    ap_50_high = _get_ap(sim_lists["high"], 0.5)
    ap_75_high = _get_ap(sim_lists["high"], 0.75)
    ap_90_high = _get_ap(sim_lists["high"], 0.90)

    scores = (
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
    )
    return scores, eval_logs


# ============================================================
# 公开接口
# ============================================================

__all__ = [
    "Graph",
    "DSLType",
    "strip_code_fence",
    # is_* 检测器
    "is_dot",
    "is_plantuml",
    "is_diagrams",
    "is_d2",
    "is_cytoscape",
    # parse_* 解析器
    "parse_dot",
    "parse_plantuml",
    "parse_diagrams",
    "parse_d2",
    "parse_cytoscape",
    # 路由 + 评估
    "parse_flowchart",
    "flowchart_similarity_graph",
    "flowchart_eval_multi",
    # re-export from mermaid_eval
    "is_mermaid",
    "parse_mermaid",
    "flowchart_similarity",
]
