"""mingrammer Diagrams（Python DSL）解析器。

纯 AST 静态分析，不执行任何代码。
"""

from __future__ import annotations

import ast
import logging
import re

from ._types import Graph, _empty_graph, strip_code_fence

logger = logging.getLogger(__name__)


def is_diagrams(text: str) -> bool:
    """判断文本是否为 mingrammer diagrams 的 Python DSL。"""
    if not text or not text.strip():
        return False
    t = strip_code_fence(text, ("python", "py"))
    if re.search(r"\bfrom\s+diagrams\b|\bimport\s+diagrams\b", t):
        return True
    if "with Diagram(" in t and re.search(r">>|<<", t):
        return True
    return False


def _const_str(node: ast.AST | None) -> str | None:
    """尝试把 AST 节点当成字符串字面量读取。"""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):  # f-string：只保留常量段
        parts: list[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
        return "".join(parts) if parts else None
    return None


def parse_diagrams(text: str) -> Graph:
    """将 Diagrams（mingrammer）Python DSL 解析为图 IR。

    支持：
        - a >> b → 边 a → b
        - a << b → 边 b → a
        - a - b  → 无向，按 a → b
        - a >> Edge(label="...") >> b → 带 label 的边
        - 链式：a >> b >> c、[a, b] >> c 等均展开
        - Cluster 本身不作为节点入图（只是分组容器）

    解析失败 → 空图。
    """
    if not text or not text.strip():
        return _empty_graph()

    code = strip_code_fence(text, ("python", "py"))

    try:
        tree = ast.parse(code)
    except Exception as e:
        logger.debug(f"[parse_diagrams] ast.parse failed: {e}")
        return _empty_graph()

    var_to_id: dict[str, str] = {}
    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    labeled_edges: list[tuple[str, str, str]] = []

    def _register(label: str, var_name: str | None) -> str:
        label = (label or "").strip() or (var_name or "").strip()
        if not label:
            return ""
        nid = label
        if nid in nodes and var_name and nodes[nid] != label:
            nid = f"{label}__{var_name}"
        if nid not in nodes:
            nodes[nid] = label
        if var_name:
            var_to_id[var_name] = nid
        return nid

    def _resolve_to_ids(node: ast.AST) -> list[str]:
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            out: list[str] = []
            for e in node.elts:
                out.extend(_resolve_to_ids(e))
            return out
        if isinstance(node, ast.Name):
            nid = var_to_id.get(node.id)
            return [nid] if nid else []
        if isinstance(node, ast.Call):
            label = None
            if node.args:
                label = _const_str(node.args[0])
            if label:
                return [_register(label, None)]
            return []
        if isinstance(node, ast.BinOp):
            left_ids = _resolve_to_ids(node.left)
            right_ids = _resolve_to_ids(node.right)
            return left_ids + right_ids if not isinstance(node.op, (ast.RShift, ast.LShift)) else right_ids
        return []

    # Pass 1：扫描所有赋值，建立 var → node_id 映射
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            var_name = n.targets[0].id
            val = n.value
            if isinstance(val, ast.Call):
                label = _const_str(val.args[0]) if val.args else None
                if label is None:
                    for kw in val.keywords:
                        if kw.arg in ("label", "name") and isinstance(kw.value, ast.Constant):
                            if isinstance(kw.value.value, str):
                                label = kw.value.value
                                break
                if label is None:
                    label = var_name
                func_name = ""
                if isinstance(val.func, ast.Name):
                    func_name = val.func.id
                elif isinstance(val.func, ast.Attribute):
                    func_name = val.func.attr
                if func_name in {"Diagram", "Cluster", "Edge", "Node"}:
                    continue
                _register(label, var_name)

    # Pass 2：扫描所有 BinOp（>> / << / -）建立边
    def _collect_binop_chain(node: ast.BinOp) -> list[tuple[str, ast.AST, ast.AST]]:
        segments: list[tuple[str, ast.AST, ast.AST]] = []

        def _op_sym(op: ast.operator) -> str | None:
            if isinstance(op, ast.RShift):
                return ">>"
            if isinstance(op, ast.LShift):
                return "<<"
            if isinstance(op, ast.Sub):
                return "-"
            return None

        def _recurse(n: ast.AST) -> ast.AST:
            if isinstance(n, ast.BinOp):
                sym = _op_sym(n.op)
                if sym is not None:
                    inner_left = _recurse(n.left)
                    segments.append((sym, inner_left, n.right))
                    return n.right
            return n

        _recurse(node)
        return segments

    def _edge_label_from_node(n: ast.AST) -> tuple[str, ast.AST | None]:
        if isinstance(n, ast.Call):
            func_name = ""
            if isinstance(n.func, ast.Name):
                func_name = n.func.id
            elif isinstance(n.func, ast.Attribute):
                func_name = n.func.attr
            if func_name == "Edge":
                label = ""
                for kw in n.keywords:
                    if kw.arg == "label":
                        lab = _const_str(kw.value)
                        if lab is not None:
                            label = lab
                            break
                return label, None
        return "", n

    # 只从"最外层链"处理，避免嵌套重复
    parent_map: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[id(child)] = parent

    def _is_chain_parent(n: ast.AST) -> bool:
        p = parent_map.get(id(n))
        if not isinstance(p, ast.BinOp):
            return False
        return isinstance(p.op, (ast.RShift, ast.LShift, ast.Sub))

    for n in ast.walk(tree):
        if not isinstance(n, ast.BinOp):
            continue
        if not isinstance(n.op, (ast.RShift, ast.LShift, ast.Sub)):
            continue
        if _is_chain_parent(n):
            continue

        segments = _collect_binop_chain(n)
        pending_label = ""
        last_real_left: ast.AST | None = None

        for sym, left_node, right_node in segments:
            right_label, right_real = _edge_label_from_node(right_node)
            if right_real is None:
                pending_label = right_label
                _, left_real_tmp = _edge_label_from_node(left_node)
                last_real_left = left_real_tmp if left_real_tmp is not None else left_node
                continue

            left_label_attempt, left_real = _edge_label_from_node(left_node)
            if left_real is None:
                left_real = last_real_left if last_real_left is not None else left_node
            last_real_left = right_real

            left_ids = _resolve_to_ids(left_real)
            right_ids = _resolve_to_ids(right_real)
            if not left_ids or not right_ids:
                pending_label = ""
                continue

            edge_lab = pending_label
            pending_label = ""

            if sym == ">>":
                for s in left_ids:
                    for d in right_ids:
                        edges.append((s, d))
                        labeled_edges.append((s, d, edge_lab))
            elif sym == "<<":
                for s in right_ids:
                    for d in left_ids:
                        edges.append((s, d))
                        labeled_edges.append((s, d, edge_lab))
            elif sym == "-":
                for s in left_ids:
                    for d in right_ids:
                        edges.append((s, d))
                        labeled_edges.append((s, d, edge_lab))

    return nodes, edges, labeled_edges
