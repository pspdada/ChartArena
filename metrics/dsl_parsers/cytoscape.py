"""Cytoscape.js JSON 格式解析器。"""

from __future__ import annotations

import re

from ._types import Graph, _empty_graph, strip_code_fence


def is_cytoscape(text: str) -> bool:
    """判断文本是否为 Cytoscape.js JSON。

    启发式：
        - 能被 json.loads 解析；
        - 结构中存在 elements（dict 或 list）且其中至少一条记录形如 {"data": {...}}，
          data 里含 id 或 source 字段之一。
    """
    if not text or not text.strip():
        return False
    t = strip_code_fence(text, ("json", "cytoscape", "cyjs"))
    try:
        import json as _json

        obj = _json.loads(t)
    except Exception:
        return False
    return _cytoscape_has_elements(obj)


def _cytoscape_has_elements(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    ele = obj.get("elements")
    if ele is None:
        return False
    # 规范结构：{"nodes": [...], "edges": [...]}
    if isinstance(ele, dict):
        for key in ("nodes", "edges"):
            lst = ele.get(key)
            if isinstance(lst, list) and lst:
                for item in lst:
                    if isinstance(item, dict) and isinstance(item.get("data"), dict):
                        d = item["data"]
                        if "id" in d or "source" in d:
                            return True
        return False
    # flat 结构：[{"data": {...}}, ...]
    if isinstance(ele, list):
        for item in ele:
            if isinstance(item, dict) and isinstance(item.get("data"), dict):
                d = item["data"]
                if "id" in d or "source" in d:
                    return True
    return False


def _cy_clean_label(s) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\\n|\\r", " ", s)
    s = re.sub(r"<\s*br\s*/?\s*>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def parse_cytoscape(text: str) -> Graph:
    """将 Cytoscape.js JSON 解析为图 IR。

    覆盖范围：
        1. {"elements": {"nodes": [...], "edges": [...]}}（最常见）；
        2. {"elements": [ ... ]} flat 模式（按 data.source 是否存在区分 node/edge）；
        3. node.data 可用 label / name 字段作为显示文本；均缺失时回退 id；
        4. edge.data 可用 label / name 字段作为边 label。
    """
    if not text or not text.strip():
        return _empty_graph()

    t = strip_code_fence(text, ("json", "cytoscape", "cyjs"))
    try:
        import json as _json

        obj = _json.loads(t)
    except Exception:
        return _empty_graph()
    if not isinstance(obj, dict):
        return _empty_graph()
    ele = obj.get("elements")
    if ele is None:
        return _empty_graph()

    nodes_list: list[dict] = []
    edges_list: list[dict] = []

    def _collect_from_list(lst: list, default_kind: str | None = None):
        for item in lst:
            if not isinstance(item, dict):
                continue
            data = item.get("data")
            if not isinstance(data, dict):
                continue
            group = (item.get("group") or "").lower()
            is_edge = group == "edges" or "source" in data
            if default_kind == "nodes":
                is_edge = False
            elif default_kind == "edges":
                is_edge = True
            (edges_list if is_edge else nodes_list).append(data)

    if isinstance(ele, dict):
        _collect_from_list(ele.get("nodes") or [], default_kind="nodes")
        _collect_from_list(ele.get("edges") or [], default_kind="edges")
    elif isinstance(ele, list):
        _collect_from_list(ele)
    else:
        return _empty_graph()

    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    labeled_edges: list[tuple[str, str, str]] = []

    def _ensure_node(nid: str, label: str | None = None) -> str:
        if not nid:
            return ""
        nid = str(nid)
        lab = _cy_clean_label(label) if label else ""
        if nid not in nodes:
            nodes[nid] = lab if lab else nid
        elif lab:
            nodes[nid] = lab
        return nid

    for d in nodes_list:
        nid = str(d.get("id", "")).strip()
        if not nid:
            continue
        label = d.get("label") or d.get("name") or d.get("title") or nid
        _ensure_node(nid, label)

    for d in edges_list:
        src = str(d.get("source", "")).strip()
        dst = str(d.get("target", "")).strip()
        if not src or not dst:
            continue
        _ensure_node(src, None)
        _ensure_node(dst, None)
        lab = _cy_clean_label(d.get("label") or d.get("name") or "")
        edges.append((src, dst))
        labeled_edges.append((src, dst, lab))

    return nodes, edges, labeled_edges
