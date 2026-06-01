"""Graphviz DOT 格式解析器。"""

from __future__ import annotations

import re

from ._types import Graph, _empty_graph, strip_code_fence

_DOT_NODE_NAME = r'(?:"(?:[^"\\]|\\.)*"|[A-Za-z_][\w.:]*|\d+)'
_DOT_ATTR_RE = re.compile(r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|[^,\]\s]+)')


def is_dot(text: str) -> bool:
    """判断文本是否为 Graphviz DOT 格式。

    检测规则：
        - 以 digraph / graph / strict digraph / strict graph 开头；
        - 内部包含 -> 或 -- 边语法；
        - 需要有 { }。
    """
    if not text or not text.strip():
        return False
    t = strip_code_fence(text, ("dot", "graphviz"))
    if not re.search(r"\b(strict\s+)?(di)?graph\b", t, flags=re.IGNORECASE):
        return False
    if not re.search(r"->|--", t):
        return False
    if "{" not in t or "}" not in t:
        return False
    return True


def _dot_unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        inner = s[1:-1]
        inner = inner.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\")
        return inner
    return s


def _dot_parse_attrs(attr_str: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in _DOT_ATTR_RE.finditer(attr_str):
        k = m.group(1).lower()
        v = _dot_unquote(m.group(2))
        out[k] = v
    return out


def parse_dot(text: str) -> Graph:
    """将 Graphviz DOT 源码解析为图 IR。

    支持：
        - digraph / graph / strict digraph / strict graph 声明；
        - "N" / N / 带属性 N [label="..."] 的节点；
        - A -> B、A -> B -> C 链式边；A -> B [label="yes"]；
        - subgraph cluster_xxx { ... } 里的节点与边；
        - 忽略注释 // / # / /* */。

    label 选择优先级：节点的 label 属性 > unquote 后的节点名 > 原始 token。
    """
    if not text or not text.strip():
        return _empty_graph()

    t = strip_code_fence(text, ("dot", "graphviz"))

    # 去注释
    t = re.sub(r"/\*.*?\*/", " ", t, flags=re.DOTALL)
    t = re.sub(r"(?m)//.*$", " ", t)
    t = re.sub(r"(?m)^\s*#.*$", " ", t)

    # 剥掉 digraph X { ... } 外壳
    body = t
    m_head = re.search(r"\b(?:strict\s+)?(?:di)?graph\b[^{]*\{", body, flags=re.IGNORECASE)
    if m_head:
        body = body[m_head.end() :]
        last = body.rfind("}")
        if last != -1:
            body = body[:last]

    # 剥掉 subgraph 外壳（保留内部语句）
    body = re.sub(r"\bsubgraph\b[^{]*\{", " ", body, flags=re.IGNORECASE)
    body = body.replace("}", " ")

    statements = [s.strip() for s in re.split(r"[;\n]", body) if s.strip()]

    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    labeled_edges: list[tuple[str, str, str]] = []

    edge_re = re.compile(
        rf"({_DOT_NODE_NAME})\s*((?:->|--)\s*{_DOT_NODE_NAME}(?:\s*(?:->|--)\s*{_DOT_NODE_NAME})*)\s*(\[[^\]]*\])?",
        flags=re.DOTALL,
    )
    node_only_re = re.compile(rf"^({_DOT_NODE_NAME})\s*(\[[^\]]*\])?\s*$", flags=re.DOTALL)

    def _ensure_node(raw_token: str, label_override: str | None = None) -> str:
        raw_token = raw_token.strip()
        nid = _dot_unquote(raw_token)
        if nid.lower() in {"graph", "node", "edge", "subgraph"}:
            return ""
        label = label_override if label_override is not None else nid
        if nid not in nodes:
            nodes[nid] = label
        elif label_override is not None:
            nodes[nid] = label
        return nid

    for stmt in statements:
        # 跳过全局 attr（graph [...] / node [...] / edge [...]）
        if re.match(r"^(graph|node|edge)\s*\[", stmt, flags=re.IGNORECASE):
            continue

        # 先试边
        m_edge = edge_re.match(stmt)
        if m_edge and ("->" in stmt or "--" in stmt):
            first_token = m_edge.group(1)
            chain_tail = m_edge.group(2)
            attr_block = m_edge.group(3) or ""
            attrs = _dot_parse_attrs(attr_block[1:-1]) if attr_block else {}
            edge_label = attrs.get("label", "") or attrs.get("xlabel", "")

            chain_tokens = re.findall(rf"(?:->|--)\s*({_DOT_NODE_NAME})", chain_tail)
            token_seq = [first_token] + chain_tokens

            node_ids: list[str] = []
            for tok in token_seq:
                nid = _ensure_node(tok)
                if nid:
                    node_ids.append(nid)
            for i in range(len(node_ids) - 1):
                src, dst = node_ids[i], node_ids[i + 1]
                edges.append((src, dst))
                # 链式边的 edge label 只粘到最后一段
                lab = edge_label if i == len(node_ids) - 2 else ""
                labeled_edges.append((src, dst, lab))
            continue

        # 纯节点定义 N [label="..."]
        m_node = node_only_re.match(stmt)
        if m_node:
            tok = m_node.group(1)
            attr_block = m_node.group(2) or ""
            attrs = _dot_parse_attrs(attr_block[1:-1]) if attr_block else {}
            label = attrs.get("label")
            _ensure_node(tok, label_override=label)
            continue

    return nodes, edges, labeled_edges
