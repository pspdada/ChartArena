"""D2 (terrastruct/d2) 格式解析器。"""

from __future__ import annotations

import re

from ._types import Graph, _empty_graph, strip_code_fence

_D2_IDENT = r"[A-Za-z_\u4e00-\u9fff][\w\u4e00-\u9fff.\-]*"
_D2_KV_RE = re.compile(
    rf"^\s*(?P<id>{_D2_IDENT})\s*:\s*(?P<val>.+?)\s*$",
    flags=re.MULTILINE,
)
_D2_EDGE_RE = re.compile(
    rf"(?P<src>{_D2_IDENT})\s*(?P<arrow>->|--|<->)\s*(?P<dst>{_D2_IDENT})"
    rf"(?:\s*:\s*(?P<label>[^\n{{]+))?",
)


def is_d2(text: str) -> bool:
    """判断文本是否为 D2 源码。

    启发式检测：
        - 去围栏后内容含 -> 或 -- 边；
        - 且含 id: "label" / id: 'label' / id: label 形式的节点声明；
        - 且不被更前置的 mermaid/dot/plantuml 关键字所拦截。
    """
    if not text or not text.strip():
        return False
    t = strip_code_fence(text, ("d2", "D2"))
    low = t.lower()
    if "digraph" in low or "strict graph" in low or "@startuml" in low:
        return False
    if re.search(r"(?m)^\s*(?:flowchart|graph)\s+[A-Z]{2}\b", t):
        return False
    if not re.search(r"\w\s*->\s*\w|\w\s*--\s*\w", t):
        return False
    if re.search(r"(?m)^\s*[\w.\-]+\s*:\s*\S", t):
        return True
    return bool(re.search(r"\w\s*->\s*\w", t))


def _d2_clean_label(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1]
    s = re.sub(r"\\n|\\r", " ", s)
    s = re.sub(r"<\s*br\s*/?\s*>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    return s.strip().rstrip(";").strip()


def parse_d2(text: str) -> Graph:
    """将 D2 源码解析为图 IR。

    覆盖范围：
        1. id: "label" / id: 'label' / id: label（id 可含点，只取第一段 token）；
        2. a -> b、a -> b -> c 链式边；: label 作为边 label；
        3. container: { ... }：外壳剥掉，内部一起扫；
        4. 注释 # / //。
    """
    if not text or not text.strip():
        return _empty_graph()

    t = strip_code_fence(text, ("d2", "D2"))
    # 去注释
    t = re.sub(r"/\*.*?\*/", " ", t, flags=re.DOTALL)
    t = re.sub(r"(?m)(^|\s)//.*$", " ", t)
    t = re.sub(r"(?m)^\s*#.*$", " ", t)
    # 剥掉容器外壳
    t = re.sub(r"(?m)^[^\n{:]+\{\s*$", " ", t)
    t = t.replace("}", " ")

    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    labeled_edges: list[tuple[str, str, str]] = []

    def _ensure_node(nid: str, label: str | None = None) -> str:
        nid = nid.strip()
        if not nid:
            return ""
        first_seg = nid.split(".", 1)[0]
        if "." in nid and label is not None:
            sub_key = nid.split(".", 1)[1].strip().lower()
            if sub_key in {"label", "shape", "style", "icon"}:
                if sub_key == "label":
                    nodes[first_seg] = _d2_clean_label(label)
                if first_seg not in nodes:
                    nodes[first_seg] = first_seg
                return first_seg
        if first_seg not in nodes:
            nodes[first_seg] = _d2_clean_label(label) if label else first_seg
        elif label:
            lab = _d2_clean_label(label)
            if lab:
                nodes[first_seg] = lab
        return first_seg

    # 先扫非边语句：登记节点
    for raw_line in t.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if re.search(r"->|<->|--", line):
            continue
        m = _D2_KV_RE.match(line)
        if m:
            nid = m.group("id")
            val = m.group("val").strip().rstrip(";").strip()
            if val.startswith("{"):
                _ensure_node(nid, None)
                continue
            _ensure_node(nid, val)
        else:
            m2 = re.match(rf"^\s*({_D2_IDENT})\s*$", line)
            if m2:
                _ensure_node(m2.group(1), None)

    # 再扫边：支持 a -> b -> c : label
    for raw_line in t.split("\n"):
        line = raw_line.strip().rstrip(";").strip()
        if not line:
            continue
        if not re.search(r"->|<->|--", line):
            continue
        # 拆出尾部 label
        label = ""
        mlab = re.match(r"^(.+?)\s*:\s*(.+?)\s*$", line)
        if mlab and not re.search(r"->|<->|--", mlab.group(2)):
            line_core = mlab.group(1).strip()
            label = _d2_clean_label(mlab.group(2))
        else:
            line_core = line

        tokens = re.split(r"\s*(->|<->|--)\s*", line_core)
        if len(tokens) < 3:
            continue
        node_tokens = [tokens[i] for i in range(0, len(tokens), 2)]
        arrow_tokens = [tokens[i] for i in range(1, len(tokens), 2)]

        for tok in node_tokens:
            tok = tok.strip()
            if tok:
                _ensure_node(tok, None)

        for i, arrow in enumerate(arrow_tokens):
            src = node_tokens[i].strip().split(".", 1)[0]
            dst = node_tokens[i + 1].strip().split(".", 1)[0]
            if not src or not dst:
                continue
            if arrow == "<->":
                edges.append((src, dst))
                edges.append((dst, src))
                labeled_edges.append((src, dst, label))
                labeled_edges.append((dst, src, label))
            else:
                edges.append((src, dst))
                labeled_edges.append((src, dst, label))

    return nodes, edges, labeled_edges
