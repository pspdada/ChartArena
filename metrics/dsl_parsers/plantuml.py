"""PlantUML 格式解析器（主要覆盖 activity/flowchart 两种语法）。"""

from __future__ import annotations

import re

from ._types import Graph, _empty_graph, strip_code_fence

_PUML_ACTIVITY_NODE_RE = re.compile(r":([^:\n;]+);")
_PUML_IF_RE = re.compile(
    r"""if\s*\(\s*(?P<cond>[^)]+?)\s*\)\s*then\s*\(\s*(?P<then_lab>[^)]*)\s*\)""",
    flags=re.IGNORECASE,
)
_PUML_ELSE_RE = re.compile(r"""else\s*(?:\(\s*(?P<else_lab>[^)]*)\s*\))?""", flags=re.IGNORECASE)
_PUML_ENDIF_RE = re.compile(r"endif", flags=re.IGNORECASE)

_PUML_ARROW_RE = re.compile(
    r"""
    (?P<src>\[[^\]]+\]|"[^"]+"|\w+|\(\*\))
    \s*(?:-+>|\.+>|=+>)\s*
    (?P<dst>\[[^\]]+\]|"[^"]+"|\w+|\(\*\))
    (?:\s*:\s*(?P<label>[^\n]+))?
    """,
    flags=re.VERBOSE,
)


def is_plantuml(text: str) -> bool:
    """判断文本是否为 PlantUML 源码。"""
    if not text or not text.strip():
        return False
    t = strip_code_fence(text, ("plantuml", "puml", "uml"))
    if "@startuml" in t or "@enduml" in t:
        return True
    # 弱判据：activity 风格的 `:node;` + `->` 箭头
    if re.search(r":[^:\n]{1,200};", t) and re.search(r"-+>|-+\[#", t):
        return True
    return False


def _puml_clean_label(s: str) -> str:
    """去掉 PlantUML 标签里常见的颜色修饰 / 尾部分号 / HTML 标签，归一化空白。"""
    s = s.strip().rstrip(";").strip()
    s = re.sub(r"\\[nrl]", " ", s)
    s = re.sub(r"[\n\r\t]+", " ", s)
    s = re.sub(r"<\s*br\s*/?\s*>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"^\*+\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def parse_plantuml(text: str) -> Graph:
    """将 PlantUML 源码解析为图 IR。

    覆盖范围（按优先级）：
        1. activity 图（新语法）：:node;、if/then/else/endif、
           fork / fork again / end fork、split / split again / end split、
           partition "..." { ... }（仅剥外壳）、back to :X;；
        2. 传统箭头边：A -> B : label、(*) -> A、[A] --> [B]；
        3. 其他（state / component / class）降级为只抽箭头边。
    """
    if not text or not text.strip():
        return _empty_graph()

    t = strip_code_fence(text, ("plantuml", "puml", "uml"))

    # 去掉 @startuml/@enduml 包裹 + 注释行
    t = re.sub(r"@start\w+[^\n]*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"@end\w+[^\n]*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"/'.*?'/", " ", t, flags=re.DOTALL)
    t = re.sub(r"(?m)^\s*'.*$", "", t)

    # 删除 note 块
    t = re.sub(r"(?is)\bnote\b(?:\s+\w+(?:\s+of\s+\S+)?)?\s*\n.*?\bend\s*note\b", " ", t)
    t = re.sub(r"(?mi)^\s*note\b.*$", "", t)

    # 剥离 partition 外壳
    t = re.sub(r"(?mi)^\s*partition\b.*$", "", t)
    t = re.sub(r"(?m)^\s*[{}]\s*$", "", t)

    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    labeled_edges: list[tuple[str, str, str]] = []

    title_nid: str = ""

    def _add_node(label: str) -> str:
        label = _puml_clean_label(label)
        if not label:
            return ""
        nid = label
        if nid not in nodes:
            nodes[nid] = label
        return nid

    def _add_edge(src: str | None, dst: str, lab: str = ""):
        if src and dst and src != dst:
            edges.append((src, dst))
            labeled_edges.append((src, dst, lab))

    prev_nodes: list[str] = []
    if_stack: list[dict] = []
    fork_stack: list[dict] = []
    pending_edge_label: str = ""

    # 把多行 `:foo\n  bar;` 里的换行规整为空格
    t_norm = re.sub(
        r":\s*([^:;]*?)\s*;",
        lambda m: ":" + m.group(1).replace("\n", " ") + ";",
        t,
        flags=re.DOTALL,
    )

    def _connect_from_prev(nid: str):
        nonlocal prev_nodes, pending_edge_label
        lab = pending_edge_label
        pending_edge_label = ""
        for src in prev_nodes:
            _add_edge(src, nid, lab)
        prev_nodes = [nid]

    for raw_line in t_norm.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        # title XXX → 作为根节点
        m_title = re.match(r"^title\s+(.+?)\s*$", line, flags=re.IGNORECASE)
        if m_title:
            title_label = _puml_clean_label(m_title.group(1))
            if title_label:
                title_nid = _add_node(title_label)
                if title_nid:
                    prev_nodes = [title_nid]
            continue

        # start / stop / end
        if re.match(r"^start\s*$", line, flags=re.IGNORECASE):
            prev_nodes = [title_nid] if title_nid else []
            pending_edge_label = ""
            continue
        if re.match(r"^(stop|end)\s*$", line, flags=re.IGNORECASE):
            prev_nodes = []
            pending_edge_label = ""
            continue

        # if / else / endif
        m_if = _PUML_IF_RE.search(line)
        if m_if:
            cond_label = _puml_clean_label(m_if.group("cond"))
            then_label = _puml_clean_label(m_if.group("then_lab") or "")
            cond_nid = _add_node(cond_label)
            for src in prev_nodes:
                _add_edge(src, cond_nid, pending_edge_label)
            pending_edge_label = ""
            prev_nodes = [cond_nid]
            ctx = {"cond": cond_nid, "branch_tails": [], "in_else": False}
            if_stack.append(ctx)
            pending_edge_label = then_label
            continue

        m_else = _PUML_ELSE_RE.match(line)
        if m_else and if_stack:
            else_label = _puml_clean_label(m_else.group("else_lab") or "")
            ctx = if_stack[-1]
            ctx["branch_tails"].extend(prev_nodes)
            prev_nodes = [ctx["cond"]]
            pending_edge_label = else_label
            ctx["in_else"] = True
            continue

        if _PUML_ENDIF_RE.match(line):
            if if_stack:
                ctx = if_stack.pop()
                merged = list(ctx["branch_tails"])
                merged.extend(prev_nodes)
                seen: set[str] = set()
                merged_unique = []
                for n in merged:
                    if n and n not in seen:
                        seen.add(n)
                        merged_unique.append(n)
                prev_nodes = merged_unique
                pending_edge_label = ""
            continue

        # fork / fork again / end fork（及 split 同义）
        if re.match(r"^(fork|split)\s*$", line, flags=re.IGNORECASE):
            ctx = {
                "heads": list(prev_nodes),
                "branch_tails": [],
                "kind": "fork" if line.lower().startswith("fork") else "split",
            }
            fork_stack.append(ctx)
            pending_edge_label = ""
            continue

        if re.match(r"^(fork\s+again|split\s+again)\s*$", line, flags=re.IGNORECASE):
            if fork_stack:
                ctx = fork_stack[-1]
                ctx["branch_tails"].extend(prev_nodes)
                prev_nodes = list(ctx["heads"])
                pending_edge_label = ""
            continue

        if re.match(r"^end\s*(fork|split|merge)\s*$", line, flags=re.IGNORECASE):
            if fork_stack:
                ctx = fork_stack.pop()
                ctx["branch_tails"].extend(prev_nodes)
                seen = set()
                merged_unique = []
                for n in ctx["branch_tails"]:
                    if n and n not in seen:
                        seen.add(n)
                        merged_unique.append(n)
                prev_nodes = merged_unique
                pending_edge_label = ""
            continue

        # back to :X; 或 back to X
        m_back = re.match(r"^back\s+to\s+:?\s*([^;]+?)\s*;?\s*$", line, flags=re.IGNORECASE)
        if m_back:
            target_label = _puml_clean_label(m_back.group(1))
            if target_label:
                tgt = _add_node(target_label)
                for src in prev_nodes:
                    _add_edge(src, tgt, pending_edge_label)
                pending_edge_label = ""
                prev_nodes = []
            continue

        # 独立的 `-> label;` 行（作为下一条入边 label）
        m_lonely = re.match(r"^-+>\s*([^;\n]+?)\s*;?\s*$", line)
        if m_lonely:
            pending_edge_label = _puml_clean_label(m_lonely.group(1))
            continue

        # 独立的 `|label|` 行（作为下一条入边 label）
        m_pipe = re.match(r"^\|([^|\n]+)\|\s*$", line)
        if m_pipe:
            pending_edge_label = _puml_clean_label(m_pipe.group(1))
            continue

        # 同一行混合语法 `:X; --> :Y;`
        if re.search(r":[^:;\n]+;\s*-+>\s*:[^:;\n]+;", line):
            parts = re.split(r"\s*-+>\s*", line)
            chain_nodes: list[str] = []
            for part in parts:
                m = re.match(r"^\s*:([^:;\n]*?)\s*;", part)
                if m:
                    nid = _add_node(m.group(1))
                    if nid:
                        chain_nodes.append(nid)
            if len(chain_nodes) >= 2:
                for a, b in zip(chain_nodes[:-1], chain_nodes[1:]):
                    _add_edge(a, b, pending_edge_label)
                    pending_edge_label = ""
                prev_nodes = [chain_nodes[-1]]
                continue
            elif len(chain_nodes) == 1:
                _connect_from_prev(chain_nodes[0])
                continue

        # 普通 activity 节点 `:text;`
        m_nodes = _PUML_ACTIVITY_NODE_RE.findall(line)
        if m_nodes:
            for nd_text in m_nodes:
                nid = _add_node(nd_text)
                if not nid:
                    continue
                _connect_from_prev(nid)
            continue

        # 传统箭头边 `A -> B : label`
        any_arrow = False
        for m_arrow in _PUML_ARROW_RE.finditer(line):
            any_arrow = True
            src_token = m_arrow.group("src").strip()
            dst_token = m_arrow.group("dst").strip()
            edge_label = _puml_clean_label(m_arrow.group("label") or "")
            src_label = src_token.strip('[]"')
            dst_label = dst_token.strip('[]"')
            if src_token == "(*)":
                src_label = "start"
            if dst_token == "(*)":
                dst_label = "end"
            src_nid = _add_node(src_label)
            dst_nid = _add_node(dst_label)
            if src_nid and dst_nid:
                edges.append((src_nid, dst_nid))
                labeled_edges.append((src_nid, dst_nid, edge_label))
        if any_arrow:
            continue

    return nodes, edges, labeled_edges
