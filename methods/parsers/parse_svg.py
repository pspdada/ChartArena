"""SE_SVG：SVG 源码 → 内部 CSV / Markdown 列表。"""

import re

from ..context import get_chart_type
from .parse_python_code import _build_hierarchy_from_positions, _clean_label, _is_plausible_node_text


def _extract_svg_snippet(text: str) -> str:
    """从模型输出中剥离代码围栏，返回纯 ``<svg>...</svg>`` 片段（或原文）。

    优先级：``<svg>..</svg>`` > ``` 代码块 > 原文 trimmed。
    """
    if not text:
        return ""
    t = text.strip()

    m = re.search(r"<svg\b[^>]*>.*?</svg>", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(0)

    m = re.search(r"```[ \t]*(?:xml|svg|html|XML|SVG|HTML)[ \t]*\n?(.*?)```", t, flags=re.DOTALL)
    if m:
        inner = m.group(1).strip()
        if inner:
            return inner

    m = re.search(r"```[a-zA-Z0-9_\-]*[ \t]*\n?(.*?)```", t, flags=re.DOTALL)
    if m:
        inner = m.group(1).strip()
        if inner and "<svg" in inner.lower():
            return inner

    return t


_SVG_NS_RE = re.compile(r"\sxmlns(?::[\w\-]+)?\s*=\s*\"[^\"]*\"", re.IGNORECASE)


# ---------------------------------------------------------------------------
# transform 展开：把 <g transform="translate/scale/matrix ..."> 全部吞掉，
# 将内部所有几何/文本坐标改写为全局（viewBox）坐标。
# 仅处理 translate / scale / matrix；rotate/skew 对数据类图表影响小，暂忽略。
# ---------------------------------------------------------------------------

_TRANSFORM_FUNC_RE = re.compile(
    r"(translate|scale|matrix|rotate|skewX|skewY)\s*\(([^)]*)\)",
    re.IGNORECASE,
)


def _parse_transform(transform_str: str) -> tuple[float, float, float, float, float, float]:
    """解析 transform 属性为仿射矩阵 (a, b, c, d, e, f)。

    仅处理 translate / scale / matrix（其余忽略，退回单位矩阵）。
    多个 transform 依次左乘。
    """
    a, b, c, d, e, f = 1.0, 0.0, 0.0, 1.0, 0.0, 0.0  # 单位矩阵
    if not transform_str:
        return a, b, c, d, e, f
    for fname, args in _TRANSFORM_FUNC_RE.findall(transform_str):
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", args)
        try:
            vals = [float(n) for n in nums]
        except Exception:
            continue
        fname = fname.lower()
        if fname == "translate":
            tx = vals[0] if vals else 0.0
            ty = vals[1] if len(vals) >= 2 else 0.0
            # 左乘：T * M
            e = a * tx + c * ty + e
            f = b * tx + d * ty + f
        elif fname == "scale":
            sx = vals[0] if vals else 1.0
            sy = vals[1] if len(vals) >= 2 else sx
            a = a * sx
            b = b * sx
            c = c * sy
            d = d * sy
        elif fname == "matrix" and len(vals) == 6:
            ma, mb, mc, md, me, mf = vals
            # (a,b,c,d,e,f) * (ma,mb,mc,md,me,mf)
            na = a * ma + c * mb
            nb = b * ma + d * mb
            nc = a * mc + c * md
            nd = b * mc + d * md
            ne = a * me + c * mf + e
            nf = b * me + d * mf + f
            a, b, c, d, e, f = na, nb, nc, nd, ne, nf
        # rotate / skew 忽略
    return a, b, c, d, e, f


def _compose_matrix(
    m1: tuple[float, float, float, float, float, float],
    m2: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """m1 * m2（先应用 m2，再应用 m1）。"""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _apply_matrix_point(m: tuple[float, float, float, float, float, float], x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = m
    return (a * x + c * y + e, b * x + d * y + f)


def _svg_flatten_transforms(svg_text: str) -> str:
    """把 SVG 中所有 <g transform="..."> 吞掉，将其子元素的坐标改写为全局坐标。

    作用：解决模型在 ``<g transform="translate(300 350)">`` 里写 ``<text y="-230">精神</text>``
    时，下游解析器拿到的是局部坐标的问题。展平后，``<text y>`` 等属性直接就是 viewBox 内
    的绝对坐标，原有所有解析器（_parse_svg_texts / _svg_parse_geometry / ...）无需修改。

    策略：
      * 用 xml.etree.ElementTree 递归遍历，对每个元素累计父级 transform；
      * 对 ``<text>/<tspan>`` 改写 x/y；
      * 对 ``<rect>`` 改写 x/y（w/h 按 scale 缩放）；
      * 对 ``<circle>`` 改写 cx/cy（r 按 scale 缩放）；
      * 对 ``<line>`` 改写 x1/y1/x2/y2；
      * 对 ``<polyline>/<polygon>`` 改写 points 列表；
      * 对 ``<path>`` 仅处理 M/L/H/V（绝对命令），相对命令保持，作近似改写；
      * 解析失败或不含 transform 时返回原文（无副作用）。
    """
    if not svg_text or "transform" not in svg_text:
        return svg_text

    try:
        import xml.etree.ElementTree as ET

        clean = _SVG_NS_RE.sub("", svg_text)
        m_root = re.search(r"<svg\b.*?</svg>", clean, flags=re.DOTALL | re.IGNORECASE)
        if m_root:
            clean = m_root.group(0)
        root = ET.fromstring(clean)
    except Exception:
        return svg_text

    identity = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    def _scale_factor(m: tuple[float, float, float, float, float, float]) -> float:
        """提取各向同性缩放近似因子（仅用于 rect 的 w/h、circle 的 r）。"""
        a, b, c, d, _, _ = m
        sx = (a * a + b * b) ** 0.5
        sy = (c * c + d * d) ** 0.5
        return max(sx, sy) if (sx > 0 and sy > 0) else 1.0

    def _fnum(s, default=0.0) -> float:
        if s is None:
            return default
        m2 = re.match(r"\s*([-+]?\d+(?:\.\d+)?)", str(s))
        if not m2:
            return default
        try:
            return float(m2.group(1))
        except Exception:
            return default

    def _fmt(v: float) -> str:
        if abs(v - round(v)) < 1e-6:
            return str(int(round(v)))
        return f"{v:.3f}".rstrip("0").rstrip(".")

    def _rewrite_points(points_str: str, m) -> str:
        if not points_str:
            return points_str
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", points_str)
        out: list[str] = []
        for i in range(0, len(nums) - 1, 2):
            try:
                x, y = float(nums[i]), float(nums[i + 1])
            except Exception:
                continue
            nx, ny = _apply_matrix_point(m, x, y)
            out.append(f"{_fmt(nx)},{_fmt(ny)}")
        return " ".join(out)

    def _rewrite_path_d(d_str: str, m) -> str:
        """对 path d 做近似重写：处理 M/L 的绝对命令点对，H/V 转 L 再处理。

        相对命令（小写）如 m/l/h/v/c/s/q/t/a 处理会复杂（依赖当前点），这里采用
        简化策略：只对首个 M 做变换（切换到绝对坐标系），后续相对命令保持——
        大多数模型生成的 path 要么全大写、要么开头 M 后跟几个绝对 L；这种近似
        足以让几何解析（如饼图扇区的 M 起点）大致落在全局坐标系。
        """
        if not d_str:
            return d_str
        # 简化：只处理最前面的 M/L/H/V 串，遇到其他字母就 break 保留原样
        tokens = re.findall(r"[MmLlHhVvZzCcSsQqTtAa]|[-+]?\d+(?:\.\d+)?", d_str)
        out: list[str] = []
        i = 0
        cur_x, cur_y = 0.0, 0.0
        while i < len(tokens):
            tk = tokens[i]
            if tk in ("M", "L"):
                out.append(tk)
                i += 1
                # 读一对坐标
                if i + 1 < len(tokens):
                    try:
                        x = float(tokens[i])
                        y = float(tokens[i + 1])
                        nx, ny = _apply_matrix_point(m, x, y)
                        out.append(_fmt(nx))
                        out.append(_fmt(ny))
                        cur_x, cur_y = nx, ny
                        i += 2
                    except Exception:
                        break
                # 后续可能有连续坐标对（隐式 L）
                while i + 1 < len(tokens) and re.match(r"[-+]?\d", tokens[i]):
                    try:
                        x = float(tokens[i])
                        y = float(tokens[i + 1])
                        nx, ny = _apply_matrix_point(m, x, y)
                        out.append(_fmt(nx))
                        out.append(_fmt(ny))
                        cur_x, cur_y = nx, ny
                        i += 2
                    except Exception:
                        break
            elif tk == "H" and i + 1 < len(tokens):
                try:
                    x = float(tokens[i + 1])
                    nx, ny = _apply_matrix_point(m, x, cur_y)
                    out.append("L")
                    out.append(_fmt(nx))
                    out.append(_fmt(ny))
                    cur_x, cur_y = nx, ny
                    i += 2
                except Exception:
                    break
            elif tk == "V" and i + 1 < len(tokens):
                try:
                    y = float(tokens[i + 1])
                    nx, ny = _apply_matrix_point(m, cur_x, y)
                    out.append("L")
                    out.append(_fmt(nx))
                    out.append(_fmt(ny))
                    cur_x, cur_y = nx, ny
                    i += 2
                except Exception:
                    break
            elif tk in ("Z", "z"):
                out.append(tk)
                i += 1
            else:
                # 其余命令（相对 / 曲线）保留原始 tokens 到末尾
                out.extend(tokens[i:])
                break
        return " ".join(out)

    def _apply_to_elem(elem, m):
        """把累计变换 m 施加到 elem 的坐标属性上。"""
        tag = elem.tag.split("}")[-1].lower()
        s = _scale_factor(m)

        if tag in ("text", "tspan"):
            if elem.get("x") is not None or elem.get("y") is not None:
                x = _fnum(elem.get("x"))
                y = _fnum(elem.get("y"))
                nx, ny = _apply_matrix_point(m, x, y)
                elem.set("x", _fmt(nx))
                elem.set("y", _fmt(ny))
        elif tag == "rect":
            x = _fnum(elem.get("x"))
            y = _fnum(elem.get("y"))
            w = _fnum(elem.get("width"))
            h = _fnum(elem.get("height"))
            nx, ny = _apply_matrix_point(m, x, y)
            elem.set("x", _fmt(nx))
            elem.set("y", _fmt(ny))
            if w:
                elem.set("width", _fmt(w * s))
            if h:
                elem.set("height", _fmt(h * s))
        elif tag == "circle":
            cx = _fnum(elem.get("cx"))
            cy = _fnum(elem.get("cy"))
            r = _fnum(elem.get("r"))
            nx, ny = _apply_matrix_point(m, cx, cy)
            elem.set("cx", _fmt(nx))
            elem.set("cy", _fmt(ny))
            if r:
                elem.set("r", _fmt(r * s))
        elif tag == "ellipse":
            cx = _fnum(elem.get("cx"))
            cy = _fnum(elem.get("cy"))
            nx, ny = _apply_matrix_point(m, cx, cy)
            elem.set("cx", _fmt(nx))
            elem.set("cy", _fmt(ny))
        elif tag == "line":
            x1 = _fnum(elem.get("x1"))
            y1 = _fnum(elem.get("y1"))
            x2 = _fnum(elem.get("x2"))
            y2 = _fnum(elem.get("y2"))
            nx1, ny1 = _apply_matrix_point(m, x1, y1)
            nx2, ny2 = _apply_matrix_point(m, x2, y2)
            elem.set("x1", _fmt(nx1))
            elem.set("y1", _fmt(ny1))
            elem.set("x2", _fmt(nx2))
            elem.set("y2", _fmt(ny2))
        elif tag in ("polyline", "polygon"):
            pts = elem.get("points")
            if pts:
                elem.set("points", _rewrite_points(pts, m))
        elif tag == "path":
            d = elem.get("d")
            if d:
                elem.set("d", _rewrite_path_d(d, m))

    def _walk(elem, parent_m):
        """DFS：把自身 transform 累乘到 parent_m，得到当前元素生效矩阵；
        对叶子几何/text 节点施加坐标变换；最后把本元素的 transform 属性清空。"""
        t_attr = elem.get("transform")
        cur_m = parent_m
        if t_attr:
            local_m = _parse_transform(t_attr)
            cur_m = _compose_matrix(parent_m, local_m)
            # 清空 transform，避免重复施加
            elem.set("transform", "")

        # 应用到当前元素的坐标属性（若是几何/文本节点）
        _apply_to_elem(elem, cur_m)

        # 递归子元素
        for child in list(elem):
            _walk(child, cur_m)

    try:
        _walk(root, identity)
        out = ET.tostring(root, encoding="unicode")
        return out
    except Exception:
        return svg_text


def _parse_svg_texts(svg_text: str) -> list[tuple[float, float, str]]:
    """从 SVG 源码中抽取所有 ``<text>`` / ``<tspan>`` 节点的 (x, y, 文本) 列表。

    策略：
      1) 优先使用 xml.etree.ElementTree 解析（剥离命名空间以简化遍历）；
      2) 解析失败时回退到正则，匹配 ``<text ...>文本</text>`` 和 ``<tspan ...>文本</tspan>``；
      3) 单个 ``<text>`` 内若含多个 ``<tspan>``，按 tspan 自身坐标或父 text 坐标分别记录。

    Args:
        svg_text: 纯 SVG 源码（已剥离代码围栏）

    Returns:
        [(x, y, text)] 列表；坐标缺失时填 0.0。文本为 strip 后的字符串（保留空格，空串跳过）。
    """
    if not svg_text or "<" not in svg_text:
        return []

    nodes: list[tuple[float, float, str]] = []

    # --- 尝试 ET 解析 ---
    try:
        import xml.etree.ElementTree as ET

        clean = _SVG_NS_RE.sub("", svg_text)
        # 兜底：匹配根节点 <svg ...> ... </svg>
        m_root = re.search(r"<svg\b.*?</svg>", clean, flags=re.DOTALL | re.IGNORECASE)
        if m_root:
            clean = m_root.group(0)

        root = ET.fromstring(clean)

        def _to_float(s: str | None) -> float | None:
            if not s:
                return None
            # 去掉单位 px / pt / em 等
            m = re.match(r"\s*([-+]?\d+(?:\.\d+)?)", s)
            if not m:
                return None
            try:
                return float(m.group(1))
            except Exception:
                return None

        def _collect_text(elem, inherited_x: float | None, inherited_y: float | None) -> str:
            """递归收集一个 <text> 子树内的所有文本，返回合并后的字符串。"""
            parts: list[str] = []
            if elem.text:
                parts.append(elem.text)
            for child in list(elem):
                tag = child.tag.split("}")[-1].lower()
                if tag == "tspan":
                    # tspan 可能有自己的绝对坐标 x/y，或相对偏移 dx/dy
                    raw_x = child.get("x")
                    raw_y = child.get("y")
                    raw_dx = child.get("dx")
                    raw_dy = child.get("dy")
                    cx = _to_float(raw_x) if raw_x is not None else inherited_x
                    cy = _to_float(raw_y) if raw_y is not None else inherited_y
                    # 累加 dx/dy 偏移
                    if raw_dx is not None:
                        dx = _to_float(raw_dx)
                        if dx is not None and cx is not None:
                            cx = cx + dx
                    if raw_dy is not None:
                        dy = _to_float(raw_dy)
                        if dy is not None and cy is not None:
                            cy = cy + dy
                    # 如果 tspan 有独立坐标/偏移，单独记为一个节点
                    sub_text = _collect_text(child, cx, cy)
                    has_own_pos = raw_x is not None or raw_y is not None or raw_dx is not None or raw_dy is not None
                    if sub_text.strip():
                        if has_own_pos:
                            nodes.append((cx or 0.0, cy or 0.0, sub_text.strip()))
                        else:
                            parts.append(sub_text)
                else:
                    # 其他子元素（不常见），递归并合并文本
                    sub_text = _collect_text(child, inherited_x, inherited_y)
                    if sub_text.strip():
                        parts.append(sub_text)
                if child.tail:
                    parts.append(child.tail)
            return "".join(parts)

        for elem in root.iter():
            tag = elem.tag.split("}")[-1].lower()
            if tag != "text":
                continue
            x = _to_float(elem.get("x")) or 0.0
            y = _to_float(elem.get("y")) or 0.0
            text = _collect_text(elem, x, y)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                nodes.append((x, y, text))

        if nodes:
            return nodes
    except Exception:
        pass

    # --- 正则兜底 ---
    pat_text = re.compile(
        r"<text\b([^>]*)>(.*?)</text>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    pat_tspan = re.compile(
        r"<tspan\b([^>]*)>(.*?)</tspan>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    pat_xy = re.compile(r'(x|y)\s*=\s*"\s*([-+]?\d+(?:\.\d+)?)', re.IGNORECASE)

    def _inner_text(raw: str) -> str:
        # 去掉嵌套标签（保留 tspan 文字由专门处理）
        txt = re.sub(r"<[^>]+>", " ", raw)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    for m in pat_text.finditer(svg_text):
        attrs, body = m.group(1), m.group(2)
        coords = dict(pat_xy.findall(attrs))
        try:
            px = float(coords.get("x", 0.0))
            py = float(coords.get("y", 0.0))
        except Exception:
            px, py = 0.0, 0.0

        # 先处理内部 tspan（若有独立坐标）
        tspan_chunks: list[tuple[float, float, str]] = []
        for mt in pat_tspan.finditer(body):
            ta, tbody = mt.group(1), mt.group(2)
            tc = dict(pat_xy.findall(ta))
            if "x" in tc or "y" in tc:
                try:
                    tx = float(tc.get("x", px))
                    ty = float(tc.get("y", py))
                except Exception:
                    tx, ty = px, py
                tt = _inner_text(tbody)
                if tt:
                    tspan_chunks.append((tx, ty, tt))

        if tspan_chunks:
            nodes.extend(tspan_chunks)
        else:
            txt = _inner_text(body)
            if txt:
                nodes.append((px, py, txt))

    return nodes


_SVG_LABEL_DROP_PATTERNS = (
    re.compile(r"^[\W_]+$", re.UNICODE),  # 纯符号
)


def _parse_svg_parent_map(svg_text: str) -> dict[str, list[str]]:
    """从 SVG 源码中解析 ``<g data-parent="X">...<text>Y</text>...</g>`` 这类
    显式父子关系的标注，返回 ``{parent: [child, ...]}`` 字典。

    约定（由新版 SE_SVG_PROMPT_LOGIC 驱动模型输出）：
      * 每个节点被单独的 ``<g>`` 容器包裹，``<g>`` 上带 ``data-parent="父节点文本"``；
      * 根节点的 ``data-parent=""`` 或缺省（将归入 ``""`` 键）；
      * ``<g>`` 内部的第一个 ``<text>`` 文字视为该节点文本。

    若 SVG 里没有这种约定，返回空字典。
    """
    if not svg_text or "data-parent" not in svg_text:
        return {}

    parent_map: dict[str, list[str]] = {}
    # 非贪婪匹配整块 <g ... data-parent="..."> ... </g>
    pat_g = re.compile(
        r'<g\b[^>]*\bdata-parent\s*=\s*"([^"]*)"[^>]*>(.*?)</g>',
        flags=re.DOTALL | re.IGNORECASE,
    )
    pat_inner_text = re.compile(
        r"<text\b[^>]*>(.*?)</text>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    for m in pat_g.finditer(svg_text):
        parent = m.group(1).strip()
        block = m.group(2)
        # 取块内**第一个** <text> 作为该节点自身文本
        it = pat_inner_text.search(block)
        if not it:
            continue
        raw = it.group(1)
        child = re.sub(r"<[^>]+>", " ", raw)
        child = re.sub(r"\s+", " ", child).strip()
        if not child:
            continue
        parent_map.setdefault(parent, []).append(child)
    return parent_map


def _build_md_from_parent_map(parent_map: dict[str, list[str]]) -> str:
    """由 ``{parent: [children]}`` 构造 Markdown 多级无序列表。

    根节点选取：
      * 优先使用 ``parent_map[""]`` 里的节点作为根；
      * 若空键不存在，取所有出现在 key 但未出现在任何 value 中的节点；
      * 若仍找不到根，取任意一个 key（避免死循环）。
    """
    if not parent_map:
        return ""

    # 所有被当作子节点引用过的名字
    all_children = set()
    for vs in parent_map.values():
        all_children.update(vs)
    all_parents = set(parent_map.keys()) - {""}

    roots: list[str] = []
    if "" in parent_map:
        roots.extend(parent_map[""])
    # 有父名但自身不在别人的子列表里 → 也视为根
    for p in all_parents:
        if p not in all_children and p not in roots:
            roots.append(p)
    if not roots and all_parents:
        roots = [next(iter(all_parents))]
    if not roots:
        return ""

    visited: set[str] = set()
    lines: list[str] = []

    def _dfs(name: str, level: int) -> None:
        if name in visited:
            return
        visited.add(name)
        lines.append(f"{'  ' * level}- {name}")
        for child in parent_map.get(name, []):
            _dfs(child, level + 1)

    for r in roots:
        _dfs(r, 0)
    return "\n".join(lines)


def _svg_text_looks_like_number(s: str) -> bool:
    """判断一段文本是否主要表达数值（支持空格千分位、百分号、范围、货币符号、中文单位）。

    注意：
      * 只含"括号 + 纯整数"的形如 ``(3)`` ``（5）`` 不视为数值（它们通常是
        逻辑类图中的阶段/序号标注，例如"顶部(3)"）；
      * 带前缀文字的数字（如"2015年"）也不视为数值；
    """
    if not s:
        return False
    t = s.strip()
    if not t:
        return False
    # 纯括号内整数：(3) （5）→ 不视为数值
    if re.fullmatch(r"[（(]\s*[-+]?\d+\s*[)）]", t):
        return False
    # 去掉尾部中文标点（如 "8.3%、" "9.1%。"）
    t = t.rstrip("、。，,.")
    if not t:
        return False
    # 去掉货币符号前缀
    compact = t
    for ch in ("$", "￥", "¥", "€", "£"):
        compact = compact.replace(ch, "")
    # 去掉中文单位后缀（万、亿、元、亿元、万元 等）
    compact = re.sub(r"[万亿元]+$", "", compact)
    # 去掉空格千分位 + % + 逗号
    compact = compact.replace(" ", "").replace(",", "").replace("%", "").replace("％", "")
    if not compact:
        return False
    # 允许首尾符号：+-、括号（负数）
    compact = compact.replace("(", "").replace(")", "").replace("（", "").replace("）", "")
    try:
        float(compact)
        return True
    except Exception:
        pass
    # 范围如 "2015-2019"
    m = re.match(r"^[-+]?\d+(?:\.\d+)?[\s\-—~～]+[-+]?\d+(?:\.\d+)?$", t)
    if m:
        return True
    return False


def _svg_parse_number(s: str) -> float | None:
    """尝试把一段文本解析为数值；解析失败返回 None。

    支持：
      * 整数 / 小数 / 负数 / 带正号
      * 百分号（返回 **原始数字值**，不除以 100）
      * 空格千分位（例如 "29 500"）
      * 全/半角逗号千分位（例如 "12,345"）
      * 货币符号前缀（$、￥、€、£）
      * 单位后缀（%，‰ 保留为普通数字）
    """
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    # 去掉尾部中文标点（如 "8.3%、" "9.1%。"）
    t = t.rstrip("、。，,.")
    if not t:
        return None
    # 纯括号整数 → None（让它被视为 label）
    if re.fullmatch(r"[（(]\s*[-+]?\d+\s*[)）]", t):
        return None
    # 移除常见前缀与千分位/百分号等
    t2 = t
    for ch in ("$", "￥", "¥", "€", "£"):
        t2 = t2.replace(ch, "")
    t2 = t2.replace(",", "").replace(" ", "").replace("\u00a0", "")
    # 处理中文单位后缀（万、亿、元、亿元、万元）
    cn_unit_multiplier = 1.0
    if t2.endswith("亿元") or t2.endswith("亿"):
        cn_unit_multiplier = 1.0  # 保留原始数字，不乘以倍数（ref 中也是 "70.3亿元" 格式）
        t2 = re.sub(r"[万亿元]+$", "", t2)
    elif t2.endswith("万元") or t2.endswith("万"):
        cn_unit_multiplier = 1.0
        t2 = re.sub(r"[万亿元]+$", "", t2)
    elif t2.endswith("元"):
        t2 = t2[:-1]
    # 记录百分号以便还原
    has_pct = "%" in t2 or "％" in t2
    t2 = t2.replace("%", "").replace("％", "").replace("‰", "")
    # 外层括号表示负数（但如果含有%则不视为负数，如 "(20.41%)" 是正数百分比）
    neg_by_paren = False
    if (t2.startswith("(") and t2.endswith(")")) or (t2.startswith("（") and t2.endswith("）")):
        t2 = t2[1:-1]
        if not has_pct:
            neg_by_paren = True
    try:
        v = float(t2)
        if neg_by_paren:
            v = -v
        return v
    except Exception:
        return None


def _cluster_1d(values: list[float], tol: float) -> list[list[int]]:
    """对一维值做链式聚类（按值排序后，相邻间距 ≤ tol 归入同一簇）。

    Args:
        values: 待聚类的数值列表
        tol: 同簇容差

    Returns:
        list of index-clusters（每个簇是原 values 的下标列表）
    """
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    clusters: list[list[int]] = []
    cur: list[int] = []
    last_v: float | None = None
    for i in order:
        v = values[i]
        if last_v is None or abs(v - last_v) <= tol:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
        last_v = v
    if cur:
        clusters.append(cur)
    return clusters


# ============================================================
# SVG 几何图元解析 —— 用于柱状 / 折线 / 箱线 / 组合 / 雷达图
# ============================================================

_SVG_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SVG_TAG_ATTR_RE = re.compile(r"<(rect|line|polyline|polygon|circle|path)\b([^>]*?)/?>", re.IGNORECASE | re.DOTALL)
_SVG_ATTR_RE = re.compile(r'([a-zA-Z_:\-]+)\s*=\s*"([^"]*)"')


def _svg_parse_geometry(svg_text: str) -> dict:
    """从 SVG 源码里解析几何图元（简化版，仅拿到评分需要的信息）。

    Returns:
        dict with keys:
          * ``rects``: [(x, y, w, h)]
          * ``lines``: [(x1, y1, x2, y2)]
          * ``polylines``: [[(x, y), ...]]
          * ``polygons``: [[(x, y), ...]]
          * ``circles``: [(cx, cy, r)]
          * ``paths``: [[(x, y), ...]]  # 只抽 M/L/H/V 的点，近似折线
    """
    out = {
        "rects": [],
        "lines": [],
        "polylines": [],
        "polygons": [],
        "circles": [],
        "paths": [],
    }
    if not svg_text:
        return out
    # 去注释避免把注释里的数字当成坐标
    cleaned = _SVG_COMMENT_RE.sub("", svg_text)

    def _fnum(s: str, default: float = 0.0) -> float:
        if s is None:
            return default
        m = re.match(r"\s*([-+]?\d+(?:\.\d+)?)", s)
        if not m:
            return default
        try:
            return float(m.group(1))
        except Exception:
            return default

    def _points_list(s: str) -> list[tuple[float, float]]:
        """把 "x1,y1 x2,y2 ..." 解析成 [(x, y), ...]，兼容逗号/空格/换行混用。"""
        if not s:
            return []
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
        pts = []
        for i in range(0, len(nums) - 1, 2):
            try:
                pts.append((float(nums[i]), float(nums[i + 1])))
            except Exception:
                pass
        return pts

    def _path_points(d: str) -> list[tuple[float, float]]:
        """从 path d 属性粗略抽取锚点：仅处理 M/m/L/l/H/h/V/v，其它命令（C/S/Q/A）忽略曲线控制点。"""
        if not d:
            return []
        pts: list[tuple[float, float]] = []
        tokens = re.findall(r"[A-Za-z]|[-+]?\d+(?:\.\d+)?", d)
        i = 0
        cur_x = 0.0
        cur_y = 0.0
        cur_cmd = "M"
        while i < len(tokens):
            tok = tokens[i]
            if tok.isalpha():
                cur_cmd = tok
                i += 1
                continue
            try:
                if cur_cmd in ("M", "L"):
                    x = float(tokens[i])
                    y = float(tokens[i + 1])
                    cur_x, cur_y = x, y
                    pts.append((cur_x, cur_y))
                    i += 2
                elif cur_cmd in ("m", "l"):
                    dx = float(tokens[i])
                    dy = float(tokens[i + 1])
                    cur_x += dx
                    cur_y += dy
                    pts.append((cur_x, cur_y))
                    i += 2
                elif cur_cmd == "H":
                    cur_x = float(tokens[i])
                    pts.append((cur_x, cur_y))
                    i += 1
                elif cur_cmd == "h":
                    cur_x += float(tokens[i])
                    pts.append((cur_x, cur_y))
                    i += 1
                elif cur_cmd == "V":
                    cur_y = float(tokens[i])
                    pts.append((cur_x, cur_y))
                    i += 1
                elif cur_cmd == "v":
                    cur_y += float(tokens[i])
                    pts.append((cur_x, cur_y))
                    i += 1
                elif cur_cmd in ("Z", "z"):
                    # 关闭路径，不推进
                    pass
                else:
                    # 其它命令（曲线）：简单跳过 2 个数
                    i += 1
                    continue
            except (ValueError, IndexError):
                i += 1
                continue

        return pts

    for m in _SVG_TAG_ATTR_RE.finditer(cleaned):
        tag = m.group(1).lower()
        attrs = dict(_SVG_ATTR_RE.findall(m.group(2)))

        if tag == "rect":
            x = _fnum(attrs.get("x"))
            y = _fnum(attrs.get("y"))
            w = _fnum(attrs.get("width"))
            h = _fnum(attrs.get("height"))
            if w > 0 and h > 0:
                out["rects"].append((x, y, w, h))
        elif tag == "line":
            x1 = _fnum(attrs.get("x1"))
            y1 = _fnum(attrs.get("y1"))
            x2 = _fnum(attrs.get("x2"))
            y2 = _fnum(attrs.get("y2"))
            out["lines"].append((x1, y1, x2, y2))
        elif tag == "polyline":
            pts = _points_list(attrs.get("points", ""))
            if len(pts) >= 2:
                out["polylines"].append(pts)
        elif tag == "polygon":
            pts = _points_list(attrs.get("points", ""))
            if len(pts) >= 3:
                out["polygons"].append(pts)
        elif tag == "circle":
            cx = _fnum(attrs.get("cx"))
            cy = _fnum(attrs.get("cy"))
            r = _fnum(attrs.get("r"))
            if r > 0:
                out["circles"].append((cx, cy, r))
        elif tag == "path":
            pts = _path_points(attrs.get("d", ""))
            if len(pts) >= 2:
                out["paths"].append(pts)

    return out


def _svg_detect_xy_axes(
    nodes: list[tuple[float, float, str]],
    tol_x: float,
    tol_y: float,
) -> dict:
    """识别 X 轴类别行 + 左/右 Y 轴刻度列。

    返回字典：
        * ``cats``:     [(x, y, label), ...] 按 x 升序（X 轴类别）；缺失则空
        * ``y_left``:   (list[(y_pixel, y_value)], text_x_median)  按 y 升序
        * ``y_right``:  同上
        * ``plot_box``: (xmin, xmax, ymin, ymax)  绘图区范围（由 cats 和 y 刻度合成）
    """
    result = {"cats": [], "y_left": None, "y_right": None, "plot_box": None}
    if not nodes:
        return result

    xs = [x for x, _, _ in nodes]
    ys = [y for _, y, _ in nodes]
    x_span = (max(xs) - min(xs)) if xs else 0.0
    y_span = (max(ys) - min(ys)) if ys else 0.0

    # ---- 识别 Y 轴刻度（数值 + x 几乎相同 + y 分布跨度大） ----
    numeric_idxs = [i for i, (_, _, t) in enumerate(nodes) if _svg_text_looks_like_number(t)]
    x_clusters_idx = _cluster_1d([nodes[i][0] for i in numeric_idxs], tol=max(tol_x, 4.0))

    y_axis_candidates: list[tuple[list[int], float, float]] = []  # (idx_list, x_median, y_range)
    for cluster_of_numeric_positions in x_clusters_idx:
        idxs = [numeric_idxs[j] for j in cluster_of_numeric_positions]
        if len(idxs) < 3:
            continue
        ys_in = sorted(nodes[i][1] for i in idxs)
        y_range = ys_in[-1] - ys_in[0]
        # 要求覆盖 y 跨度的至少 40%
        if y_span > 0 and y_range < 0.4 * y_span:
            continue
        # 要求刻度数值 *单调* 或 *近似单调*（避免把横排的数据值当 Y 刻度）
        pairs = sorted(
            [(nodes[i][1], _svg_parse_number(nodes[i][2])) for i in idxs],
            key=lambda p: p[0],
        )
        vals = [v for _, v in pairs if v is not None]
        if len(vals) < 3:
            continue
        inc = sum(1 for a, b in zip(vals, vals[1:]) if b > a)
        dec = sum(1 for a, b in zip(vals, vals[1:]) if b < a)
        mono = max(inc, dec) / max(1, len(vals) - 1)
        if mono < 0.7:
            continue
        xs_in = [nodes[i][0] for i in idxs]
        x_med = sum(xs_in) / len(xs_in)
        y_axis_candidates.append((idxs, x_med, y_range))

    # 选出最靠左 / 最靠右的两条作为 y_left / y_right
    y_axis_candidates.sort(key=lambda p: p[1])
    left_axis = y_axis_candidates[0] if y_axis_candidates else None
    right_axis = (
        y_axis_candidates[-1] if len(y_axis_candidates) >= 2 and y_axis_candidates[-1][1] != left_axis[1] else None
    )

    def _pack_axis(idxs: list[int]) -> list[tuple[float, float]]:
        pairs = []
        for i in idxs:
            v = _svg_parse_number(nodes[i][2])
            if v is None:
                continue
            pairs.append((nodes[i][1], v))
        pairs.sort(key=lambda p: p[0])
        return pairs

    if left_axis:
        result["y_left"] = _pack_axis(left_axis[0])
        result["y_left_x"] = left_axis[1]
    if right_axis:
        result["y_right"] = _pack_axis(right_axis[0])
        result["y_right_x"] = right_axis[1]

    # ---- 识别 X 轴类别行（按 y 聚类、文本多数非数字、x 分布均匀） ----
    # 排除已被用作 Y 轴刻度的 text，以及所有在 Y 轴 x 附近的其它数字 text，避免
    # 把 Y 轴最下方的刻度（0/50 这种靠近 X 轴位置）当作类别行节点
    axis_node_idxs: set[int] = set()
    if left_axis:
        axis_node_idxs.update(left_axis[0])
    if right_axis:
        axis_node_idxs.update(right_axis[0])

    # 额外：贴近左/右 Y 轴 x 的数字 text（可能是轴上漏掉的刻度）
    def _near_axis_x(x: float) -> bool:
        if result.get("y_left_x") is not None and abs(x - result["y_left_x"]) <= max(tol_x, 10.0):
            return True
        if result.get("y_right_x") is not None and abs(x - result["y_right_x"]) <= max(tol_x, 10.0):
            return True
        return False

    cat_candidate_idxs = [
        i
        for i in range(len(nodes))
        if i not in axis_node_idxs and not (_svg_text_looks_like_number(nodes[i][2]) and _near_axis_x(nodes[i][0]))
    ]
    cand_set = set(cat_candidate_idxs)

    y_clusters = _cluster_1d(ys, tol_y)

    def _score_cat_row(idxs: list[int]) -> float:
        if len(idxs) < 2:
            return -1.0
        # 尽量避免把 Y 刻度整行错当 X 类别
        xs_in = sorted(nodes[i][0] for i in idxs)
        spread = xs_in[-1] - xs_in[0]
        if x_span > 0 and spread < 0.4 * x_span:
            return -1.0
        # 非数值文本占比：高更好
        non_num_ratio = sum(1 for i in idxs if not _svg_text_looks_like_number(nodes[i][2])) / len(idxs)
        # 均匀度
        if len(xs_in) >= 2:
            gaps = [xs_in[k + 1] - xs_in[k] for k in range(len(xs_in) - 1)]
            mean_gap = sum(gaps) / len(gaps)
            var_gap = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps) if gaps else 0.0
            uniform = 1.0 / (1.0 + var_gap / (mean_gap * mean_gap + 1e-6))
        else:
            uniform = 0.0
        # 行的 y：越偏下越好（绘图区底部通常是 X 轴）
        y_mean = sum(nodes[i][1] for i in idxs) / len(idxs)
        bottom_bonus = (y_mean - min(ys)) / max(y_span, 1.0)
        # 节点数、x 跨度、均匀度、非数占比、下方奖励 加权
        return (
            len(idxs) * 1.2
            + (spread / max(x_span, 1.0)) * 2.5
            + uniform * 1.5
            + non_num_ratio * 2.0
            + bottom_bonus * 1.5
        )

    best_idxs: list[int] | None = None
    best_score = -1.0
    for idxs in y_clusters:
        # 过滤掉不在候选集里的节点
        filtered = [i for i in idxs if i in cand_set]
        if len(filtered) < 2:
            continue
        score = _score_cat_row(filtered)
        if score > best_score:
            best_score = score
            best_idxs = filtered

    if best_idxs and len(best_idxs) >= 2:
        cats = sorted([nodes[i] for i in best_idxs], key=lambda n: n[0])
        result["cats"] = cats

    # ---- 绘图区 ----
    if result["cats"] and (result["y_left"] or result["y_right"]):
        xs_c = [c[0] for c in result["cats"]]
        ys_axis = []
        if result["y_left"]:
            ys_axis += [y for y, _ in result["y_left"]]
        if result["y_right"]:
            ys_axis += [y for y, _ in result["y_right"]]
        cats_y = sum(n[1] for n in result["cats"]) / len(result["cats"])
        # 绘图区下沿用 X 类别行的 y，上沿用 Y 刻度最小 y
        result["plot_box"] = (
            min(xs_c),
            max(xs_c),
            min(ys_axis) if ys_axis else min(ys),
            cats_y,
        )

    return result


def _svg_make_y_interpolator(axis_points: list[tuple[float, float]]):
    """基于 [(y_pixel, y_value), ...] 构造 y_pixel → y_value 的线性插值函数。

    要求点已按 y_pixel 升序。两点及以上才有效；返回 None 表示无法插值。
    """
    if not axis_points or len(axis_points) < 2:
        return None
    pts = list(axis_points)
    # 保证 y_pixel 严格递增
    pts.sort(key=lambda p: p[0])
    dedup: list[tuple[float, float]] = []
    for p in pts:
        if not dedup or p[0] - dedup[-1][0] > 1e-6:
            dedup.append(p)
    if len(dedup) < 2:
        return None

    def interp(y_pixel: float) -> float:
        if y_pixel <= dedup[0][0]:
            # 在首端外推
            (p0_y, p0_v), (p1_y, p1_v) = dedup[0], dedup[1]
        elif y_pixel >= dedup[-1][0]:
            (p0_y, p0_v), (p1_y, p1_v) = dedup[-2], dedup[-1]
        else:
            # 二分定位
            lo, hi = 0, len(dedup) - 1
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if dedup[mid][0] <= y_pixel:
                    lo = mid
                else:
                    hi = mid
            (p0_y, p0_v), (p1_y, p1_v) = dedup[lo], dedup[lo + 1]
        if abs(p1_y - p0_y) < 1e-9:
            return p0_v
        return p0_v + (p1_v - p0_v) * (y_pixel - p0_y) / (p1_y - p0_y)

    return interp


def _svg_format_number(v: float, has_pct: bool = False, decimals_hint: int | None = None) -> str:
    """把反推出的数值格式化为接近参考答案风格的字符串。

    启发式：
      * 绝对值 ≥ 100 或非小数位→ 输出整数；
      * 否则根据 decimals_hint 或自适应保留 1~2 位小数。
    """
    if v is None:
        return ""
    sign = "-" if v < 0 else ""
    av = abs(v)
    if decimals_hint is not None:
        s = f"{av:.{decimals_hint}f}"
    elif av >= 100 or abs(av - round(av)) < 0.05:
        s = f"{int(round(av))}"
    elif av >= 10:
        s = f"{av:.1f}"
    else:
        s = f"{av:.2f}"
    # 去除无意义的尾随 0
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if has_pct:
        s = s + "%"
    return sign + s


def _svg_infer_decimals(axis_vals: list[float], ref_has_pct: bool) -> tuple[int | None, bool]:
    """基于 Y 轴刻度推测小数位数和是否使用百分号。

    返回 (decimals_hint, has_pct_flag)
    """
    if not axis_vals:
        return None, ref_has_pct
    dec_hint: int | None = None
    max_dec = 0
    for v in axis_vals:
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        if "." in s:
            max_dec = max(max_dec, len(s.split(".")[1]))
    if max_dec > 0:
        dec_hint = max_dec
    return dec_hint, ref_has_pct


def _svg_pick_axis_for_x(
    cx: float,
    y_left: list[tuple[float, float]] | None,
    y_left_x: float | None,
    y_right: list[tuple[float, float]] | None,
    y_right_x: float | None,
    prefer: str = "left",
) -> tuple[list[tuple[float, float]] | None, float | None]:
    """根据几何图元的 x 位置与偏好，决定使用左/右 Y 轴。

    ``prefer``：
      * ``"left"``（默认）：强制使用左轴；单轴时退回唯一轴；
      * ``"right"``：强制使用右轴；单轴时退回唯一轴；
      * ``"nearest"``：按距离选择（旧行为）。
    """
    if prefer == "left":
        if y_left:
            return y_left, y_left_x
        if y_right:
            return y_right, y_right_x
        return None, None
    if prefer == "right":
        if y_right:
            return y_right, y_right_x
        if y_left:
            return y_left, y_left_x
        return None, None
    # nearest
    if y_left and y_right and y_left_x is not None and y_right_x is not None:
        d_l = abs(cx - y_left_x)
        d_r = abs(cx - y_right_x)
        if d_r + 5.0 < d_l:
            return y_right, y_right_x
        return y_left, y_left_x
    if y_left:
        return y_left, y_left_x
    if y_right:
        return y_right, y_right_x
    return None, None


def _svg_rect_is_in_plot(rect: tuple[float, float, float, float], plot_box) -> bool:
    if not plot_box:
        return True
    x, y, w, h = rect
    xmin, xmax, ymin, ymax = plot_box
    cx = x + w / 2
    # cy 取柱中心
    cy = y + h / 2
    # x 方向放宽一点（防止首末类别靠边），y 方向相对收紧
    dx = (xmax - xmin) * 0.10 + 5.0
    dy_up = (ymax - ymin) * 0.05 + 5.0
    dy_down = (ymax - ymin) * 0.15 + 5.0
    if not ((xmin - dx) <= cx <= (xmax + dx)):
        return False
    # 要求 **rect 顶 y** 不能比 ymin 高太多（否则就是图例等位于绘图区上方的元素）
    if y < ymin - dy_up:
        return False
    # rect 底 y 不能超过 x 轴下方太多
    if (y + h) > ymax + dy_down:
        return False
    return True


def _svg_assign_rects_to_categories(
    rects: list[tuple[float, float, float, float]],
    cats: list[tuple[float, float, str]],
    plot_box,
) -> dict[int, list[tuple[float, float, float, float]]]:
    """把每个 <rect> 分配到最近的 X 类别（按柱中心 x 距离）。

    过滤：
      * 不在绘图区的 rect 剔除；
      * 过宽的 rect（宽度 > 相邻类别间距的 3 倍）剔除，避免背景块；
    """
    if not cats:
        return {}
    cat_xs = [c[0] for c in cats]
    if len(cat_xs) >= 2:
        gaps = [cat_xs[i + 1] - cat_xs[i] for i in range(len(cat_xs) - 1)]
        mean_gap = sum(gaps) / len(gaps)
    else:
        mean_gap = 50.0

    assign: dict[int, list[tuple[float, float, float, float]]] = {i: [] for i in range(len(cats))}
    # 计算绘图区高度，用于过滤高度过小的 rect（图例等）
    plot_h = None
    if plot_box:
        plot_h = plot_box[3] - plot_box[2]

    for rect in rects:
        x, y, w, h = rect
        if not _svg_rect_is_in_plot(rect, plot_box):
            continue
        # 过大的背景 rect：宽度 > 2.5 * 平均间距 视为非数据柱
        if mean_gap > 0 and w > mean_gap * 2.5:
            continue
        # 高度过小剔除：绝对高度 < 2 或者 相对于绘图区 < 2%
        if h < 2:
            continue
        if plot_h and h < plot_h * 0.02:
            continue
        cx = x + w / 2
        # 归入最近类别
        i = min(range(len(cat_xs)), key=lambda j: abs(cat_xs[j] - cx))
        # 距离过远（超过 1.5 * mean_gap）也剔除
        if mean_gap > 0 and abs(cat_xs[i] - cx) > mean_gap * 0.9:
            continue
        assign[i].append(rect)

    # 每个类别内按 x 升序
    for i in assign:
        assign[i].sort(key=lambda r: r[0])
    return assign


# ============================================================
# SVG → CSV 的按图表类型专项解析
# ============================================================


def _svg_has_pct_in_ref(nodes: list[tuple[float, float, str]]) -> bool:
    """判断 SVG 的数值文本里是否常带 % 号（全局层面，用于饼图）。"""
    cnt = 0
    pct = 0
    for _, _, t in nodes:
        if _svg_text_looks_like_number(t):
            cnt += 1
            if "%" in t:
                pct += 1
    return cnt > 0 and pct / cnt >= 0.4


def _svg_axis_has_pct(
    axis_pts: list[tuple[float, float]] | None,
    nodes: list[tuple[float, float, str]],
    axis_x: float | None,
    tol_x: float,
) -> bool:
    """判断某条 Y 轴的刻度文本是否主要是百分比。

    通过查找 axis_x 附近（tol_x 容差内）且 y 与 axis_pts 相近的数字 text，
    统计其中含 ``%`` 的比例。
    """
    if not axis_pts or axis_x is None or not nodes:
        return False
    axis_ys = {round(y, 1) for y, _ in axis_pts}
    hits = 0
    total = 0
    for x, y, t in nodes:
        if not _svg_text_looks_like_number(t):
            continue
        if abs(x - axis_x) > max(tol_x, 15.0):
            continue
        # 找最接近的 axis y
        if not any(abs(y - ay) <= 5.0 for ay in axis_ys):
            continue
        total += 1
        if "%" in t or "％" in t:
            hits += 1
    if total == 0:
        return False
    return hits / total >= 0.5


def _svg_to_csv_bar(
    nodes: list[tuple[float, float, str]],
    geom: dict,
    axes: dict,
) -> str:
    """柱状图/组合图 的柱部分：每个 <rect> 的高度经 Y 轴校准反推数值。"""
    cats = axes.get("cats") or []
    rects = geom.get("rects") or []
    if not cats or not rects:
        return ""

    y_left = axes.get("y_left")
    y_right = axes.get("y_right")
    y_left_x = axes.get("y_left_x")
    y_right_x = axes.get("y_right_x")
    if not y_left and not y_right:
        return ""

    assign = _svg_assign_rects_to_categories(rects, cats, axes.get("plot_box"))
    # 每类列数众数
    counts = [len(assign[i]) for i in assign]
    if not counts or max(counts) == 0:
        return ""
    from collections import Counter

    target_cols = Counter(counts).most_common(1)[0][0]
    if target_cols == 0:
        target_cols = max(counts)

    # 小数位与百分号（按轴判断）
    xs_all = [n[0] for n in nodes]
    x_span = (max(xs_all) - min(xs_all)) if xs_all else 0.0
    tol_x = max(x_span * 0.04, 2.0) if x_span > 0 else 2.0
    has_pct_left = _svg_axis_has_pct(y_left, nodes, y_left_x, tol_x)
    has_pct_right = _svg_axis_has_pct(y_right, nodes, y_right_x, tol_x)
    left_vals = [v for _, v in (y_left or [])]
    right_vals = [v for _, v in (y_right or [])]
    dec_l, _ = _svg_infer_decimals(left_vals, has_pct_left)
    dec_r, _ = _svg_infer_decimals(right_vals, has_pct_right)

    lines: list[str] = []
    header = [""] + [""] * target_cols
    lines.append(" \\t ".join(header))

    for i, cat in enumerate(cats):
        cat_label = _clean_label(cat[2])
        cells = [cat_label]
        rects_in_cat = assign.get(i, [])
        for k in range(target_cols):
            if k < len(rects_in_cat):
                rx, ry, rw, rh = rects_in_cat[k]
                cx = rx + rw / 2
                # 柱状图：所有柱统一用左轴（主数值轴）
                axis_pts, axis_x = _svg_pick_axis_for_x(cx, y_left, y_left_x, y_right, y_right_x, prefer="left")
                interp = _svg_make_y_interpolator(axis_pts) if axis_pts else None
                if interp is None:
                    cells.append("")
                    continue
                # 柱顶 y（SVG 坐标系里 y 越小越靠上；柱顶 = ry）
                v_top = interp(ry)
                v_bot = interp(ry + rh)
                # 数值 = 柱顶 - 柱底（若柱底 ≈ 轴零点，结果就是柱顶值）
                val = v_top - v_bot
                # 多数情况下基线在 0，这里把 "负值反转 = 真正正值" 的常规柱形还原出来
                val = abs(val) if abs(val) > 1e-6 else v_top
                use_left = axis_pts is y_left
                dec = dec_l if use_left else dec_r
                has_pct_here = has_pct_left if use_left else has_pct_right
                cells.append(_svg_format_number(val, has_pct=has_pct_here, decimals_hint=dec))
            else:
                cells.append("")
        lines.append(" \\t ".join(cells))

    return " \\n ".join(lines)


def _svg_to_csv_line(
    nodes: list[tuple[float, float, str]],
    geom: dict,
    axes: dict,
) -> str:
    """折线图：从 polyline/path 取顶点，按 x 近邻归入类别。"""
    cats = axes.get("cats") or []
    if not cats:
        return ""
    y_left = axes.get("y_left")
    y_right = axes.get("y_right")
    y_left_x = axes.get("y_left_x")
    y_right_x = axes.get("y_right_x")
    if not y_left and not y_right:
        return ""

    # 收集所有折线：polylines + paths（paths 的 M/L 序列）
    plines: list[list[tuple[float, float]]] = []
    plines.extend(geom.get("polylines") or [])
    plines.extend(geom.get("paths") or [])

    # 过滤：至少含 2 个点 + 跨度占绘图区 x 范围的较大比例
    plot_box = axes.get("plot_box")
    if plot_box:
        x_range = plot_box[1] - plot_box[0]
    else:
        x_range = max(p[0] for p in [(c[0], c[1]) for c in cats]) - min(c[0] for c in cats)

    filtered_lines: list[list[tuple[float, float]]] = []
    for pts in plines:
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        span = max(xs) - min(xs)
        if span < x_range * 0.4:
            continue
        filtered_lines.append(pts)

    if not filtered_lines:
        return ""

    cat_xs = [c[0] for c in cats]
    if len(cat_xs) >= 2:
        mean_gap = (cat_xs[-1] - cat_xs[0]) / (len(cat_xs) - 1)
    else:
        mean_gap = 50.0

    # 每条折线 → 每个类别一个值
    series_values: list[list[float | None]] = []
    series_axis: list[str] = []  # "left"/"right"
    for pts in filtered_lines:
        # 按 x 排序后每类别取最近点
        pts_sorted = sorted(pts, key=lambda p: p[0])
        values_per_cat: list[float | None] = []
        # 判断归哪条 Y 轴：基于折线 y 范围与轴的覆盖区的重合度
        y_min_pts = min(p[1] for p in pts_sorted)
        y_max_pts = max(p[1] for p in pts_sorted)

        def _axis_fit(ax_pts):
            if not ax_pts or len(ax_pts) < 2:
                return -1.0
            y_min_a = min(p[0] for p in ax_pts)
            y_max_a = max(p[0] for p in ax_pts)
            # 重合度：折线 y 范围相对于轴 y 范围的占比（要求在轴覆盖内）
            ov = max(0.0, min(y_max_pts, y_max_a) - max(y_min_pts, y_min_a))
            cov = ov / max(1.0, y_max_a - y_min_a)
            # 加上对轴覆盖折线的奖励
            contain = 1.0 if (y_min_pts >= y_min_a - 2 and y_max_pts <= y_max_a + 2) else 0.3
            return cov * contain

        fit_l = _axis_fit(y_left)
        fit_r = _axis_fit(y_right)
        if fit_l < 0 and fit_r < 0:
            continue
        if fit_r > fit_l + 0.05:
            axis_pts, _ = _svg_pick_axis_for_x(0, y_left, y_left_x, y_right, y_right_x, prefer="right")
        else:
            axis_pts, _ = _svg_pick_axis_for_x(0, y_left, y_left_x, y_right, y_right_x, prefer="left")
        interp = _svg_make_y_interpolator(axis_pts) if axis_pts else None
        if interp is None:
            continue
        for cx in cat_xs:
            # 找最近点
            nearest = min(pts_sorted, key=lambda p: abs(p[0] - cx))
            if abs(nearest[0] - cx) > mean_gap * 0.6:
                values_per_cat.append(None)
            else:
                values_per_cat.append(interp(nearest[1]))
        series_values.append(values_per_cat)
        series_axis.append("left" if axis_pts is y_left else "right")

    if not series_values:
        return ""

    # 百分号：按每条折线归属的轴判断
    xs_all = [n[0] for n in nodes]
    x_span = (max(xs_all) - min(xs_all)) if xs_all else 0.0
    tol_x = max(x_span * 0.04, 2.0) if x_span > 0 else 2.0
    has_pct_left = _svg_axis_has_pct(y_left, nodes, y_left_x, tol_x)
    has_pct_right = _svg_axis_has_pct(y_right, nodes, y_right_x, tol_x)
    left_vals = [v for _, v in (y_left or [])]
    right_vals = [v for _, v in (y_right or [])]
    dec_l, _ = _svg_infer_decimals(left_vals, has_pct_left)
    dec_r, _ = _svg_infer_decimals(right_vals, has_pct_right)

    n_series = len(series_values)
    lines: list[str] = []
    header = [""] + [""] * n_series
    lines.append(" \\t ".join(header))

    for i, cat in enumerate(cats):
        cat_label = _clean_label(cat[2])
        cells = [cat_label]
        for s_idx in range(n_series):
            v = series_values[s_idx][i]
            if v is None:
                cells.append("")
                continue
            use_left = series_axis[s_idx] == "left"
            dec = dec_l if use_left else dec_r
            has_pct_here = has_pct_left if use_left else has_pct_right
            cells.append(_svg_format_number(v, has_pct=has_pct_here, decimals_hint=dec))
        lines.append(" \\t ".join(cells))

    return " \\n ".join(lines)


def _svg_to_csv_box(
    nodes: list[tuple[float, float, str]],
    geom: dict,
    axes: dict,
) -> str:
    """箱线图：每个类别一个 <rect>（箱体上下沿=Q3/Q1），whisker=min/max，中位数 line 在箱体内。

    输出 5 列：最小值, Q1, 中位数, Q3, 最大值
    """
    cats = axes.get("cats") or []
    rects = geom.get("rects") or []
    lines_geo = geom.get("lines") or []
    if not cats or not rects:
        return ""
    y_left = axes.get("y_left")
    y_right = axes.get("y_right")
    if not y_left and not y_right:
        return ""

    # 选一条 Y 轴（优先左轴）
    axis_pts = y_left or y_right
    interp = _svg_make_y_interpolator(axis_pts)
    if interp is None:
        return ""
    axis_vals = [v for _, v in axis_pts]
    has_pct = _svg_has_pct_in_ref(nodes)
    dec, _ = _svg_infer_decimals(axis_vals, has_pct)

    assign = _svg_assign_rects_to_categories(rects, cats, axes.get("plot_box"))

    # 每类别中选一个最合理的箱 rect：高度占 Y 范围最大的那个
    box_per_cat: dict[int, tuple[float, float, float, float]] = {}
    for i, rs in assign.items():
        if not rs:
            continue
        box_per_cat[i] = max(rs, key=lambda r: r[3])

    if not box_per_cat:
        return ""

    cat_xs = [c[0] for c in cats]
    if len(cat_xs) >= 2:
        mean_gap = (cat_xs[-1] - cat_xs[0]) / (len(cat_xs) - 1)
    else:
        mean_gap = 50.0

    # 收集 whisker / 中位数：对每类搜集竖直 line（x1≈x2、跨度大）和水平 line（y1≈y2、穿过箱体）
    lines_out: list[str] = []
    header = [""] + [""] * 5
    lines_out.append(" \\t ".join(header))

    for i, cat in enumerate(cats):
        cat_label = _clean_label(cat[2])
        cells = [cat_label]
        if i not in box_per_cat:
            cells += [""] * 5
            lines_out.append(" \\t ".join(cells))
            continue
        rx, ry, rw, rh = box_per_cat[i]
        box_cx = rx + rw / 2
        q3_px = ry
        q1_px = ry + rh

        # 竖直 whisker 线：x1≈x2≈box_cx 且 |y1-y2| 较大
        vertical_lines = []
        horiz_lines_in_box = []
        for x1, y1, x2, y2 in lines_geo:
            if abs(x1 - x2) < 3 and abs(x1 - box_cx) < mean_gap * 0.3:
                vertical_lines.append((min(y1, y2), max(y1, y2)))
            elif abs(y1 - y2) < 3:
                # 水平线：x 范围与箱体水平重合
                xa, xb = min(x1, x2), max(x1, x2)
                if xa <= box_cx <= xb and abs(xb - xa) >= rw * 0.5:
                    horiz_lines_in_box.append((y1, abs(xb - xa)))

        # Whisker 上下端
        whisker_top_px = None
        whisker_bot_px = None
        if vertical_lines:
            # 找一条覆盖箱体的竖直线：min_y < q3_px, max_y > q1_px
            candidates = [
                v for v in vertical_lines if v[0] < q3_px + 5 and v[1] > q1_px - 5 and (v[1] - v[0]) >= rh * 0.8
            ]
            if candidates:
                c = max(candidates, key=lambda v: v[1] - v[0])
                whisker_top_px = c[0]
                whisker_bot_px = c[1]

        # 中位数：箱体内的水平线（q3_px < y < q1_px），宽度 ≈ 箱体宽
        median_px = None
        in_box_hs = [(y, w) for (y, w) in horiz_lines_in_box if q3_px - 1 < y < q1_px + 1]
        if in_box_hs:
            in_box_hs.sort(key=lambda p: abs(p[1] - rw))
            median_px = in_box_hs[0][0]

        # 兜底：没找到中位数 → 取箱体中心
        if median_px is None:
            median_px = ry + rh / 2
        if whisker_top_px is None:
            whisker_top_px = q3_px
        if whisker_bot_px is None:
            whisker_bot_px = q1_px

        # 数值转换（SVG y 向下大，Y_value 上大下小）
        v_min = interp(whisker_bot_px)
        v_q1 = interp(q1_px)
        v_med = interp(median_px)
        v_q3 = interp(q3_px)
        v_max = interp(whisker_top_px)

        for v in (v_min, v_q1, v_med, v_q3, v_max):
            cells.append(_svg_format_number(v, has_pct=has_pct, decimals_hint=dec))
        lines_out.append(" \\t ".join(cells))

    return " \\n ".join(lines_out)


def _svg_to_csv_combo(
    nodes: list[tuple[float, float, str]],
    geom: dict,
    axes: dict,
) -> str:
    """组合图：柱 + 折线；先组 bar 列，再把 line 列追加。"""
    cats = axes.get("cats") or []
    rects = geom.get("rects") or []
    plines = (geom.get("polylines") or []) + (geom.get("paths") or [])
    if not cats:
        return ""
    y_left = axes.get("y_left")
    y_right = axes.get("y_right")
    y_left_x = axes.get("y_left_x")
    y_right_x = axes.get("y_right_x")
    if not y_left and not y_right:
        return ""

    # 复用 bar 逻辑得到 bar 部分 csv（仅当有有效 rect 才启用）
    bar_rows: list[list[str]] = []
    bar_cols = 0
    if rects:
        assign = _svg_assign_rects_to_categories(rects, cats, axes.get("plot_box"))
        counts = [len(assign[i]) for i in assign]
        if counts and max(counts) > 0:
            from collections import Counter

            bar_cols = Counter(counts).most_common(1)[0][0] or max(counts)

    has_pct_left_flag = False
    has_pct_right_flag = False
    xs_all = [n[0] for n in nodes]
    x_span_local = (max(xs_all) - min(xs_all)) if xs_all else 0.0
    tol_x_local = max(x_span_local * 0.04, 2.0) if x_span_local > 0 else 2.0
    has_pct_left_flag = _svg_axis_has_pct(y_left, nodes, y_left_x, tol_x_local)
    has_pct_right_flag = _svg_axis_has_pct(y_right, nodes, y_right_x, tol_x_local)
    left_vals = [v for _, v in (y_left or [])]
    right_vals = [v for _, v in (y_right or [])]
    dec_l, _ = _svg_infer_decimals(left_vals, has_pct_left_flag)
    dec_r, _ = _svg_infer_decimals(right_vals, has_pct_right_flag)

    # 过滤折线
    if axes.get("plot_box"):
        pb = axes["plot_box"]
        x_range = pb[1] - pb[0]
    else:
        x_range = max(c[0] for c in cats) - min(c[0] for c in cats)
    filtered_lines = []
    for pts in plines:
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        if max(xs) - min(xs) < x_range * 0.4:
            continue
        filtered_lines.append(pts)

    line_cols = len(filtered_lines)
    total_cols = bar_cols + line_cols
    if total_cols == 0:
        return ""

    cat_xs = [c[0] for c in cats]
    if len(cat_xs) >= 2:
        mean_gap = (cat_xs[-1] - cat_xs[0]) / (len(cat_xs) - 1)
    else:
        mean_gap = 50.0

    rows: list[list[str]] = []
    header = [""] + [""] * total_cols
    rows.append(header)

    # 对每个类别生成：bar 部分 + line 部分
    assign = _svg_assign_rects_to_categories(rects, cats, axes.get("plot_box")) if rects else {}
    for i, cat in enumerate(cats):
        row = [_clean_label(cat[2])]
        # bar 列（统一用左轴）
        rs = assign.get(i, [])
        for k in range(bar_cols):
            if k < len(rs):
                rx, ry, rw, rh = rs[k]
                cx = rx + rw / 2
                axis_pts, _ = _svg_pick_axis_for_x(cx, y_left, y_left_x, y_right, y_right_x, prefer="left")
                interp = _svg_make_y_interpolator(axis_pts) if axis_pts else None
                if interp is None:
                    row.append("")
                else:
                    v_top = interp(ry)
                    v_bot = interp(ry + rh)
                    val = abs(v_top - v_bot) or v_top
                    use_left = axis_pts is y_left
                    dec = dec_l if use_left else dec_r
                    has_pct_here = has_pct_left_flag if use_left else has_pct_right_flag
                    row.append(_svg_format_number(val, has_pct=has_pct_here, decimals_hint=dec))
            else:
                row.append("")

        # line 列（双轴时归右轴，单轴则用唯一轴）
        line_prefer = "right" if (y_left and y_right) else "left"
        for pts in filtered_lines:
            pts_sorted = sorted(pts, key=lambda p: p[0])
            cx = cat_xs[i]
            nearest = min(pts_sorted, key=lambda p: abs(p[0] - cx))
            if abs(nearest[0] - cx) > mean_gap * 0.6:
                row.append("")
                continue
            axis_pts, _ = _svg_pick_axis_for_x(0, y_left, y_left_x, y_right, y_right_x, prefer=line_prefer)
            interp = _svg_make_y_interpolator(axis_pts) if axis_pts else None
            if interp is None:
                row.append("")
            else:
                v = interp(nearest[1])
                use_left = axis_pts is y_left
                dec = dec_l if use_left else dec_r
                has_pct_here = has_pct_left_flag if use_left else has_pct_right_flag
                row.append(_svg_format_number(v, has_pct=has_pct_here, decimals_hint=dec))

        rows.append(row)

    return " \\n ".join(" \\t ".join(r) for r in rows)


def _svg_to_csv_radar(
    nodes: list[tuple[float, float, str]],
    geom: dict,
    axes: dict,
) -> str:
    """雷达图：每条多边形的顶点按 (cx,cy) 角度排序对应 X 类别；半径→值。"""
    cats = axes.get("cats") or []
    polygons = geom.get("polygons") or []
    if not cats or not polygons:
        return ""
    y_left = axes.get("y_left")
    y_right = axes.get("y_right")
    axis_pts = y_left or y_right
    interp = _svg_make_y_interpolator(axis_pts) if axis_pts else None
    if interp is None:
        return ""

    # 过滤标记箭头（小多边形）
    filtered = [poly for poly in polygons if len(poly) >= len(cats) - 1]
    if not filtered:
        return ""

    # 雷达中心 = 所有 polygon 顶点的平均
    all_pts = [p for poly in filtered for p in poly]
    cx = sum(p[0] for p in all_pts) / len(all_pts)
    cy = sum(p[1] for p in all_pts) / len(all_pts)

    # 每个类别（label）的角度（根据 label 位置相对中心计算）
    import math

    cat_angles = []
    for x, y, _ in cats:
        ang = math.atan2(y - cy, x - cx)
        cat_angles.append(ang)

    def _closest_cat_idx(angle: float) -> int:
        return min(
            range(len(cat_angles)),
            key=lambda i: min(
                abs(cat_angles[i] - angle),
                abs(cat_angles[i] - angle + 2 * math.pi),
                abs(cat_angles[i] - angle - 2 * math.pi),
            ),
        )

    has_pct = _svg_has_pct_in_ref(nodes)
    axis_vals = [v for _, v in axis_pts]
    dec, _ = _svg_infer_decimals(axis_vals, has_pct)
    v_center = interp(cy)

    # 每条 polygon → 每个类别的值
    series_values: list[list[float | None]] = []
    for poly in filtered:
        per_cat: list[float | None] = [None] * len(cats)
        for px, py in poly:
            ang = math.atan2(py - cy, px - cx)
            i = _closest_cat_idx(ang)
            # 半径 → 数值：数值 = |py 对应轴值 - 中心轴值|（近似）
            v_py = interp(py)
            per_cat[i] = abs(v_py - v_center)
        series_values.append(per_cat)

    n_series = len(series_values)
    lines_out: list[str] = []
    lines_out.append(" \\t ".join([""] + [""] * n_series))
    for i, cat in enumerate(cats):
        row = [_clean_label(cat[2])]
        for s in range(n_series):
            v = series_values[s][i]
            if v is None:
                row.append("")
            else:
                row.append(_svg_format_number(v, has_pct=has_pct, decimals_hint=dec))
        lines_out.append(" \\t ".join(row))
    return " \\n ".join(lines_out)


def _svg_to_csv_pie(nodes: list[tuple[float, float, str]]) -> str:
    """饼图：延用原 <text>-only 方案（百分比配 label）。

    改进：对 label 文本做 **多行合并** ——y 接近且 x 水平相邻的多段 label
    合并为一个完整标签，解决一些 SVG 中 label 被拆行展示的情况（如
    "Communication" / "with oversea" / "clients" 三段）。
    """
    # 百分比节点
    pct_nodes = [(x, y, t) for x, y, t in nodes if "%" in t and _svg_text_looks_like_number(t)]
    label_nodes_raw = [(x, y, t) for x, y, t in nodes if not _svg_text_looks_like_number(t)]
    if not pct_nodes or not label_nodes_raw:
        return ""

    # 合并：按 y 聚类，每组内按 x 排序，相邻 x 间距 < 阈值则合并
    ys = [y for _, y, _ in label_nodes_raw]
    y_span = (max(ys) - min(ys)) if ys else 0.0
    tol_y = max(y_span * 0.02, 4.0)
    xs = [x for x, _, _ in label_nodes_raw]
    x_span = (max(xs) - min(xs)) if xs else 0.0

    # 对每个 label 先按 y 升序排
    by_y: list[list[tuple[float, float, str]]] = []
    sorted_labels = sorted(label_nodes_raw, key=lambda p: p[1])
    cur: list[tuple[float, float, str]] = []
    last_y: float | None = None
    for lab in sorted_labels:
        if last_y is None or abs(lab[1] - last_y) <= tol_y:
            cur.append(lab)
        else:
            by_y.append(cur)
            cur = [lab]
        last_y = lab[1]
    if cur:
        by_y.append(cur)

    # 合并：按 x 排序，相邻 x 间距 ≤ 合并阈值视为同一 label
    merge_tol_x = max(x_span * 0.10, 30.0)
    label_merged: list[tuple[float, float, str]] = []
    for group in by_y:
        group_sorted = sorted(group, key=lambda p: p[0])
        buf: list[tuple[float, float, str]] = []
        for lab in group_sorted:
            if not buf or lab[0] - buf[-1][0] <= merge_tol_x + len(buf[-1][2]) * 8:
                buf.append(lab)
            else:
                # flush
                x_avg = sum(p[0] for p in buf) / len(buf)
                y_avg = sum(p[1] for p in buf) / len(buf)
                combined = " ".join(p[2] for p in buf).strip()
                label_merged.append((x_avg, y_avg, combined))
                buf = [lab]
        if buf:
            x_avg = sum(p[0] for p in buf) / len(buf)
            y_avg = sum(p[1] for p in buf) / len(buf)
            combined = " ".join(p[2] for p in buf).strip()
            label_merged.append((x_avg, y_avg, combined))

    # 第二轮合并：同一 label 在 SVG 中可能跨 y（"Communication" / "with oversea" / "clients"）
    # 按 x 聚类，相邻 x 小于 merge_tol_x 视为同一 label 簇，按 y 合并
    label_final: list[tuple[float, float, str]] = []
    used = set()
    for i, (lx, ly, lt) in enumerate(label_merged):
        if i in used:
            continue
        bucket = [(lx, ly, lt)]
        used.add(i)
        for j, (ox, oy, ot) in enumerate(label_merged):
            if j in used or j == i:
                continue
            if abs(ox - lx) <= merge_tol_x * 0.8 and abs(oy - ly) <= max(y_span * 0.08, 40.0):
                bucket.append((ox, oy, ot))
                used.add(j)
        # 按 y 合并
        bucket.sort(key=lambda p: p[1])
        x_avg = sum(p[0] for p in bucket) / len(bucket)
        y_avg = sum(p[1] for p in bucket) / len(bucket)
        combined = " ".join(p[2] for p in bucket).strip()
        label_final.append((x_avg, y_avg, combined))

    if not label_final:
        label_final = label_merged

    rows: list[tuple[str, str]] = []
    for nx, ny, nt in pct_nodes:
        nearest = min(label_final, key=lambda p: (p[0] - nx) ** 2 + (p[1] - ny) ** 2)
        rows.append((_clean_label(nearest[2]), nt.strip()))
    lines = [" \\t ".join(["", ""])]
    for lbl, val in rows:
        lines.append(" \\t ".join([lbl, val]))
    return " \\n ".join(lines)


def _svg_to_csv_text_only(
    nodes: list[tuple[float, float, str]],
    raw_nodes: list[tuple[float, float, str]],
) -> str:
    """原始 <text>-only 方案（保留作兜底）。等价于旧版 svg_to_internal_csv 的核心逻辑。"""
    if not nodes:
        return ""
    numeric_nodes = [(x, y, t) for x, y, t in nodes if _svg_text_looks_like_number(t)]
    label_nodes = [(x, y, t) for x, y, t in nodes if not _svg_text_looks_like_number(t)]
    if not numeric_nodes:
        return ""
    xs = [x for x, _, _ in nodes]
    ys = [y for _, y, _ in nodes]
    x_span = (max(xs) - min(xs)) if xs else 0.0
    y_span = (max(ys) - min(ys)) if ys else 0.0
    tol_y = max(y_span * 0.03, 3.0) if y_span > 0 else 3.0
    tol_x = max(x_span * 0.04, 2.0) if x_span > 0 else 2.0
    y_clusters = _cluster_1d(ys, tol_y)

    def _row_score(idxs):
        if len(idxs) < 2:
            return -1.0
        row_xs = sorted(nodes[i][0] for i in idxs)
        spread = row_xs[-1] - row_xs[0]
        if len(row_xs) >= 2:
            gaps = [row_xs[k + 1] - row_xs[k] for k in range(len(row_xs) - 1)]
            mean_gap = sum(gaps) / len(gaps)
            var_gap = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps) if gaps else 0.0
            uniform = 1.0 / (1.0 + var_gap / (mean_gap * mean_gap + 1e-6))
        else:
            uniform = 0.0
        return len(idxs) * 2.0 + (spread / max(x_span, 1.0)) * 3.0 + uniform * 2.0

    best = None
    best_s = -1.0
    for idxs in y_clusters:
        s = _row_score(idxs)
        if s > best_s:
            best_s, best = s, idxs
    if not best or len(best) < 2:
        return _svg_fallback_single_column(numeric_nodes, label_nodes)
    categories = sorted([nodes[i] for i in best], key=lambda n: n[0])
    cat_xs = [c[0] for c in categories]
    used = set(best)
    col_data = []
    for cx in cat_xs:
        col = []
        for i, (nx, ny, nt) in enumerate(nodes):
            if i in used:
                continue
            if abs(nx - cx) <= tol_x:
                col.append((ny, nt, _svg_text_looks_like_number(nt)))
        col.sort(key=lambda p: p[0])
        col_data.append(col)
    from collections import Counter as _C

    num_counts = [sum(1 for _, _, isn in col if isn) for col in col_data]
    if not num_counts or max(num_counts) == 0:
        return _svg_fallback_single_column(numeric_nodes, label_nodes)
    mode_cnt = _C(num_counts).most_common(1)[0][0]
    target = mode_cnt or max(num_counts)
    lines = [" \\t ".join([""] + [""] * target)]
    for cat, col in zip(categories, col_data):
        cells = [_clean_label(cat[2])]
        nums_in_col = [t for _, t, isn in col if isn]
        for k in range(target):
            cells.append(nums_in_col[k].strip() if k < len(nums_in_col) else "")
        lines.append(" \\t ".join(cells))
    return " \\n ".join(lines)


def _svg_guess_chart_type(
    nodes: list[tuple[float, float, str]],
    geom: dict,
    axes: dict,
) -> str:
    """粗略判断 SVG 图表类型（当外部没有明确 chart_type 时使用）。

    返回 "pie" / "radar" / "box" / "bar" / "line" / "combo" / "unknown"。
    """
    has_pct = any("%" in t for _, _, t in nodes if _svg_text_looks_like_number(t))
    n_num_texts = sum(1 for _, _, t in nodes if _svg_text_looks_like_number(t))
    total_texts = len(nodes)
    # 饼图：百分比占数值类文本多数 + 没有明显 Y 轴刻度
    if (
        has_pct
        and total_texts > 0
        and n_num_texts / max(1, total_texts) >= 0.3
        and not (axes.get("y_left") or axes.get("y_right"))
    ):
        return "pie"
    # 雷达图：有 polygon 且多个 circle（同心圆）
    polys = geom.get("polygons") or []
    circles = geom.get("circles") or []
    if polys and len(circles) >= 3:
        return "radar"
    rects = geom.get("rects") or []
    plines = (geom.get("polylines") or []) + (geom.get("paths") or [])
    cats = axes.get("cats") or []
    # 箱线图：rects 数 ≈ cats 数 + 大量竖直 line
    if cats and rects:
        in_plot = [r for r in rects if _svg_rect_is_in_plot(r, axes.get("plot_box"))]
        if len(in_plot) <= len(cats) * 1.5 and len(in_plot) >= max(2, len(cats) - 2):
            lines_geo = geom.get("lines") or []
            vlines = [ln for ln in lines_geo if abs(ln[0] - ln[2]) < 3 and abs(ln[1] - ln[3]) > 10]
            if len(vlines) >= len(cats):
                return "box"
    # 组合：同时有 rects + polylines
    if rects and plines:
        return "combo"
    if rects:
        return "bar"
    if plines:
        return "line"
    return "unknown"


def svg_to_internal_csv(text: str) -> str:
    """从 SVG 代码中提取图表数据，输出内部 CSV (``\\t`` 列、``\\n`` 行) 格式。

    新版核心策略（geometry-aware）：
      1) 抽取 ``<text>`` / ``<tspan>`` 的 (x, y, text) 和几何图元（rect/line/
         polyline/polygon/circle/path）；
      2) 识别 **X 轴类别行** 与 **左/右 Y 轴刻度**，构造 ``y_pixel → y_value`` 的
         线性插值；
      3) 依据 chart_type（或启发式推断）路由到对应解析器：
         * **饼图**：百分比 ``<text>`` 与最近 label 配对；
         * **柱状图/组合图**：``<rect>`` 柱顶 y 经 Y 插值反算数值；
         * **折线图/组合图**：``<polyline>`` / ``<path>`` 顶点按类别 x 近邻取值；
         * **箱线图**：每类 rect 上下沿=Q3/Q1、whisker=min/max、内部水平 line=中位数；
         * **雷达图**：``<polygon>`` 顶点按角度归类、半径经 Y 插值反算；
      4) 以上均失败时，回退到旧版 **``<text>``-only** 方案（按坐标聚类）。

    Args:
        text: 模型输出的 SVG（可能含 ``xml`` 代码围栏）

    Returns:
        内部 CSV 字符串；完全无文本则返回空串。
    """
    svg_src = _extract_svg_snippet(text)
    if not svg_src or "<" not in svg_src:
        return ""

    # 先把嵌套的 <g transform="translate/scale/matrix ..."> 展平为全局坐标，
    # 否则模型在局部坐标系里写的 <text y="-230">... 会被下游解析器误判。
    svg_src = _svg_flatten_transforms(svg_src)

    raw_nodes = _parse_svg_texts(svg_src)
    if not raw_nodes:
        return ""

    # 过滤：去重 + 去除纯符号
    seen_texts: set = set()
    nodes: list[tuple[float, float, str]] = []
    for x, y, t in raw_nodes:
        tt = t.strip()
        if not tt:
            continue
        if _SVG_LABEL_DROP_PATTERNS[0].match(tt):
            continue
        key = (round(x, 2), round(y, 2), tt)
        if key in seen_texts:
            continue
        seen_texts.add(key)
        nodes.append((x, y, tt))

    if not nodes:
        return ""

    # 解析几何图元
    geom = _svg_parse_geometry(svg_src)

    # 识别坐标轴
    xs_all = [x for x, _, _ in nodes]
    ys_all = [y for _, y, _ in nodes]
    x_span = (max(xs_all) - min(xs_all)) if xs_all else 0.0
    y_span = (max(ys_all) - min(ys_all)) if ys_all else 0.0
    tol_y = max(y_span * 0.03, 3.0) if y_span > 0 else 3.0
    tol_x = max(x_span * 0.04, 2.0) if x_span > 0 else 2.0
    axes = _svg_detect_xy_axes(nodes, tol_x, tol_y)

    # 由评分上下文取到 chart_type（若有）
    ct_hint = ""
    try:
        ct_hint = get_chart_type() or ""
    except Exception:
        ct_hint = ""

    # 中/英关键词 → 解析器名
    ct_lower = ct_hint.lower()
    if "饼" in ct_hint or "pie" in ct_lower:
        ct_kind = "pie"
    elif "雷达" in ct_hint or "radar" in ct_lower:
        ct_kind = "radar"
    elif "箱" in ct_hint or "box" in ct_lower or "whisker" in ct_lower:
        ct_kind = "box"
    elif "柱" in ct_hint or "bar" in ct_lower or "column" in ct_lower or "直方" in ct_hint:
        ct_kind = "bar"
    elif "折线" in ct_hint or "line" in ct_lower:
        ct_kind = "line"
    elif "组合" in ct_hint or "combo" in ct_lower or "双轴" in ct_hint or "mixed" in ct_lower:
        ct_kind = "combo"
    else:
        ct_kind = _svg_guess_chart_type(nodes, geom, axes)

    # 【关键】text-first 优先：若图里数值 <text> 丰富，直接用 text 节点重建 CSV，
    # 不走几何解析器（后者容易对 transform、数据标签错位等情况误判）。
    # 判定阈值：数值 text 数量 >= 2（饼图/雷达图）或 >= 4（其他）。
    num_text_count = sum(1 for _, _, t in nodes if _svg_text_looks_like_number(t))
    # 饼图/雷达图常把 "label+num" 合并到一个 text，num_text_count 可能偏低；
    # 因此饼图/雷达图只要总 text 数 >= 4 就尝试 text_first。
    tf_threshold = 2 if ct_kind in ("pie", "radar") else 4
    if num_text_count >= tf_threshold or (ct_kind in ("pie", "radar") and len(nodes) >= 4):
        tf = _svg_text_first_parse(nodes, axes, ct_kind)
        if tf and tf.strip():
            return tf

    # 路由（几何解析器）
    csv_out = ""
    try:
        if ct_kind == "pie":
            csv_out = _svg_to_csv_pie(nodes)
        elif ct_kind == "radar":
            csv_out = _svg_to_csv_radar(nodes, geom, axes)
        elif ct_kind == "box":
            csv_out = _svg_to_csv_box(nodes, geom, axes)
        elif ct_kind == "bar":
            csv_out = _svg_to_csv_bar(nodes, geom, axes)
        elif ct_kind == "line":
            csv_out = _svg_to_csv_line(nodes, geom, axes)
        elif ct_kind == "combo":
            csv_out = _svg_to_csv_combo(nodes, geom, axes)
    except Exception:
        csv_out = ""

    if csv_out and csv_out.strip():
        return csv_out

    # 回退 1: text-first 统一解析（若前面没触发，这里再试一次）
    tf = _svg_text_first_parse(nodes, axes, ct_kind)
    if tf and tf.strip():
        return tf

    # 回退 2: 文本驱动通用（按 x 邻近匹配类别，不区分 chart_type）
    td = _svg_to_csv_text_driven(nodes, axes)
    if td and td.strip():
        return td

    # 回退 3: text-only 方案
    return _svg_to_csv_text_only(nodes, raw_nodes)


# ---------------------------------------------------------------------------
# Text-first 统一解析器（transform 展平后，模型写出的数值 <text> 通常全对，
# 专用几何解析器反而容易出错。这里按 chart_type 用纯文本节点重建 CSV）
# ---------------------------------------------------------------------------


def _svg_merge_multiline_labels(
    nodes: list[tuple[float, float, str]],
    x_tol: float = 20.0,
    y_gap_max: float = 55.0,
) -> list[tuple[float, float, str]]:
    """把"近似同 x 且 y 紧邻"的多个 text 合并成一个（label 多行回归单行）。

    典型场景：
      * 饼图扇区 label：``Teachers'`` (570,330) + ``salaries`` (570,365) →
        ``"Teachers' salaries"`` ；
      * ``Normal`` (350,785) + ``(n=18)`` (350,815) → ``"Normal (n=18)"``；
      * tspan 拆词：``逻辑-`` + ``数学智能`` 共享坐标 → ``"逻辑-数学智能"``；
      * 饼图扇区值 text 紧跟在 label 块后面也会被合并（这种情况下后续的
        tail_num_re 依然能拆出数值，不影响正确性）。

    注意：**只合并相同 label 段**，不要把百分比/数字也合并进 label。
    为此：如果某个 text 是纯数字或纯百分比，**不参与合并**（保持独立）。

    Args:
      nodes: 原始 text 节点 ``[(x, y, text), ...]``
      x_tol: 同列 x 容差（像素）
      y_gap_max: 两行 y 间距上限（像素），超过则认为不属于同一 label 块

    Returns:
      合并后的 nodes（顺序：保留合并后块的第一个 text 的插入顺序）。
    """
    if not nodes:
        return []

    pure_num_re = re.compile(r"^\s*[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?\s*%?\s*$")
    money_re = re.compile(r"^\s*[$￥¥€£]\s*[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?\s*$")

    def _is_numeric_like(t: str) -> bool:
        tt = t.strip()
        return bool(pure_num_re.match(tt) or money_re.match(tt))

    # 把 nodes 按 x 分桶（桶内按 y 升序）
    # 然后在每个桶内找连续 y 紧邻的非数字 text 合并
    sorted_nodes = sorted(enumerate(nodes), key=lambda p: (p[1][0], p[1][1]))

    # 桶划分：按 x
    buckets: list[list[tuple[int, tuple[float, float, str]]]] = []
    for idx, n in sorted_nodes:
        placed = False
        for b in buckets:
            ref_x = sum(p[1][0] for p in b) / len(b)
            if abs(ref_x - n[0]) <= x_tol:
                b.append((idx, n))
                placed = True
                break
        if not placed:
            buckets.append([(idx, n)])

    # 对每个桶按 y 排序，然后找"连续非数字 y 紧邻段"做合并
    merged_map: dict[int, tuple[float, float, str]] = {}  # orig_idx → new node
    keep_idx: set[int] = set()  # 保留的原始 idx（作为合并块代表）
    drop_idx: set[int] = set()  # 被合并掉的原始 idx

    # 同 y 容差：y2 必须严格大于 y1 + y_min_gap（避免相同 y 的 tspan 被乱序合并）
    y_min_gap = 8.0

    for b in buckets:
        b_sorted = sorted(b, key=lambda p: p[1][1])
        i = 0
        while i < len(b_sorted):
            idx_i, n_i = b_sorted[i]
            if _is_numeric_like(n_i[2]):
                # 数字独立保留
                i += 1
                continue
            # 从 i 开始，收集连续 y 紧邻的非数字 text（y 必须严格递增）
            group = [(idx_i, n_i)]
            j = i + 1
            while j < len(b_sorted):
                idx_j, n_j = b_sorted[j]
                if _is_numeric_like(n_j[2]):
                    break
                prev_y = group[-1][1][1]
                # y 间距：既要有最小 gap（排除同 y 乱序），也要不超过 y_gap_max
                dy = n_j[1] - prev_y
                if dy < y_min_gap or dy > y_gap_max:
                    break
                group.append((idx_j, n_j))
                j += 1
            if len(group) >= 2:
                # 合并
                texts = [_clean_label(g[1][2]) for g in group]
                # 保留原始分隔（tspan 常无空格）：去掉 "-" "—" 结尾时直接拼接
                merged_text_parts: list[str] = []
                for k, t in enumerate(texts):
                    if k == 0:
                        merged_text_parts.append(t)
                    else:
                        prev = merged_text_parts[-1]
                        # 如果前一个以 "-"/"—"/":"/" "结尾 或下一个以 "("开头，
                        # 拼接时不加空格；否则加空格
                        if prev.endswith(("-", "—", ":")) or t.startswith("'"):
                            merged_text_parts.append(t)
                        else:
                            merged_text_parts.append(" ")
                            merged_text_parts.append(t)
                merged_text = "".join(merged_text_parts).strip()
                # 代表节点用 group[0] 的坐标
                rep_idx = group[0][0]
                rep_x = group[0][1][0]
                rep_y = group[0][1][1]
                merged_map[rep_idx] = (rep_x, rep_y, merged_text)
                keep_idx.add(rep_idx)
                for g in group[1:]:
                    drop_idx.add(g[0])
            else:
                keep_idx.add(idx_i)
            i = j if len(group) >= 2 else i + 1

    # 构造返回：按原始插入顺序遍历，对 keep_idx 输出（数字节点也在其中）
    result: list[tuple[float, float, str]] = []
    for idx, n in enumerate(nodes):
        if idx in drop_idx:
            continue
        if idx in merged_map:
            result.append(merged_map[idx])
        else:
            result.append(n)
    return result


def _svg_find_bottom_axis_cats(
    nodes: list[tuple[float, float, str]],
    axes: dict,
) -> list[tuple[float, float, str]]:
    """重新识别 X 轴类别行，**允许纯数字类别**（如 Stage 0/1/2/3/4）。

    原 axes['cats'] 由 _svg_detect_xy_axes 产生，它倾向排除纯数字，
    但箱线图/柱状图的 X 类别经常就是数字（年份、编号）。这里做更宽松的识别：

    策略：
      1) 先对 nodes 做多行 label 合并（纯数字不参与合并），避免 ``Normal`` +
         ``(n=18)`` 被分开处理；
      2) 找所有 y 在图下部（y 排序后 70%~100%）的 text；
      3) 按 y 聚类，选节点数最多的一个 cluster；排除"同 x 垂直堆叠的图例"
         （cluster 的 x 跨度过小 / x 值过于密集在一列）；
      4) 过滤掉 Y 轴刻度列；
      5) 若 cluster 中至少 2 个节点、x 均匀分布，作为 X 类别返回；
      6) 否则回退到原 axes['cats']。
    """
    # 若原有 cats 可信（非纯数字占比高），直接用它
    orig_cats = axes.get("cats") or []
    if orig_cats:
        non_num = sum(1 for _, _, t in orig_cats if not _svg_text_looks_like_number(t))
        if non_num >= len(orig_cats) * 0.5 and len(orig_cats) >= 2:
            return list(orig_cats)

    if not nodes:
        return list(orig_cats)

    ys = [y for _, y, _ in nodes]
    if not ys:
        return list(orig_cats)
    y_min, y_max = min(ys), max(ys)
    y_span = y_max - y_min
    if y_span <= 0:
        return list(orig_cats)

    y_left_x = axes.get("y_left_x")
    y_right_x = axes.get("y_right_x")

    def _on_y_axis(x: float) -> bool:
        if y_left_x is not None and abs(x - y_left_x) <= 20.0:
            return True
        if y_right_x is not None and abs(x - y_right_x) <= 20.0:
            return True
        return False

    lower_thresh = y_min + 0.65 * y_span
    cand = [n for n in nodes if n[1] >= lower_thresh and not _on_y_axis(n[0])]
    if len(cand) < 2:
        return list(orig_cats)

    tol_y = max(y_span * 0.03, 3.0)
    cand_sorted = sorted(cand, key=lambda n: n[1])
    clusters: list[list[tuple[float, float, str]]] = []
    cur: list[tuple[float, float, str]] = []
    cur_y: float | None = None
    for n in cand_sorted:
        if cur_y is None or abs(n[1] - cur_y) <= tol_y:
            cur.append(n)
            cur_y = n[1] if cur_y is None else (cur_y + n[1]) / 2
        else:
            clusters.append(cur)
            cur = [n]
            cur_y = n[1]
    if cur:
        clusters.append(cur)

    clusters.sort(key=lambda c: -len(c))
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        xs = [n[0] for n in cluster]
        x_span_all = max(n[0] for n in nodes) - min(n[0] for n in nodes)
        spread = max(xs) - min(xs)
        if x_span_all > 0 and spread < x_span_all * 0.3:
            continue
        cluster_sorted = sorted(cluster, key=lambda n: n[0])
        return cluster_sorted

    return list(orig_cats)


def _svg_text_first_parse(
    nodes: list[tuple[float, float, str]],
    axes: dict,
    ct_kind: str,
) -> str:
    """Text-first 解析：只依赖 <text> 节点（不用 <rect>/<polygon> 等几何），
    针对不同 chart_type 走不同的组装策略。

    触发条件（由调用方把关）：数值 text 数量充足。

    ct_kind: "pie" | "radar" | "box" | "bar" | "line" | "combo"

    返回空串代表本路径无法解析，交回调用方走几何解析器。
    """
    try:
        if ct_kind == "pie":
            # 对饼图：先合并同 x 相邻的多行 text 块（如 "Teachers'" + "salaries" → "Teachers' salaries"）
            merged_nodes = _svg_merge_multiline_labels(nodes)
            return _svg_text_first_pie(merged_nodes)
        if ct_kind == "radar":
            # 雷达图不做 merge（tspan 同 y 情况特殊，合并易错）
            return _svg_text_first_radar(nodes)
        if ct_kind == "box":
            return _svg_text_first_box(nodes, axes)
        if ct_kind in ("bar", "line", "combo"):
            return _svg_text_first_xy(nodes, axes, ct_kind)
    except Exception:
        return ""
    return ""


def _svg_text_first_pie(nodes: list[tuple[float, float, str]]) -> str:
    """饼图的 text-first：识别扇区 label + 数值，支持 2 列（label+pct）或
    3 列（label+金额+pct）。

    策略：
      0) 先做多行 label 合并（如 ``Teachers'`` + ``salaries`` → ``Teachers' salaries``）；
      1) 按 text 内容分类：
         - 纯百分比 "N%"           → pct_nums
         - 纯金额 "$N" / "￥N" 等    → money_nums
         - 其他纯数字 "N"            → plain_nums
         - 合并 text "label N%" / "label" → labels 或 pairs_merged
      2) 若 pairs_merged >= 2（形如 "label N%"）且无额外金额 → 直接用 2 列；
      3) 若 money_nums 和 pct_nums 都 >= 2 → 尝试 3 列（label+金额+pct），
         label 从独立 labels 中按就近匹配；
      4) 否则按 pct_nums/plain_nums 与 labels 就近配对出 2 列；
      5) 都失败返回空串。

    另外：对 label 做"去尾部金额/百分比"预处理，避免把 "Housing, $16887" 当成
    一个整体 label。
    """
    # 多行 label 合并：把竖排的 Teachers'/salaries/... 拼回单行
    nodes = _svg_merge_multiline_labels(nodes)

    money_re = re.compile(r"^\s*[$￥¥€£]\s*[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?\s*$")
    # 纯百分比：支持尾部中文标点（如 "8.3%、" "9.1%。"）和括号包裹（如 "(20.41%)"）
    pct_re = re.compile(r"^\s*[（(]?\s*[-+]?\d+(?:\.\d+)?\s*%\s*[)）]?\s*[、。，,.]?\s*$")
    plain_num_re = re.compile(r"^\s*[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?\s*$")
    # 合并 "label ...%"：支持中文标点分隔（如 "583元、9.1%。"）
    tail_pct_re = re.compile(r"^\s*(.+?)[\s、，。：:,\-]*[（(]?\s*([-+]?\d+(?:\.\d+)?\s*%)\s*[)）]?\s*[、。，,.]?\s*$")
    tail_money_re = re.compile(
        r"^\s*(.+?)[\s、，。：:,\-]*([$￥¥€£]\s*[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?)\s*[、。，,.]?\s*$"
    )
    # 中文金额模式：如 "528元" "2084元"
    cn_money_re = re.compile(r"^\s*[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?\s*元\s*[、。，,.]?\s*$")
    # 纯金额+百分比（无label）：如 "583元、9.1%。" "79元、10.4%。"
    cn_money_pct_re = re.compile(
        r"^\s*([-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?\s*元)\s*[、，。\s]*([-+]?\d+(?:\.\d+)?\s*%)\s*[、。，,.]?\s*$"
    )
    # 合并 "label N元 N%"：如 "衣着453元 7.1%"（label 必须含非数字字符）
    tail_cn_money_pct_re = re.compile(
        r"^\s*([^\d].+?)\s*([-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?\s*元)\s*[、，。\s]*([-+]?\d+(?:\.\d+)?\s*%)\s*[、。，,.]?\s*$"
    )
    # 合并 "label N元"（无百分比）：如 "医疗保健528元"
    tail_cn_money_re = re.compile(r"^\s*([^\d].+?)\s*([-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?\s*元)\s*[、。，,.]?\s*$")

    pairs_merged: list[tuple[str, str]] = []
    pct_nums: list[tuple[float, float, str]] = []
    money_nums: list[tuple[float, float, str]] = []
    plain_nums: list[tuple[float, float, str]] = []
    labels: list[tuple[float, float, str]] = []

    for x, y, t in nodes:
        tt = t.strip()
        if not tt:
            continue
        # 纯百分比（支持中文标点后缀和括号包裹）
        if pct_re.match(tt):
            # 清理尾部标点和括号，只保留数字、小数点和%
            clean_pct = re.sub(r"[（()）、。，,\s]", "", tt)
            pct_nums.append((x, y, clean_pct))
            continue
        # 纯金额（西文货币符号）
        if money_re.match(tt):
            money_nums.append((x, y, tt))
            continue
        # 中文金额（如 "528元"）
        if cn_money_re.match(tt):
            clean_money = tt.rstrip("、。，,.")
            money_nums.append((x, y, clean_money))
            continue
        # 纯数字（非百分比非货币）
        if plain_num_re.match(tt):
            plain_nums.append((x, y, tt))
            continue
        # 纯金额+百分比（无label）：如 "583元、9.1%。"
        m = cn_money_pct_re.match(tt)
        if m:
            mny = m.group(1).strip()
            pct = m.group(2).strip()
            money_nums.append((x, y, mny))
            pct_nums.append((x, y, pct))
            continue
        # 合并 "label N元 N%"（如 "衣着453元 7.1%"）
        m = tail_cn_money_pct_re.match(tt)
        if m:
            label = m.group(1).strip(" 、，。：:,-")
            mny = m.group(2).strip()
            pct = m.group(3).strip()
            if label and not plain_num_re.match(label):
                pairs_merged.append((label, pct))
                money_nums.append((x, y, mny))
                continue
        # 合并 "label N元"（如 "医疗保健528元"）
        m = tail_cn_money_re.match(tt)
        if m:
            label = m.group(1).strip(" 、，。：:,-")
            mny = m.group(2).strip()
            if label and not plain_num_re.match(label):
                labels.append((x, y, label))
                money_nums.append((x, y, mny))
                continue
        # 合并 "label ...%"
        m = tail_pct_re.match(tt)
        if m:
            label = m.group(1).strip(" 、，。：:,-")
            pct = m.group(2).strip()
            if label and not plain_num_re.match(label):
                # label 再去掉尾部金额（"Housing, $16887"）
                m2 = tail_money_re.match(label)
                if m2:
                    label = m2.group(1).strip(" 、，。：:,-")
                pairs_merged.append((label, pct))
                continue
        # 合并 "label $N"（无百分号）
        m = tail_money_re.match(tt)
        if m:
            label = m.group(1).strip(" 、，。：:,-")
            mny = m.group(2).strip()
            if label and not plain_num_re.match(label):
                pairs_merged.append((label, mny))
                continue
        labels.append((x, y, tt))

    sep_col = " \\t "
    sep_row = " \\n "

    # ============ 3 列饼图（label + 金额 + 百分比）============
    # 触发：金额 ≥ 2 且百分比 ≥ 2
    if len(money_nums) >= 2 and len(pct_nums) >= 2:
        # 收集 label：合并 text 里去掉金额/百分比后的 label，以及独立 labels
        all_label_pool: list[tuple[float, float, str]] = list(labels)
        # 从 pairs_merged 还原出 label text（仅有它们的 label 时用 "合并 text" 的 x,y）
        # 但 pairs_merged 丢了坐标——这里退而求其次：对每个金额找最近的独立 label
        # 若独立 labels 不够多，则放弃 3 列模式
        if len(all_label_pool) < len(money_nums) - 1:
            # 回退到 2 列
            pass
        else:
            rows: list[tuple[str, str, str]] = []
            # 对每个金额 money，找最近 label 和最近 pct
            used_lbl: set[int] = set()
            used_pct: set[int] = set()
            for mx, my, mv in money_nums:
                # 最近 label
                best_l = -1
                best_d = float("inf")
                for i, (lx, ly, _) in enumerate(all_label_pool):
                    if i in used_lbl:
                        continue
                    d = (lx - mx) ** 2 + (ly - my) ** 2
                    if d < best_d:
                        best_d = d
                        best_l = i
                # 最近 pct
                best_p = -1
                best_pd = float("inf")
                for j, (px, py, _) in enumerate(pct_nums):
                    if j in used_pct:
                        continue
                    d = (px - mx) ** 2 + (py - my) ** 2
                    if d < best_pd:
                        best_pd = d
                        best_p = j
                if best_l < 0 or best_p < 0:
                    continue
                used_lbl.add(best_l)
                used_pct.add(best_p)
                lbl = _clean_label(all_label_pool[best_l][2])
                pct = pct_nums[best_p][2]
                rows.append((lbl, mv, pct))
            if len(rows) >= 2:
                lines = [sep_col.join(["", "", ""])]
                for lbl, mny, pct in rows:
                    lines.append(sep_col.join([lbl, mny, pct]))
                return sep_row.join(lines)

    # ============ 2 列饼图（合并 text 充足）============
    if len(pairs_merged) >= 2:
        lines = [sep_col.join(["", ""])]
        for lbl, val in pairs_merged:
            lines.append(sep_col.join([_clean_label(lbl), val]))
        return sep_row.join(lines)

    # ============ 2 列饼图（独立 label + 独立数值）============
    indep_nums = pct_nums + plain_nums + money_nums
    if indep_nums and labels:
        paired: list[tuple[str, str]] = []
        for nx, ny, nt in indep_nums:
            best_i = -1
            best_d = float("inf")
            for i, (lx, ly, _) in enumerate(labels):
                d = (lx - nx) ** 2 + (ly - ny) ** 2
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0:
                paired.append((_clean_label(labels[best_i][2]), nt))
        if len(paired) >= 2:
            lines = [sep_col.join(["", ""])]
            for lbl, val in paired:
                lines.append(sep_col.join([lbl, val]))
            return sep_row.join(lines)

    return ""


def _svg_radar_filter_axis_ticks(
    nums: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """过滤雷达图中的 Y 轴刻度数值。

    雷达图 SVG 中常有 Y 轴刻度（如 0, 0.2, 0.4, 0.6, 0.8, 1.0 或 0, 20, 40, 60, 80, 100），
    这些数值构成等差数列，不是真正的数据值。如果检测到这种模式，则将其过滤掉。

    判定条件：
      1) 数值按值排序后构成等差数列（容差 5%）
      2) 这些数值的 x 坐标或 y 坐标高度集中（在同一条线上），说明是轴刻度
    """
    if len(nums) < 3:
        return nums

    # 解析数值
    parsed: list[tuple[int, float]] = []
    for i, (x, y, t) in enumerate(nums):
        tt = t.strip().replace("%", "")
        try:
            v = float(tt)
            parsed.append((i, v))
        except ValueError:
            pass

    if len(parsed) < 3:
        return nums

    # 按值排序
    parsed_sorted = sorted(parsed, key=lambda p: p[1])
    values = [v for _, v in parsed_sorted]

    # 检查是否等差
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    if not diffs:
        return nums
    mean_diff = sum(diffs) / len(diffs)
    if mean_diff == 0:
        # 所有值相同 → 不是刻度
        return nums
    is_arithmetic = all(abs(d - mean_diff) <= abs(mean_diff) * 0.05 + 0.001 for d in diffs)

    if not is_arithmetic:
        return nums

    # 进一步检查：这些数值的坐标是否集中在一条线上（x 或 y 方向）
    tick_indices = {idx for idx, _ in parsed_sorted}
    tick_xs = [nums[i][0] for i in tick_indices]
    tick_ys = [nums[i][1] for i in tick_indices]

    x_range = max(tick_xs) - min(tick_xs) if tick_xs else 0
    y_range = max(tick_ys) - min(tick_ys) if tick_ys else 0

    # 如果 x 范围很小（同一列）或 y 范围很小（同一行），说明是轴刻度
    # 对于雷达图，刻度通常沿一条径向线排列
    all_xs = [x for x, _, _ in nums]
    all_ys = [y for _, y, _ in nums]
    total_x_range = max(all_xs) - min(all_xs) if all_xs else 1
    total_y_range = max(all_ys) - min(all_ys) if all_ys else 1

    # 刻度的 x 或 y 范围占总范围的比例 < 30%，认为是轴刻度
    x_concentrated = x_range < max(total_x_range * 0.3, 20) if total_x_range > 0 else True
    y_concentrated = y_range < max(total_y_range * 0.3, 20) if total_y_range > 0 else True

    if x_concentrated or y_concentrated:
        # 过滤掉这些刻度数值
        return [n for i, n in enumerate(nums) if i not in tick_indices]

    return nums


def _svg_text_first_radar(nodes: list[tuple[float, float, str]]) -> str:
    """雷达图的 text-first：处理"轴名 + 数值"合并 text 或分离 text。

    最常见：``<text>精神 6</text>`` 一条里既有类别又有数值。
    也处理：``<text>精神</text>`` 和 ``<text>6</text>`` 分别出现。

    策略：
      1) 先从每个 text 里尾部拆数字，成功的直接成对；
      2) 剩余纯数字 text 按 "到 label text 最近" 配对；
      3) 输出 2 列 CSV。
    """
    # 注意：使用 [：:\-] 作为分隔符（不含空格），避免把 "Indicator 1" 误拆为
    # label="Indicator" + value="1"。只有明确的分隔符（冒号、破折号）才触发拆分。
    # 对于 "精神 6" 这种中文+空格+数字的情况，用单独的正则处理。
    tail_num_re = re.compile(r"^\s*(.+?)[：:\-]+([-+]?\d+(?:\.\d+)?\s*%?)\s*$")
    # 中文/日文 label + (可选空格) + 数值（label 必须以 CJK 字符开头，
    # 且 label 最后一个字符必须是 CJK 字符，避免 "Indicator 1" 误匹配）
    # 支持 "精神 6"、"青节与重音68"、"吉林74%" 等格式
    tail_num_cjk_re = re.compile(
        r"^\s*([\u4e00-\u9fff\u3040-\u30ff][\u4e00-\u9fff\u3040-\u30ff\w\s]*[\u4e00-\u9fff\u3040-\u30ff])\s*([-+]?\d+(?:\.\d+)?\s*%?)\s*$"
    )
    pure_num_re = re.compile(r"^[-+]?\d+(?:\.\d+)?\s*%?$")

    pairs_merged: list[tuple[str, str]] = []
    independent_nums: list[tuple[float, float, str]] = []
    labels: list[tuple[float, float, str]] = []

    for x, y, t in nodes:
        tt = t.strip()
        if not tt:
            continue
        if pure_num_re.match(tt):
            independent_nums.append((x, y, tt))
            continue
        m = tail_num_re.match(tt)
        if m:
            label = m.group(1).strip(" ：:-")
            value = m.group(2).strip()
            if label and not pure_num_re.match(label) and len(label) <= 30:
                pairs_merged.append((label, value))
                continue
        # 中文 label + 空格 + 数值
        m = tail_num_cjk_re.match(tt)
        if m:
            label = m.group(1).strip()
            value = m.group(2).strip()
            if label and not pure_num_re.match(label) and len(label) <= 30:
                pairs_merged.append((label, value))
                continue
        # 只保留短 label（避免把标题、日期当类别）
        if len(tt) <= 20 and not _svg_text_looks_like_number(tt):
            labels.append((x, y, tt))

    if len(pairs_merged) >= 2:
        sep_col = " \\t "
        sep_row = " \\n "
        lines = [sep_col.join(["", ""])]
        for lbl, val in pairs_merged:
            lines.append(sep_col.join([_clean_label(lbl), val]))
        return sep_row.join(lines)

    # 分离情况：回到"每个数字找最近 label"的贪心匹配（允许多个数字归到同一 label，
    # 与原有语义一致）。这是为了避免"一对一"强制在 label/数字数量失配时出错。
    if independent_nums and labels:
        # 过滤 Y 轴刻度：如果数值构成等差数列（如 0, 0.2, 0.4, 0.6, 0.8, 1.0），
        # 则这些是 Y 轴刻度而非数据值，应排除。
        filtered_nums = _svg_radar_filter_axis_ticks(independent_nums)
        if filtered_nums and labels:
            paired: list[tuple[str, str]] = []
            for nx, ny, nt in filtered_nums:
                best_i = -1
                best_d = float("inf")
                for i, (lx, ly, _) in enumerate(labels):
                    d = (lx - nx) ** 2 + (ly - ny) ** 2
                    if d < best_d:
                        best_d = d
                        best_i = i
                if best_i >= 0:
                    paired.append((_clean_label(labels[best_i][2]), nt))
            if len(paired) >= 2:
                sep_col = " \\t "
                sep_row = " \\n "
                lines = [sep_col.join(["", ""])]
                for lbl, val in paired:
                    lines.append(sep_col.join([lbl, val]))
                return sep_row.join(lines)

    return ""


def _svg_text_first_box(nodes: list[tuple[float, float, str]], axes: dict) -> str:
    """箱线图的 text-first：每类箱子 5 个数值 → min/Q1/median/Q3/max。

    策略：
      1) 先对 nodes 做多行 label 合并（如 ``Normal`` + ``(n=18)`` →
         ``"Normal (n=18)"``；数字节点不参与合并，保持独立）；
      2) 用 _svg_find_bottom_axis_cats 重新找 X 轴类别（允许纯数字，
         如 Stage 0/1/2/3/4）；
      3) 剩余数字 text 按 x 就近归到类别；
      4) 每类把 5 个数字按值排序，作为 min/Q1/median/Q3/max。
    """
    # 合并多行 label（数字节点自动保留不合并）
    nodes = _svg_merge_multiline_labels(nodes)

    cats = _svg_find_bottom_axis_cats(nodes, axes)
    if len(cats) < 1:
        return ""
    y_left_x = axes.get("y_left_x")
    y_right_x = axes.get("y_right_x")

    cats_y_mean = sum(c[1] for c in cats) / len(cats)
    cat_xs = [c[0] for c in cats]
    cat_labels = [_clean_label(c[2]) for c in cats]
    cat_keys = {(round(c[0], 2), round(c[1], 2), c[2]) for c in cats}

    def _is_on_y_axis(x: float) -> bool:
        if y_left_x is not None and abs(x - y_left_x) <= 20.0:
            return True
        if y_right_x is not None and abs(x - y_right_x) <= 20.0:
            return True
        return False

    per_cat_vals: dict[int, list[float]] = {i: [] for i in range(len(cats))}
    for x, y, t in nodes:
        if (round(x, 2), round(y, 2), t) in cat_keys:
            continue
        if not _svg_text_looks_like_number(t):
            continue
        if _is_on_y_axis(x):
            continue
        if abs(y - cats_y_mean) < 10.0:
            continue
        v = _svg_parse_number(t)
        if v is None:
            continue
        # 归到最近的类别
        idx = min(range(len(cat_xs)), key=lambda i: abs(cat_xs[i] - x))
        # 距离过远不要——阈值放宽到 med_sp * 0.9（数据点常在箱体两侧 whisker 处，
        # 偏离类别中心 x 较多，但只要不跨越相邻类别即可）
        if len(cat_xs) >= 2:
            sps = sorted(abs(cat_xs[i] - cat_xs[i - 1]) for i in range(1, len(cat_xs)))
            med_sp = sps[len(sps) // 2] if sps else 0.0
            if med_sp > 0 and abs(cat_xs[idx] - x) > med_sp * 0.9:
                continue
        per_cat_vals[idx].append(v)

    # 要求每类恰好 5 个（> 5 则取 5 个最靠近 y 中值的；< 5 则跳过该类别后兜底）
    # 简化：只要 ≥ 3 个，就截断或按出现顺序填，按排序输出 min..max
    rows: list[list[str]] = []
    for i, lst in per_cat_vals.items():
        if len(lst) < 3:
            rows.append([cat_labels[i], "", "", "", "", ""])
            continue
        lst_sorted = sorted(lst)
        # 若 > 5 个，从中间保留 5 个（去掉异常）
        if len(lst_sorted) > 5:
            # 取均匀分布的 5 个：min, 25%, 50%, 75%, max
            n = len(lst_sorted)
            picks = [
                lst_sorted[0],
                lst_sorted[int(n * 0.25)],
                lst_sorted[int(n * 0.5)],
                lst_sorted[int(n * 0.75)],
                lst_sorted[-1],
            ]
            lst_sorted = picks
        # 不足 5 填空
        while len(lst_sorted) < 5:
            lst_sorted.append(None)
        rows.append([cat_labels[i]] + [_svg_format_number(v) if v is not None else "" for v in lst_sorted])

    # 若没有任何行有完整 5 值，放弃
    good_rows = sum(1 for r in rows if all(c.strip() for c in r))
    if good_rows < max(1, len(cats) // 2):
        return ""

    sep_col = " \\t "
    sep_row = " \\n "
    header = [""] * 6
    lines = [sep_col.join(header)]
    for r in rows:
        lines.append(sep_col.join(r))
    return sep_row.join(lines)


def _svg_text_first_xy(
    nodes: list[tuple[float, float, str]],
    axes: dict,
    ct_kind: str,
) -> str:
    """柱状/折线/组合图的 text-first：数据点 <text> 按 x 就近归到类别。

    策略：
      1) 用 _svg_find_bottom_axis_cats 重新找 X 轴类别（允许纯数字类别）；
      2) 剔除 Y 轴刻度（左右轴 x 附近）和类别行；
      3) 剩余数字 text 归到最近类别，同一类别内按 y 升序，作为多系列数值；
      4) 每类取 "众数长度" 列数。
    """
    cats = _svg_find_bottom_axis_cats(nodes, axes)
    if len(cats) < 2:
        return ""
    y_left_x = axes.get("y_left_x")
    y_right_x = axes.get("y_right_x")
    cats_y_mean = sum(c[1] for c in cats) / len(cats)
    cat_xs = [c[0] for c in cats]
    cat_labels = [_clean_label(c[2]) for c in cats]
    cat_keys = {(round(c[0], 2), round(c[1], 2), c[2]) for c in cats}

    def _is_on_y_axis(x: float) -> bool:
        if y_left_x is not None and abs(x - y_left_x) <= 20.0:
            return True
        if y_right_x is not None and abs(x - y_right_x) <= 20.0:
            return True
        return False

    # 归类
    per_cat: dict[int, list[tuple[float, float]]] = {i: [] for i in range(len(cats))}
    sps = sorted(abs(cat_xs[i] - cat_xs[i - 1]) for i in range(1, len(cat_xs))) if len(cat_xs) >= 2 else []
    med_sp = sps[len(sps) // 2] if sps else 0.0

    for x, y, t in nodes:
        if (round(x, 2), round(y, 2), t) in cat_keys:
            continue
        if not _svg_text_looks_like_number(t):
            continue
        if _is_on_y_axis(x):
            continue
        if abs(y - cats_y_mean) < 10.0:
            continue
        v = _svg_parse_number(t)
        if v is None:
            continue
        idx = min(range(len(cat_xs)), key=lambda i: abs(cat_xs[i] - x))
        if med_sp > 0 and abs(cat_xs[idx] - x) > med_sp * 0.55:
            continue
        per_cat[idx].append((y, v))

    # 众数列数
    from collections import Counter

    lens = [len(lst) for lst in per_cat.values() if lst]
    if not lens:
        return ""
    n_cols = Counter(lens).most_common(1)[0][0]
    if n_cols < 1:
        return ""

    # 要求至少一半类别有数据
    non_empty = sum(1 for lst in per_cat.values() if lst)
    if non_empty < max(2, len(cats) // 2 + 1):
        return ""

    sep_col = " \\t "
    sep_row = " \\n "
    header = [""] * (n_cols + 1)
    lines = [sep_col.join(header)]
    for i in range(len(cats)):
        items = sorted(per_cat[i], key=lambda p: p[0])
        # 取前 n_cols 个
        cells = [_svg_format_number(v) for _, v in items[:n_cols]]
        while len(cells) < n_cols:
            cells.append("")
        lines.append(sep_col.join([cat_labels[i]] + cells))
    return sep_row.join(lines)


def _svg_to_csv_text_driven(
    nodes: list[tuple[float, float, str]],
    axes: dict,
) -> str:
    """文本驱动的 CSV 提取：当图里已有足够的数值 ``<text>`` 时，直接按 x 就近
    把数值文本归到类别行，不走几何插值。

    触发条件：
      * 识别到 X 轴类别行（``axes['cats']`` 非空，且 ≥ 2 个类别）；
      * "非 X 轴类别、非 Y 轴刻度"的数值 ``<text>`` 数量 ≥ 类别数；

    流程：
      1) 从 nodes 中剔除 X 轴类别与 Y 轴刻度（贴近 y_left_x / y_right_x 的数值 text）；
      2) 剩余"数值文本"按 x 归到最近的类别；
      3) 每个类别内，数值按 y 升序（屏幕 y 越小 = 视觉越上）；
      4) 按类别数 × 每类数值数输出表格，空表头（让 csv_eval 走无表头对齐）。

    这是一种"模型已经把数字写在图上了"的情形，典型的组合图、带数据标签的柱/折线。
    若触发条件不满足，返回空串由外层走 text_only。
    """
    cats = axes.get("cats") or []  # [(x, y, text), ...]
    y_left_x = axes.get("y_left_x")
    y_right_x = axes.get("y_right_x")
    cats_y = (sum(c[1] for c in cats) / len(cats)) if cats else None

    if len(cats) < 2:
        return ""

    # 类别 key 集合（用于剔除类别行）
    cat_keys: set = set()
    for cx, cy, ct in cats:
        cat_keys.add((round(cx, 2), round(cy, 2), ct))

    # 候选数值节点：非类别行，且不贴近左/右 Y 轴刻度列
    def _is_on_y_axis(x: float) -> bool:
        if y_left_x is not None and abs(x - y_left_x) <= 12.0:
            return True
        if y_right_x is not None and abs(x - y_right_x) <= 12.0:
            return True
        return False

    candidates: list[tuple[float, float, float]] = []  # (x, y, parsed_value)
    for x, y, t in nodes:
        if (round(x, 2), round(y, 2), t) in cat_keys:
            continue
        if not _svg_text_looks_like_number(t):
            continue
        if _is_on_y_axis(x):
            continue
        # 跳过位于类别行 y ±tol 的（避免把类别行里混的数字当数据标签）
        if cats_y is not None and abs(y - cats_y) < 8.0:
            continue
        v = _svg_parse_number(t)
        if v is None:
            continue
        candidates.append((x, y, v))

    # 触发条件：数值节点数量 ≥ 类别数
    if len(candidates) < len(cats):
        return ""

    cat_xs = [c[0] for c in cats]
    cat_labels = [_clean_label(c[2]) for c in cats]

    # 类别间距（用于限制就近匹配的最大距离）
    spacings = sorted(abs(cat_xs[i] - cat_xs[i - 1]) for i in range(1, len(cat_xs)))
    median_sp = spacings[len(spacings) // 2] if spacings else 0.0

    # 按 x 就近匹配每个数值到某个类别
    groups: dict[int, list[tuple[float, float]]] = {i: [] for i in range(len(cats))}
    for cx, cy, cv in candidates:
        idx = min(range(len(cat_xs)), key=lambda i: abs(cat_xs[i] - cx))
        if median_sp > 0 and abs(cat_xs[idx] - cx) > median_sp * 0.55:
            continue
        groups[idx].append((cy, cv))

    per_cat_values: list[list[float]] = []
    for i in range(len(cats)):
        items = sorted(groups[i], key=lambda p: p[0])
        per_cat_values.append([v for _, v in items])

    non_empty = sum(1 for lst in per_cat_values if lst)
    if non_empty < max(2, len(cats) // 2 + 1):
        return ""

    from collections import Counter

    lens = [len(lst) for lst in per_cat_values if lst]
    if not lens:
        return ""
    n_cols = Counter(lens).most_common(1)[0][0]
    if n_cols < 1:
        return ""

    # 避免与成熟的专用解析器冲突：若 n_cols == 5（箱线图） / n_cols == 1（柱状图单值）
    # 这两类在专用解析器足够稳时不应被文本驱动覆盖——但专用解析器失败才会进到这里，
    # 所以继续产出。
    norm_rows: list[list[str]] = []
    for i, lst in enumerate(per_cat_values):
        row_cells = [_svg_format_number(v) if v is not None else "" for v in lst[:n_cols]]
        while len(row_cells) < n_cols:
            row_cells.append("")
        norm_rows.append([cat_labels[i]] + row_cells)

    sep_col = " \\t "
    sep_row = " \\n "
    header = [""] * (n_cols + 1)
    lines = [sep_col.join(header)]
    for r in norm_rows:
        lines.append(sep_col.join(r))
    return sep_row.join(lines)


def _svg_fallback_single_column(
    numeric_nodes: list[tuple[float, float, str]],
    label_nodes: list[tuple[float, float, str]],
) -> str:
    """识别 X 轴失败时的回退：每个数字找 x 最近的非数字 label 作为行头。"""
    if not numeric_nodes:
        return ""
    rows: list[tuple[str, str]] = []
    for nx, ny, nt in numeric_nodes:
        if label_nodes:
            nearest = min(
                label_nodes,
                key=lambda p: abs(p[0] - nx) + abs(p[1] - ny) * 0.3,
            )
            rows.append((_clean_label(nearest[2]), nt.strip()))
        else:
            rows.append(("", nt.strip()))
    header = ["", ""]
    lines = [" \\t ".join(header)]
    for lbl, val in rows:
        lines.append(" \\t ".join([lbl, val]))
    return " \\n ".join(lines)


def svg_to_markdown_list(text: str) -> str:
    """从 SVG 代码中提取逻辑结构图信息，转换为 Markdown 多级无序列表。

    策略：
      1) 抽取所有 ``<text>`` / ``<tspan>`` 的 (x, y, 文字)；
      2) 过滤纯符号、样式字符串等非节点文本；
      3) 识别根节点：y 值显著小于其他节点（"顶部中心" 布局）→ 该节点为根；
         或 x 值显著小于其他节点（"左→右" 布局）→ 该节点为根；
      4) 若检测到"有明确根"的布局，则：
         - 根为 L0；其他节点按 x 聚类推断层级（根节点所在层不参与聚类）；
         - 同层节点按 y 排序，每个子节点的父 = 上一层中 y 最接近的节点；
      5) 否则复用 ``_build_hierarchy_from_positions`` 按 x 左→右分层；
      6) 最终若层级推断失败，退化为扁平列表。

    Args:
        text: 模型输出的 SVG（可能含 ``xml`` 代码围栏）

    Returns:
        Markdown 多级无序列表字符串；完全无节点则返回空串。
    """
    svg_src = _extract_svg_snippet(text)
    if not svg_src or "<" not in svg_src:
        return ""

    # 同 svg_to_internal_csv：先展平 transform
    svg_src = _svg_flatten_transforms(svg_src)

    raw_nodes = _parse_svg_texts(svg_src)
    if not raw_nodes:
        return ""

    # 过滤：去重 + 去除数字节点（SVG 中的坐标刻度）+ 去除不合格节点
    seen: set = set()
    filtered: list[tuple[float, float, str]] = []
    for x, y, t in raw_nodes:
        tt = _clean_label(t)
        if not tt or tt in seen:
            continue
        if _svg_text_looks_like_number(tt):
            continue
        if not _is_plausible_node_text(tt):
            continue
        seen.add(tt)
        filtered.append((x, y, tt))

    if not filtered:
        return ""

    if len(filtered) == 1:
        return f"- {filtered[0][2]}"

    # 优先 0: 若 SVG 中含 data-parent 属性（由新版 SE_SVG_PROMPT_LOGIC 驱动输出），
    # 直接用父子映射构造 MD，绕开所有几何启发式
    parent_map = _parse_svg_parent_map(svg_src)
    if parent_map:
        md_pm = _build_md_from_parent_map(parent_map)
        if md_pm:
            return md_pm

    # 检测"顶部中心根 / 左侧根"布局
    md = _svg_build_hierarchy_with_root_detection(filtered)
    if md:
        return md

    # 回退到按 x 分层
    md = _build_hierarchy_from_positions(filtered)
    if md:
        return md

    # 兜底：扁平列表。根节点选 y 最小者（思维导图常为顶部/中心总标题）
    root_node = min(filtered, key=lambda n: n[1])
    others = [n for n in filtered if n[2] != root_node[2]]
    # 其他节点按 y 升序排列，作为 L1 子节点
    others.sort(key=lambda n: (n[1], n[0]))
    lines = [f"- {root_node[2]}"]
    for _, _, t in others:
        lines.append(f"  - {t}")
    return "\n".join(lines)


def _svg_build_hierarchy_with_root_detection(
    nodes: list[tuple[float, float, str]],
) -> str:
    """先识别根节点（顶部中心或左侧），再按坐标推断层级。

    根节点判定：
      * 若某节点 y 显著小于其他（差距 ≥ y 跨度的 30%）→ 顶部根；
      * 若某节点 x 显著小于其他（差距 ≥ x 跨度的 30%）→ 左侧根；
      * 否则返回空串（交给调用方走其他策略）。

    确认根节点后，其他节点按 **到根的主导距离方向** 分层：
      * 顶部根：其他节点按 y 聚类成若干层；
      * 左侧根：其他节点按 x 聚类成若干层。

    每层节点按副轴坐标排序，相邻层之间以副轴坐标就近原则建立父子关系。
    """
    if len(nodes) < 3:
        return ""

    xs = [x for x, _, _ in nodes]
    ys = [y for _, y, _ in nodes]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)

    # 按 y 最小找候选根（顶部）
    sorted_by_y = sorted(nodes, key=lambda n: n[1])
    top_root: tuple[float, float, str] | None = None
    if y_span > 0 and len(sorted_by_y) >= 2:
        gap_top = sorted_by_y[1][1] - sorted_by_y[0][1]
        if gap_top >= y_span * 0.3:
            top_root = sorted_by_y[0]

    # 按 x 最小找候选根（左侧）
    sorted_by_x = sorted(nodes, key=lambda n: n[0])
    left_root: tuple[float, float, str] | None = None
    if x_span > 0 and len(sorted_by_x) >= 2:
        gap_left = sorted_by_x[1][0] - sorted_by_x[0][0]
        if gap_left >= x_span * 0.3:
            left_root = sorted_by_x[0]

    # 顶部根优先
    if top_root is not None:
        return _svg_build_hierarchy_from_root(nodes, top_root, main_axis="y")
    if left_root is not None:
        return _svg_build_hierarchy_from_root(nodes, left_root, main_axis="x")

    return ""


def _svg_build_hierarchy_from_root(
    nodes: list[tuple[float, float, str]],
    root: tuple[float, float, str],
    main_axis: str,
) -> str:
    """给定根节点和主轴方向，按主轴坐标对其他节点分层输出 Markdown。

    Args:
        nodes: 所有节点 (x, y, text)
        root: 根节点
        main_axis: "y" (顶部根，层级沿 y 递增) 或 "x" (左侧根，层级沿 x 递增)
    """
    others = [n for n in nodes if n[2] != root[2]]
    if not others:
        return f"- {root[2]}"

    def _main(n: tuple[float, float, str]) -> float:
        return n[1] if main_axis == "y" else n[0]

    def _cross(n: tuple[float, float, str]) -> float:
        return n[0] if main_axis == "y" else n[1]

    main_vals = [_main(n) for n in others]
    main_span = max(main_vals) - min(main_vals)
    tol_main = max(main_span * 0.08, 3.0) if main_span > 0 else 3.0

    # 沿主轴聚类
    sorted_others = sorted(others, key=_main)
    layers: list[list[tuple[float, float, str]]] = []
    cur: list[tuple[float, float, str]] = []
    cur_v: float | None = None
    for n in sorted_others:
        v = _main(n)
        if cur_v is None or abs(v - cur_v) <= tol_main:
            cur.append(n)
            cur_v = v if cur_v is None else (cur_v + v) / 2
        else:
            layers.append(cur)
            cur = [n]
            cur_v = v
    if cur:
        layers.append(cur)

    if not layers:
        return f"- {root[2]}"

    # 每层按副轴排序
    for layer in layers:
        layer.sort(key=_cross)

    # 父子关系：第 1 层的父是根；后续每层的父 = 上一层中副轴最近
    children_map: dict[str, list[tuple[float, float, str]]] = {}
    for child in layers[0]:
        children_map.setdefault(root[2], []).append(child)
    for i in range(1, len(layers)):
        parent_layer = layers[i - 1]
        for child in layers[i]:
            parent = min(parent_layer, key=lambda p: abs(_cross(p) - _cross(child)))
            children_map.setdefault(parent[2], []).append(child)

    # DFS 输出
    lines: list[str] = []
    visited: set = set()

    def _dfs(node: tuple[float, float, str], level: int) -> None:
        if node[2] in visited:
            return
        visited.add(node[2])
        lines.append(f"{'  ' * level}- {node[2]}")
        for c in children_map.get(node[2], []):
            _dfs(c, level + 1)

    _dfs(root, 0)
    return "\n".join(lines)


# ============================================================
# 归一化入口：根据 task 将 prediction 转换为评分需要的格式
# ============================================================
