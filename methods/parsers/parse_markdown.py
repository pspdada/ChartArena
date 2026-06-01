"""Markdown 表格 ↔ 内部 CSV；标准 CSV ↔ 内部 CSV；HTML 表格 → 内部 CSV。"""

import csv
import html as _html
import io
import re

from ..context import strip_code_fence

# ============================================================
# HTML <table> → 内部 CSV
# ============================================================

_HTML_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)
_HTML_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
# 带标签开头的完整单元格（含属性与内容），用于解析 colspan/rowspan
_HTML_CELL_FULL_RE = re.compile(r"<(t[hd])([^>]*)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_HTML_CAPTION_RE = re.compile(r"<caption[^>]*>.*?</caption>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_SPAN_ATTR_RE = re.compile(r"""(colspan|rowspan)\s*=\s*["']?\s*(\d+)\s*["']?""", re.IGNORECASE)


def _clean_html_cell(cell: str) -> str:
    """清理 HTML 单元格文本：去内部标签、解码实体、规范空白。"""
    # <br> 先换成空格，避免把两行数值粘到一起
    cell = re.sub(r"<br\s*/?>", " ", cell, flags=re.IGNORECASE)
    # 去掉其他所有标签（保留文本内容）
    cell = _HTML_TAG_RE.sub("", cell)
    # 解码 HTML 实体（&lt; &gt; &amp; &nbsp; 等）
    cell = _html.unescape(cell)
    # 规范空白
    cell = _WS_RE.sub(" ", cell).strip()
    return cell


def is_html_table(text: str) -> bool:
    """判断文本是否包含 HTML <table> 表格。

    快速短路：直接检查 `<table` 字面（大小写两种）与 `<tr` / `<td>/<th>`，
    避免对整条 pred 做 text.lower() 的全字符串拷贝（大文本场景省内存）。
    """
    if not text:
        return False
    # <table 必须出现（小写或大写中的一种；LLM 输出几乎都是小写）
    if "<table" not in text and "<TABLE" not in text:
        return False
    if "<tr" not in text and "<TR" not in text:
        return False
    if "<td" not in text and "<TD" not in text and "<th" not in text and "<TH" not in text:
        return False
    return True


def html_table_to_csv(text: str) -> str:
    """将包含 HTML <table> 的文本转换为内部 CSV 格式（` \\t ` 列、` \\n ` 行）。

    - 剥离 ```markdown / ```html 代码栅栏
    - 忽略 <caption>（避免把标题混进数据三元组）
    - 遇到多个 <table> 时取第一个（和 MD 表格语义一致：只取第一张表）
    - **支持 colspan / rowspan**：按 CSS 网格规则展开（值复制而非共享），
      避免 `<td rowspan=3>江苏省</td>` 这类样本丢失维度键
    - 跳过整行空 cell
    """
    if not text:
        return ""
    # 去掉代码栅栏
    t = strip_code_fence(text, "markdown")
    t = strip_code_fence(t, "html")
    # 去掉 caption（它会被当作独立行混进数据）
    t = _HTML_CAPTION_RE.sub("", t)

    m = _HTML_TABLE_RE.search(t)
    if not m:
        return ""
    table_body = m.group(1)

    # grid[r] 是第 r 行的单元格列表（按列索引存放）；用 dict 便于稀疏填充后再按列排序
    grid: list[dict[int, str]] = []
    # pending_rowspan[col_idx] = (remaining_rows, value)，表示该列后续还有 N 行需要填入 value
    pending: dict[int, tuple[int, str]] = {}

    for tr_m in _HTML_TR_RE.finditer(table_body):
        tr_body = tr_m.group(1)
        row: dict[int, str] = {}
        col = 0

        def _skip_occupied(c: int) -> int:
            while c in pending and pending[c][0] > 0:
                remaining, value = pending[c]
                row[c] = value
                pending[c] = (remaining - 1, value)
                if pending[c][0] <= 0:
                    del pending[c]
                c += 1
            return c

        col = _skip_occupied(col)

        for cell_m in _HTML_CELL_FULL_RE.finditer(tr_body):
            attrs = cell_m.group(2) or ""
            inner = cell_m.group(3)
            colspan = 1
            rowspan = 1
            for sp in _SPAN_ATTR_RE.finditer(attrs):
                key = sp.group(1).lower()
                val = max(1, int(sp.group(2)))
                if key == "colspan":
                    colspan = val
                elif key == "rowspan":
                    rowspan = val
            value = _clean_html_cell(inner)

            # 按 colspan 横向填充；按 rowspan 登记未来行
            for dc in range(colspan):
                row[col] = value
                if rowspan > 1:
                    pending[col] = (rowspan - 1, value)
                col += 1
            col = _skip_occupied(col)

        if row:
            grid.append(row)

    if not grid:
        return ""

    # 对齐列数：每行按 max_col 展平，缺失补空
    max_col = max(max(r.keys()) for r in grid) + 1
    rows_out: list[str] = []
    for r in grid:
        cells = [r.get(c, "") for c in range(max_col)]
        # 跳过全空行
        if all(not c for c in cells):
            continue
        rows_out.append(" \\t ".join(cells))

    if not rows_out:
        return ""
    return " \\n ".join(rows_out)


def markdown_to_csv(md_text: str) -> str:
    """将 Markdown 表格转换为 CSV 格式（\\t 分隔列，\\n 分隔行）"""
    lines = md_text.strip().split("\n")
    csv_rows = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not line.startswith("|"):
            continue
        # 跳过分隔符行
        if re.match(r"^\|[\s\-:]+\|$", line) or ("-" in line and all(c in "-|: " for c in line)):
            continue

        cells = line.strip("|").split("|")
        cells = [cell.strip() for cell in cells]
        csv_rows.append(" \\t ".join(cells))

    return " \\n ".join(csv_rows)


def is_markdown_table(text: str) -> bool:
    """判断文本是否为 Markdown 表格格式"""
    lines = text.strip().split("\n")
    has_pipe_line = False
    has_separator = False

    for line in lines:
        line = line.strip()
        if line.startswith("|") and line.endswith("|"):
            has_pipe_line = True
        if re.match(r"^\|[\s\-:]+\|$", line) or (
            line.startswith("|") and "-" in line and all(c in "-|: " for c in line)
        ):
            has_separator = True

    return has_pipe_line and has_separator


# ============================================================
# "无分隔线、无首尾竖线" 的 pipe 表格（TinyChart 等模型专属）
# 形如:
#     Year | Value
#     1965 | 1.0
#     1970 | 1.1
# 或带 ASCII 边框:
#     +----+----+
#     | 52.0 | 47.0 |
#     | 46.0 | 46.0 |
# ============================================================


# ASCII 表格边框 / 纯分隔行：+---+---+  /  |---|---|  /  --- 等
_ASCII_BORDER_RE = re.compile(r"^[+\-|\s:=]+$")


def is_pipe_table(text: str) -> bool:
    """判断文本是否为 "无分隔线、无首尾竖线" 的 pipe 表格。

    识别依据：
      * 至少 2 行非空、非 ASCII 边框行
      * 这些行里，至少 70% 含有 ``|`` 分隔符，且去掉首尾 ``|`` 后列数一致
      * 最常见列数 >= 2（单列无意义）
      * 不在 ``is_markdown_table`` 识别范围内（避免与正规 md 表格互斥）
    """
    if not text or not text.strip():
        return False
    if is_markdown_table(text):
        return False

    raw_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    # 去除 ASCII 边框行
    data_lines = [ln for ln in raw_lines if not _ASCII_BORDER_RE.match(ln)]
    if len(data_lines) < 2:
        return False

    col_counts: list[int] = []
    for ln in data_lines:
        if "|" not in ln:
            continue
        body = ln.strip("|").strip()
        # 计列：原始 split('|')；保留空列
        col_counts.append(len(body.split("|")))

    # 70% 以上的数据行含 `|`
    if len(col_counts) < max(2, int(len(data_lines) * 0.7)):
        return False
    if not col_counts or max(col_counts) < 2:
        return False

    # 列数稳定：最高频列数至少占含 `|` 行的 70%
    from collections import Counter as _C

    cc = _C(col_counts)
    top_cnt, top_n = cc.most_common(1)[0]
    if top_n < max(2, int(len(col_counts) * 0.7)):
        return False
    return True


def pipe_table_to_csv(text: str) -> str:
    """将 "无分隔线 pipe 表格" 转换为内部 ``\\t/\\n`` 分隔 CSV。

    规则：
      * 以换行分行；去除 ASCII 边框 / 分隔线；跳过不含 ``|`` 的行
      * 每行剥离首尾 ``|`` 再按 ``|`` split，每格 strip
      * 以最高频列数对齐：少于该列数的行补空，多的截断
        （避免个别长行污染整体列数）
      * **自动识别需要转置的两行横向表格**：若结果仅 2 行、
        首行大多为非数值文本、第二行大多为数值，则做转置，
        让每一列变成 ``<label>, <value>`` 的一行（TinyChart 等模型常用的排版）。
      * 至少 2 行有效才返回；否则返回 ``""``
    """
    if not text:
        return ""
    raw_lines = [ln.strip() for ln in text.split("\n")]
    rows: list[list[str]] = []
    for ln in raw_lines:
        if not ln:
            continue
        if _ASCII_BORDER_RE.match(ln):
            continue
        if "|" not in ln:
            continue
        body = ln.strip("|").strip()
        cells = [c.strip() for c in body.split("|")]
        if all(not c for c in cells):
            continue
        rows.append(cells)

    if len(rows) < 2:
        return ""

    # 以最高频列数对齐
    from collections import Counter as _C

    cc = _C(len(r) for r in rows)
    target = cc.most_common(1)[0][0]
    if target < 2:
        return ""

    aligned: list[list[str]] = []
    for cells in rows:
        if len(cells) < target:
            cells = cells + [""] * (target - len(cells))
        elif len(cells) > target:
            cells = cells[:target]
        aligned.append(cells)

    # ---- 横向表格转置启发 ----
    # 典型形态：
    #     cat_a | cat_b | cat_c | cat_d
    #     45    | 30    | 20    | 25
    # 此时第 1 行是"分类名"、第 2 行是"数值"，与通常的 "行头+列值" 结构相反。
    # 判断条件（保守，避免误伤）：
    #   1. 恰好 2 行
    #   2. 列数 >= 3（2 列时转置与原始语义等价，无需处理）
    #   3. 首行几乎全是非数值文本（非数值占比 >= 80%）
    #   4. 次行几乎全是数值或带百分号（数值占比 >= 80%）
    if len(aligned) == 2 and target >= 3:
        head, data = aligned[0], aligned[1]

        def _is_numeric(s: str) -> bool:
            if not s:
                return False
            t = s.strip().rstrip("%").replace(",", "")
            if not t:
                return False
            try:
                float(t)
                return True
            except ValueError:
                return False

        head_nonnum = sum(1 for c in head if c and not _is_numeric(c))
        data_num = sum(1 for c in data if _is_numeric(c))
        if head_nonnum >= max(2, int(0.8 * target)) and data_num >= max(2, int(0.8 * target)):
            # 转置为 N 行 2 列："label, value"
            aligned = [[h, d] for h, d in zip(head, data)]

    if len(aligned) < 2:
        return ""
    out_rows = [" \\t ".join(cells) for cells in aligned]
    return " \\n ".join(out_rows)


def is_csv_format(text: str) -> bool:
    """判断文本是否为 `\\t/\\n` 分隔的内部 CSV 格式"""
    return "\\t" in text and "\\n" in text


# ============================================================
# 标准 CSV（逗号+换行） → 内部 CSV (\t/\n 分隔)
# ============================================================


def standard_csv_to_internal(text: str) -> str:
    """将标准 CSV（逗号+换行）转换为内部 `\\t/\\n` 分隔 CSV 格式。

    使用 csv 模块解析，正确处理引号包裹的字段、转义等。

    Args:
        text: 标准 CSV 文本（逗号分隔，换行分隔行）

    Returns:
        内部 CSV 格式字符串（` \\t `/` \\n ` 分隔）
    """
    if not text or not text.strip():
        return ""

    text = strip_code_fence(text, "csv")
    text = text.strip()

    try:
        reader = csv.reader(io.StringIO(text))
        rows = [row for row in reader if any(cell.strip() for cell in row)]
    except Exception:
        return ""

    if not rows:
        return ""

    csv_rows = [" \\t ".join(cell.strip() for cell in row) for row in rows]
    return " \\n ".join(csv_rows)


def is_standard_csv(text: str) -> bool:
    """粗略判断是否为标准 CSV（逗号分隔，多行）"""
    if not text or not text.strip():
        return False
    t = strip_code_fence(text, "csv").strip()
    if "\\t" in t and "\\n" in t:
        # 内部格式
        return False
    lines = [line for line in t.split("\n") if line.strip()]
    if len(lines) < 2:
        return False
    # 至少一半行包含逗号
    with_comma = sum(1 for line in lines if "," in line)
    return with_comma >= max(1, len(lines) // 2)


# ============================================================
# JSON → 内部 CSV (\t/\n) 或 Markdown 无序列表
# ============================================================
