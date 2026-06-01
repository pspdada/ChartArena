"""饼图 pred 归一化：处理 "名称+数值" 粘连 / 左右平铺 / mermaid pie 语法。

背景：
    饼图（chart_type 含 "饼图" / "pie"）的 ref 一般是严格 2 列 `"名称" | "数值%"`，
    但模型 pred 经常出现以下不一致形式，导致 csv_eval 0 分：

    1. Mermaid pie 语法：
           pie title "..."
           "名称, 27.74%" : 27.74
       → 被当成整段无结构文本，无法解析为 CSV。

    2. 多列平铺（2 列或 4 列），每 cell = "名称 + 空格 + 数值%"：
           | 卢国建 28.01% | 深圳市海联智合... 16.54% |
       → 与 2 列 ref 列数匹配但内容粘连。

    3. 内部 CSV 中 "name, value" 粘连在同一列：
           Oppo, 8% \t -3%
       → 首列把名称和占比黏到一起；用 "," 分隔。

本模块只作用在 pred 侧，且仅当 ``chart_type`` 命中饼图时才启用，
完全不触碰 ref，保证对非饼图样本零副作用。
"""

from __future__ import annotations

import re
from typing import List

_SEP_COL = r" \t "
_SEP_ROW = r" \n "

# 形如 "12", "12.3", "12%", "-12.3%", "1,234", "1,234.5%" 的数值（允许千分位逗号）
_NUM_PAT = r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?"
# 数值检测（严格：整个字符串都是数值）
_NUM_FULL_RE = re.compile(rf"^\s*{_NUM_PAT}\s*$")

# 末尾数值：用于把 "名称 27.74%" 或 "名称(27.74%)" / "名称（27.74%）" 拆开
# name 至少 1 字符，中间至少 1 个空格/左括号把数值和名称隔开
_NAME_VALUE_RE = re.compile(
    rf"^\s*(?P<name>.+?)\s*(?:[（(]\s*)?(?P<num>{_NUM_PAT})\s*(?:[）)]\s*)?$",
)


def _is_pie_chart_type(chart_type: str) -> bool:
    if not chart_type:
        return False
    ct = chart_type.lower()
    return ("饼图" in chart_type) or ("pie" in ct)


def _split_rows(csv_text: str) -> List[List[str]]:
    if not csv_text:
        return []
    return [row.split(_SEP_COL) for row in csv_text.split(_SEP_ROW)]


def _join_rows(rows: List[List[str]]) -> str:
    return _SEP_ROW.join(_SEP_COL.join(r) for r in rows)


def _is_number_cell(s: str) -> bool:
    return bool(s and _NUM_FULL_RE.match(s))


def _try_split_name_value(cell: str) -> tuple[str, str] | None:
    """尝试把单个 cell 拆成 (name, value)。

    命中规则：
      - "名称 27.74%"            （空格分隔）
      - "名称(27.74%)"           （半/全角括号包数值）
      - "名称, 27.74%"           （逗号分隔——用于内部 CSV 单格粘连）
    必须满足：name 非空 且 name 不全是数字。
    """
    cell = (cell or "").strip()
    if not cell:
        return None
    # 护栏：过长 cell 不尝试（避免正则回溯灾难）
    if len(cell) > 200:
        return None
    # 若整 cell 本来就是纯数值，不拆
    if _is_number_cell(cell):
        return None

    # 优先：逗号分隔 "name, value"
    m = re.match(rf"^(?P<name>.+?)\s*,\s*(?P<num>{_NUM_PAT})\s*$", cell)
    if m:
        name = m.group("name").strip()
        num = m.group("num").strip()
        if name and not _is_number_cell(name):
            return name, num

    # 其次：括号包裹 "name(value)" / "name（value）"
    m = re.match(rf"^(?P<name>.+?)\s*[（(]\s*(?P<num>{_NUM_PAT})\s*[）)]\s*$", cell)
    if m:
        name = m.group("name").strip()
        num = m.group("num").strip()
        if name and not _is_number_cell(name):
            return name, num

    # 最后：空格分隔 "name 27.74%"（要求 value 以 % 结尾或 cell 里至少含 1 个空格）
    m = re.match(rf"^(?P<name>.+?)\s+(?P<num>{_NUM_PAT})\s*$", cell)
    if m:
        name = m.group("name").strip()
        num = m.group("num").strip()
        # 只有当数值形如百分比、或 name 非纯数字时才拆（避免把 "2024 3" 这种年份数对错拆）
        if name and not _is_number_cell(name) and ("%" in num):
            return name, num
    return None


# ============================================================
# 分支 1: Mermaid pie 语法 → 2 列 CSV
# ============================================================
_MERMAID_PIE_LINE_RE = re.compile(rf'^\s*"(?P<name>[^"]+?)"\s*:\s*(?P<num>{_NUM_PAT})\s*$')
_MERMAID_PIE_HEAD_RE = re.compile(r"^\s*pie(\s+.*)?$", re.IGNORECASE)


def _try_mermaid_pie(raw_pred: str) -> str | None:
    """从原始 prediction 文本里识别 Mermaid pie 语法，生成 2 列内部 CSV。"""
    if not raw_pred or "pie" not in raw_pred.lower():
        return None
    t = raw_pred.strip()
    # 剥离代码围栏：用纯字符串切分，避免 re DOTALL 在长文本上递归爆栈
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl != -1:
            body = t[first_nl + 1 :]
            last_fence = body.rfind("```")
            if last_fence != -1:
                t = body[:last_fence].strip()

    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    # 必须出现 pie 头（允许首行可能带 showData）
    if not any(_MERMAID_PIE_HEAD_RE.match(ln) for ln in lines[:3]):
        return None

    rows: list[list[str]] = []
    for ln in lines:
        if _MERMAID_PIE_HEAD_RE.match(ln):
            continue
        if ln.lower().startswith("title "):
            continue
        m = _MERMAID_PIE_LINE_RE.match(ln)
        if not m:
            continue
        name = m.group("name").strip()
        num = m.group("num").strip()
        # name 里若内嵌 ", 27.74%" / "（27.74%）" 形式，剥离
        split = _try_split_name_value(name)
        if split is not None:
            name = split[0]
        # pie 的 value 没带 %，看上下文补上（与 ref 的 "%" 风格统一）
        if "%" not in num:
            try:
                # 只在数值范围 [0, 100]（含整数也可能超出）时补 %，否则原样保留
                v = float(num.replace(",", ""))
                if 0.0 <= v <= 100.0:
                    num = num + "%"
            except ValueError:
                pass
        rows.append([name, num])
    if len(rows) < 2:
        return None
    # 空表头（与 markdown `|  |  |` 的 ref 风格对齐）
    header = ["", ""]
    return _join_rows([header] + rows)


# ============================================================
# 分支 2: 单格内粘连 → 拆列
# ============================================================


def _try_split_glued_column(rows: List[List[str]]) -> List[List[str]] | None:
    """若某**整列**（表头除外）每个 cell 都能拆成 (name, value)，则把该列拆为 2 列。"""
    if len(rows) < 2:
        return None
    n_cols = len(rows[0])
    # 只处理列数 >=1 的情形；从左到右找第一个"整列都能拆"的列
    body = rows[1:]
    for ci in range(n_cols):
        splits: list[tuple[str, str]] = []
        ok = True
        for r in body:
            if ci >= len(r):
                ok = False
                break
            sp = _try_split_name_value(r[ci])
            if sp is None:
                ok = False
                break
            splits.append(sp)
        if not ok or not splits:
            continue
        # 命中：原列替换为 2 列
        new_header = rows[0][:ci] + ["", ""] + rows[0][ci + 1 :]
        # 若 header 原有 n_cols 列，这里插入 2 列（替换原 ci 列为 2 列）
        new_rows = [new_header]
        for r, (name, num) in zip(body, splits):
            new_rows.append(r[:ci] + [name, num] + r[ci + 1 :])
        return new_rows
    return None


# ============================================================
# 分支 3: 多列平铺 (2 列或 4 列，每 cell 都是 "name value") → 展开为 2 列
# ============================================================


def _try_expand_tiled_pairs(rows: List[List[str]]) -> List[List[str]] | None:
    """整 grid（含多列平铺）每个非空 cell 都能拆成 (name, value) → 展开为 2 列。

    典型输入:
        | 股东A 28.01% | 股东B 16.54% |
        | 股东C 2.76%  | 股东D 2.72%  |
    输出:
        | 股东A | 28.01% |
        | 股东B | 16.54% |
        | 股东C | 2.76%  |
        | 股东D | 2.72%  |
    """
    if len(rows) < 2:
        return None
    body = rows[1:]
    if not body:
        return None
    # 护栏：grid 规模过大不处理（饼图正常应该 < 50 片）
    total_cells = sum(len(r) for r in body)
    if total_cells > 200:
        return None
    # 每行至少得有 >=1 列
    all_pairs: list[tuple[str, str]] = []
    for r in body:
        for cell in r:
            if not (cell or "").strip():
                continue
            sp = _try_split_name_value(cell)
            if sp is None:
                return None  # 只要有一个非空 cell 不符合，就放弃（保守）
            all_pairs.append(sp)
    if len(all_pairs) < 2:
        return None
    new_rows: list[list[str]] = [["", ""]]
    for name, num in all_pairs:
        new_rows.append([name, num])
    return new_rows


# ============================================================
# 对外入口
# ============================================================


def normalize_pie_prediction(raw_pred: str, pred_csv: str, chart_type: str) -> str:
    """当 chart_type 命中饼图时，尝试进一步改写 pred_csv。

    优先级：
      1) 原始 pred 是 Mermaid pie 语法 → 直接重建 2 列 CSV
      2) pred_csv grid 中所有非空 cell 都是 "name value" → 展开为 2 列
      3) pred_csv grid 中某整列都是 "name, value" 粘连 → 该列拆为 2 列
    任何一个命中即返回改写结果，否则原样返回 pred_csv。
    """
    if not _is_pie_chart_type(chart_type):
        return pred_csv
    # 输入长度护栏：极长 pred / pred_csv 直接跳过，避免正则回溯拖慢评测
    if (raw_pred and len(raw_pred) > 50000) or (pred_csv and len(pred_csv) > 50000):
        return pred_csv

    # 1) mermaid pie 语法从原始 pred 识别
    merm = _try_mermaid_pie(raw_pred or "")
    if merm:
        return merm

    if not pred_csv:
        return pred_csv

    rows = _split_rows(pred_csv)
    if not rows or not rows[0]:
        return pred_csv

    # 2) 多列平铺 (2 列/4 列) 每格都是 "name value"
    tiled = _try_expand_tiled_pairs(rows)
    if tiled is not None:
        return _join_rows(tiled)

    # 3) 整列粘连 "name, value"
    glued = _try_split_glued_column(rows)
    if glued is not None:
        return _join_rows(glued)

    return pred_csv
