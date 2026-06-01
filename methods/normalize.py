"""归一化入口：根据 task 将 prediction 转换为评分需要的格式。"""

import re

from .context import get_chart_type, strip_code_fence
from .parsers import (
    adjacency_csv_to_markdown_list,
    html_table_to_csv,
    is_csv_format,
    is_html_table,
    is_markdown_table,
    is_pipe_table,
    is_standard_csv,
    json_to_internal_csv,
    json_tree_to_markdown_list,
    markdown_to_csv,
    normalize_pie_prediction,
    pipe_table_to_csv,
    python_code_to_internal_csv,
    python_code_to_markdown_list,
    python_code_to_mermaid,
    standard_csv_to_internal,
    svg_to_internal_csv,
    svg_to_markdown_list,
)

# Mermaid 相关（SE_MD / STRUCTUAL_EXTRACTION 兜底）
try:
    from metrics.mermaid_eval import is_mermaid, mermaid_to_markdown_list
except Exception:  # pragma: no cover

    def is_mermaid(_t: str) -> bool:
        return False

    def mermaid_to_markdown_list(t: str) -> str:
        return t


# Markdown list 识别（用于判断 pred 是否已是标准列表）
try:
    from metrics.tree_eval import is_markdown_list
except Exception:  # pragma: no cover

    def is_markdown_list(_t: str) -> bool:
        return False


# ============================================================
# 纯文本/纯缩进 → Markdown 无序列表（思维导图 pred 兜底）
# ============================================================

# 分隔线 / 标题线：--- / === / ___ 等
_SEP_LINE_RE = re.compile(r"^[-=_*~#]{2,}\s*$")
# 行首已有列表符号
_LIST_PREFIX_RE = re.compile(r"^\s*[-*+]\s+")
# 行首编号（1. / 1) / 1、 / 一、 等），视作无序列表项
_NUM_PREFIX_RE = re.compile(r"^\s*(\d+[\.\)、]|[一二三四五六七八九十]+[、\.])\s*")
# HTML 标签起始
_HTML_TAG_RE = re.compile(r"^\s*<[^>]+>")


def _pipe_table_to_markdown_list(text: str) -> str:
    """将管道分隔的表格格式转为 Markdown 多级无序列表。

    使用场景：某些模型（如 PaddleOCR_VL）把思维导图当成表格输出，格式如：
        ``项目 | 数值``
        ``客户探索 | 争取支持``
        ``客户探索 | 提出假设``
        ``客户效验 | 明确产品定位``

    策略：
      1) 检测是否是管道分隔格式（≥50% 的行含 ``|``）
      2) 跳过第一行（通常是表头，如 "Category | Value"）
      3) 利用管道符左边的分类名构建层级：
         - 如果同一个左边值出现多次，它就是父节点，右边的值是子节点
         - 如果左边有缩进，用缩进推断层级
      4) 去重：相同内容的节点只保留第一次出现
    """
    if not text or not text.strip():
        return ""

    lines = text.strip().split("\n")
    if len(lines) < 2:
        return ""

    # 检测是否是管道分隔格式
    pipe_lines = sum(1 for ln in lines if "|" in ln)
    if pipe_lines < len(lines) * 0.5:
        return ""

    # 判断第一行是否是表头
    first_line = lines[0].strip()
    skip_first = "|" in first_line
    has_root = not skip_first and first_line

    data_lines = lines[1:] if skip_first or has_root else lines

    # 解析每行的 (缩进, 左边, 右边)
    parsed_rows: list[tuple[int, str, str]] = []
    for line in data_lines:
        raw = line.rstrip()
        if not raw.strip():
            continue

        # 计算前导空白
        indent_spaces = 0
        for ch in raw:
            if ch == " ":
                indent_spaces += 1
            elif ch == "\t":
                indent_spaces += 2
            else:
                break

        if "|" in raw:
            parts = [p.strip() for p in raw.split("|")]
            left = parts[0].replace("**", "").strip() if parts[0] else ""
            right = parts[1].replace("**", "").strip() if len(parts) > 1 and parts[1] else ""
            # 过滤纯数字
            if left and re.fullmatch(r"[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?%?", left):
                left = ""
            if right and re.fullmatch(r"[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?%?", right):
                right = ""
            # 如果有 ≥3 个管道分隔部分，把所有非空、非数字的部分都收集
            extra_parts = []
            for p in parts[2:]:
                p_clean = p.replace("**", "").strip()
                if p_clean and not re.fullmatch(r"[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?%?", p_clean):
                    extra_parts.append(p_clean)
            parsed_rows.append((indent_spaces, left, right, extra_parts))
        else:
            parsed_rows.append((indent_spaces, raw.strip(), "", []))

    if not parsed_rows:
        return ""

    # 统计左边值出现的次数，判断哪些是"分类名"（父节点）
    from collections import Counter

    left_counts = Counter()
    for _, left, right, _ in parsed_rows:
        if left and right:  # 只有左右都非空时，左边才可能是分类名
            left_counts[left.lower()] += 1

    # 如果某个左边值出现 ≥2 次且有对应的右边值，它就是父节点
    category_lefts = {k for k, v in left_counts.items() if v >= 2}

    # 构建层级结构
    out_lines: list[str] = []
    seen: set[str] = set()

    if has_root:
        out_lines.append("- " + first_line)
        seen.add(first_line.lower())

    if category_lefts:
        # 有分类结构：左边是父节点，右边是子节点
        current_category = ""
        for indent, left, right, extras in parsed_rows:
            left_key = left.lower() if left else ""

            if left_key in category_lefts:
                # 左边是分类名（父节点）
                if left_key != current_category:
                    current_category = left_key
                    if left_key not in seen:
                        seen.add(left_key)
                        out_lines.append("- " + left)
                # 右边是子节点
                if right:
                    right_key = right.lower()
                    if right_key not in seen:
                        seen.add(right_key)
                        out_lines.append("  - " + right)
                # 额外列也作为子节点
                for extra in extras:
                    extra_key = extra.lower()
                    if extra_key not in seen:
                        seen.add(extra_key)
                        out_lines.append("  - " + extra)
            else:
                # 左边不是分类名，把所有非空部分都作为独立节点
                for name in [left, right] + extras:
                    if name:
                        name_key = name.lower()
                        if name_key not in seen:
                            seen.add(name_key)
                            out_lines.append("- " + name)
    else:
        # 没有分类结构，利用缩进推断层级
        uniq_indents = sorted({sp for sp, _, _, _ in parsed_rows})
        indent_to_level = {sp: lvl for lvl, sp in enumerate(uniq_indents)}

        for indent, left, right, extras in parsed_rows:
            lvl = indent_to_level[indent]
            # 把所有非空部分都作为节点
            for name in [left, right] + extras:
                if name:
                    name_key = name.lower()
                    if name_key not in seen:
                        seen.add(name_key)
                        out_lines.append("  " * lvl + "- " + name)

    if not out_lines:
        return ""

    return "\n".join(out_lines)


def plain_text_to_markdown_list(text: str) -> str:
    """将"纯文本/纯缩进/不规范"的 pred 兜底转为 Markdown 多级无序列表。

    使用场景：思维导图 ref 是标准 Markdown list，而模型 pred 是以下之一：
        1. 每行一个节点、无任何列表符号也无缩进（完全扁平）
        2. 按"空格/tab 缩进"表达层级但没有 `- ` 前缀
        3. 混合：部分行有 `- *` 前缀、部分行无；有的行被 `---` 分隔线打断

    归一化策略：
        - 跳过空行、分隔线、HTML 标签
        - 统计每行的前导空白字符数，映射为缩进级别（每 2 空格/1 tab 一级）
        - 每行内容若已带列表符号则剥离，统一用 `- ` 前缀输出
        - 若所有行缩进都为 0（完全扁平），则整体变成同级 root 列表
          （至少让 tree_eval 能识别到所有节点，给部分分而非 0 分）
    """
    if not text or not text.strip():
        return ""

    # 逐行收集 (indent_spaces, content)
    rows: list[tuple[int, str]] = []
    for line in text.split("\n"):
        # 保留前导空白，去掉行尾空白
        raw = line.rstrip()
        if not raw.strip():
            continue
        # 跳过分隔线
        if _SEP_LINE_RE.match(raw.strip()):
            continue
        # 跳过 HTML 标签行
        if _HTML_TAG_RE.match(raw):
            continue

        # 计算前导空白：空格=1，tab=2
        indent_spaces = 0
        idx = 0
        for ch in raw:
            if ch == " ":
                indent_spaces += 1
                idx += 1
            elif ch == "\t":
                indent_spaces += 2
                idx += 1
            else:
                break
        body = raw[idx:]

        # 剥离可能的列表符号与编号前缀
        m = _LIST_PREFIX_RE.match(body)
        if m:
            body = body[m.end() :]
        else:
            m = _NUM_PREFIX_RE.match(body)
            if m:
                body = body[m.end() :]
        body = body.strip()
        if not body:
            continue

        rows.append((indent_spaces, body))

    if not rows:
        return ""

    # 规范化缩进：把真实空白数映射为 "每级 2 空格"
    # 做法：取所有出现过的缩进值，从小到大编号为 0,1,2,...
    uniq_indents = sorted({sp for sp, _ in rows})
    indent_to_level = {sp: lvl for lvl, sp in enumerate(uniq_indents)}

    out_lines: list[str] = []
    for sp, body in rows:
        lvl = indent_to_level[sp]
        out_lines.append("  " * lvl + "- " + body)

    return "\n".join(out_lines)


def normalize_to_csv(text: str) -> str:
    """将文本统一转换为内部 CSV 格式（用于 SE_MD 的数值类分支）"""
    if not text or not text.strip():
        return ""

    text = text.strip()

    # 优先识别 HTML <table>（pred 常见；MD 表格很少被包成 HTML）
    if is_html_table(text):
        csv_text = html_table_to_csv(text)
        if csv_text:
            return csv_text
        # 失败则继续走后面的 MD / CSV 识别

    if is_markdown_table(text):
        return markdown_to_csv(text)

    # TinyChart 等模型输出的"无分隔线 pipe 表格"
    #   e.g. `Year | Value\n1965 | 1.0`
    # 提到 is_csv_format 之前识别，避免极少数 pred 同时含 `\t` / `\n` 字面子串
    # （例如 pred 里带有说明性文字 "...using \t as separator..."）被误判成内部 CSV。
    if is_pipe_table(text):
        csv_text = pipe_table_to_csv(text)
        if csv_text:
            return csv_text

    if is_csv_format(text):
        return text

    return text


def _strip_ref_header_for_svg(ref_csv: str) -> str:
    """将 ref CSV 的首行替换为与其列数相同的空表头，使 csv2triples 进入无表头模式。

    仅当原 ref 的首行非空表头时才替换；若首行已是无表头（_is_empty_header==True），
    返回原样。
    """
    if not ref_csv:
        return ref_csv
    sep_col = r" \t "
    sep_row = r" \n "
    lines = ref_csv.strip().split(sep_row)
    if not lines:
        return ref_csv
    header = lines[0].split(sep_col)
    try:
        from metrics.SCRM import _is_empty_header  # local import to avoid cycle

        if _is_empty_header(header):
            return ref_csv
    except Exception:
        pass
    n = len(header)
    empty = sep_col.join([""] * n)
    return sep_row.join([empty] + lines[1:])


def normalize_prediction_for_data(prediction: str, task: str) -> str:
    """对数值类图表的 prediction 进行归一化，统一输出到内部 CSV 格式。

    Args:
        prediction: 模型原始输出
    task: 任务类型（SE_MD / SE_JSON / SE_CSV / SE_CODE / SE_SVG / STRUCTUAL_EXTRACTION）
    """
    if not prediction:
        return ""
    pred = prediction.strip()

    def _finalize(pred_csv: str) -> str:
        """饼图专用 pred 后处理（非饼图样本原样返回）。"""
        chart_type = get_chart_type()
        return normalize_pie_prediction(pred, pred_csv, chart_type)

    if task == "SE_JSON":
        csv_text = json_to_internal_csv(pred)
        if csv_text:
            return _finalize(csv_text)
        # 兜底：尝试 markdown/csv 识别
        return _finalize(normalize_to_csv(pred))

    if task == "SE_CSV":
        # 标准 CSV（逗号+换行）
        pred_clean = strip_code_fence(pred, "csv")
        if is_standard_csv(pred_clean):
            return _finalize(standard_csv_to_internal(pred_clean))
        return _finalize(normalize_to_csv(pred_clean))

    if task == "SE_CODE":
        csv_text = python_code_to_internal_csv(pred)
        if csv_text:
            return _finalize(csv_text)
        return _finalize(normalize_to_csv(pred))

    if task == "SE_SVG":
        csv_text = svg_to_internal_csv(pred)
        if csv_text:
            return _finalize(csv_text)
        # SVG 解析失败时直接返回空串，避免把原始 SVG 代码当成 pred_csv 传入评分
        # （normalize_to_csv 对 SVG 代码会走到最后的 return text 兜底，
        #  把整个 SVG 代码原样返回，导致 csv_eval 得 0 分）
        return ""

    # SE_MD / STRUCTUAL_EXTRACTION：按原逻辑识别
    return _finalize(normalize_to_csv(pred))


def normalize_prediction_for_logic(prediction: str, task: str) -> str:
    """对逻辑结构图的 prediction 进行归一化，统一输出到 Markdown 多级无序列表。

    Args:
        prediction: 模型原始输出
        task: 任务类型
    """
    if not prediction:
        return ""
    pred = prediction.strip()

    if task == "SE_JSON":
        md = json_tree_to_markdown_list(pred)
        if md:
            return md
        return pred

    if task == "SE_CSV":
        md = adjacency_csv_to_markdown_list(pred)
        if md:
            return md
        return pred

    if task == "SE_CODE":
        md = python_code_to_markdown_list(pred)
        if md:
            return md
        return pred

    if task == "SE_SVG":
        md = svg_to_markdown_list(pred)
        if md:
            return md
        return pred

    # SE_MD / STRUCTUAL_EXTRACTION
    if is_mermaid(pred):
        return mermaid_to_markdown_list(pred)
    # 兜底：如果 pred 已经是标准 markdown list 直接返回；
    # 否则尝试把"纯文本/纯缩进"的思维导图 pred 转成 markdown list，
    # 避免 parse_markdown_list 因无 `- ` 前缀而丢弃全部行导致 0 分。
    if is_markdown_list(pred):
        return pred
    converted = plain_text_to_markdown_list(pred)
    if converted and is_markdown_list(converted):
        return converted
    return pred


def normalize_prediction_for_flowchart(prediction: str, task: str) -> str:
    """对流程图类参考（mermaid ref）的 prediction 进行归一化。

    目的：当 pred 与 ref 的语言不一致时（例如 ref 是 mermaid、
    pred 是 Python/networkx 代码），先把 pred 翻译成 mermaid，
    供 ``flowchart_eval_multi`` 正常解析，避免直接 parse 失败 0 分。

    当前支持：
      * SE_CODE：python_code_to_mermaid（networkx / edges 列表）
      * SE_MERMAID：当 pred 不是 mermaid 格式（如管道分隔表格）时，
        尝试提取节点名构建 mermaid 流程图
    其它 task 与原 prediction 保持一致，交给下游 DSL 解析器处理。
    """
    if not prediction:
        return ""
    pred = prediction.strip()

    if task == "SE_CODE":
        converted = python_code_to_mermaid(pred)
        if converted:
            return converted
        return pred

    if task == "SE_MERMAID":
        # 如果已经是 mermaid 格式，直接返回
        if is_mermaid(pred):
            return pred
        # 尝试从管道分隔表格中提取节点名，构建 mermaid 流程图（默认拆分模式）
        converted = _pipe_table_to_mermaid(pred, merge_cells=False)
        if converted:
            return converted
        return pred

    return pred


def _pipe_table_to_mermaid(text: str, merge_cells: bool = False) -> str:
    """将管道分隔表格或纯文本列表格式的流程图输出转为 mermaid 流程图。

    使用场景：某些模型（如 PaddleOCR_VL）把流程图当成表格输出，格式如：
        ``Category | Item | Value``
        ``售前操作流程 | 未下单前咨询``
        ``店铺活动介绍 | 正面、耐心、仔细...``

    Args:
        text: 输入文本
        merge_cells: 是否将同一行的多个 cell 合并为一个节点。
            False（默认）：每个 cell 独立作为一个节点
            True：同一行的 cell 合并为一个节点

    策略：
      1) 检测是否是管道分隔格式或纯文本列表
      2) 跳过第一行（表头）
      3) 提取节点名（根据 merge_cells 决定是否合并）
      4) 去重后构建 mermaid 流程图（节点顺序串联）
    """
    if not text or not text.strip():
        return ""

    lines = text.strip().split("\n")
    if len(lines) < 2:
        return ""

    # 检测是否是管道分隔格式
    pipe_lines = sum(1 for ln in lines if "|" in ln)
    is_pipe = pipe_lines >= len(lines) * 0.4

    # 也支持纯文本列表（每行一个节点名）
    if not is_pipe:
        # 纯文本列表：每行一个节点名
        node_names = _extract_nodes_from_text_lines(lines)
    else:
        # 管道分隔表格：跳过第一行（表头）
        data_lines = lines[1:]
        if merge_cells:
            node_names = _extract_nodes_from_pipe_lines_merged(data_lines)
        else:
            node_names = _extract_nodes_from_pipe_lines_split(data_lines)

    if len(node_names) < 2:
        return ""

    # 构建 mermaid 流程图
    mermaid_lines = ["flowchart TD"]
    for i, name in enumerate(node_names):
        escaped = name.replace('"', "'")
        node_id = chr(ord("A") + i) if i < 26 else f"N{i}"
        mermaid_lines.append(f'    {node_id}["{escaped}"]')

    # 添加边：顺序串联
    for i in range(len(node_names) - 1):
        src_id = chr(ord("A") + i) if i < 26 else f"N{i}"
        dst_id = chr(ord("A") + i + 1) if i + 1 < 26 else f"N{i + 1}"
        mermaid_lines.append(f"    {src_id} --> {dst_id}")

    return "\n".join(mermaid_lines)


def _clean_cell(p: str) -> str:
    """清理单个 cell 文本：去掉 markdown 加粗、HTML 标签、列表符号等。"""
    p = p.replace("**", "").strip()
    p = re.sub(r"<br\s*/?>", " ", p)
    p = re.sub(r"<[^>]+>", "", p)
    p = re.sub(r"^[•·\-\*]\s*", "", p).strip()
    return p


def _is_skip_cell(p: str) -> bool:
    """判断 cell 是否应该跳过（纯数字、过短等）。"""
    if not p:
        return True
    # 跳过纯数字（含百分比、千分位）
    if re.fullmatch(r"[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?%?", p):
        return True
    # 跳过过短的内容（单个字符）
    if len(p) <= 1:
        return True
    return False


def _extract_nodes_from_pipe_lines_split(data_lines: list[str]) -> list[str]:
    """从管道分隔的数据行中提取节点名（每个 cell 独立作为一个节点）。"""
    seen: set[str] = set()
    node_names: list[str] = []

    for line in data_lines:
        raw = line.strip()
        if not raw:
            continue

        parts = [p.strip() for p in raw.split("|")]
        for p in parts:
            p_clean = _clean_cell(p)
            if _is_skip_cell(p_clean):
                continue
            # 跳过过短的内容（如 "Yes"、"No"）
            if len(p_clean) <= 2:
                continue
            key = p_clean.lower()
            if key in seen:
                continue
            seen.add(key)
            node_names.append(p_clean)

    return node_names


def _extract_nodes_from_pipe_lines_merged(data_lines: list[str]) -> list[str]:
    """从管道分隔的数据行中提取节点名。

    核心策略：将同一行的多个有意义的 cell 合并为一个节点名。
    """
    seen: set[str] = set()
    node_names: list[str] = []

    for line in data_lines:
        raw = line.strip()
        if not raw:
            continue

        parts = [p.strip() for p in raw.split("|")]
        # 清理每个 cell
        cleaned_parts = []
        for p in parts:
            p_clean = _clean_cell(p)
            if _is_skip_cell(p_clean):
                continue
            cleaned_parts.append(p_clean)

        if not cleaned_parts:
            continue

        # 合并同一行的所有有意义的 cell 为一个节点名
        node_name = " ".join(cleaned_parts)

        # 去重
        key = node_name.lower()
        if key in seen:
            continue
        seen.add(key)
        node_names.append(node_name)

    return node_names


def _extract_nodes_from_text_lines(lines: list[str]) -> list[str]:
    """从纯文本行中提取节点名（每行一个节点）。"""
    seen: set[str] = set()
    node_names: list[str] = []

    for line in lines:
        p_clean = _clean_cell(line.strip())
        if _is_skip_cell(p_clean):
            continue
        # 去重
        key = p_clean.lower()
        if key in seen:
            continue
        seen.add(key)
        node_names.append(p_clean)

    return node_names


# ============================================================
# 评分结果构造
# ============================================================
