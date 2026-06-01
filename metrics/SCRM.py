import logging
import re
from typing import Any, Literal

import Levenshtein
import numpy as np

logger = logging.getLogger(__name__)

"""
一个图表结构化数据提取的评估指标文件，实现了 SCRM（Structured Chart Recognition Metric） 评估方法，用于衡量模型从图表中提取结构化数据（如表格/CSV）的准确性。

将预测的 CSV 和标注的 CSV 都转换为 三元组（triples） 形式 (实体, 属性, 值)，然后通过计算三元组集合之间的交集/并集相似度来评估提取质量，类似于目标检测中的 mAP 评估体系。
"""

POST_PROCESS_TRIPLES = True


def dedup_logs(logs: list[str]) -> list[str]:
    """对日志列表进行去重：相邻重复的日志合并为一条，标注重复次数。"""
    if not logs:
        return logs
    deduped: list[str] = []
    prev_msg = logs[0]
    count = 1
    for msg in logs[1:]:
        if msg == prev_msg:
            count += 1
        else:
            deduped.append(prev_msg if count == 1 else f"{prev_msg} [Repeat x{count}]")
            prev_msg = msg
            count = 1
    deduped.append(prev_msg if count == 1 else f"{prev_msg} [Repeat x{count}]")
    return deduped


def is_int(val) -> bool:
    try:
        int(val)
        return True
    except ValueError:
        return False


def is_float(val) -> bool:
    try:
        float(val)
        return True
    except ValueError:
        return False


# ============================================================
# 全角 → 半角 标点归一化映射
# ============================================================
# 用于把中文全角标点统一成英文半角，以消除 ref/pred 纯符号差异带来的 0 分。
# 仅针对"对语义无影响的常见标点"，不处理顿号"、"（用户指定不纳入）。
_FULLWIDTH_TO_HALFWIDTH: dict[str, str] = {
    "：": ":",
    "（": "(",
    "）": ")",
    "，": ",",
    "；": ";",
    "％": "%",
}


def _normalize_fullwidth(text: str) -> str:
    """将文本中的全角标点归一化为半角（见 _FULLWIDTH_TO_HALFWIDTH）。

    对所有 SE 任务类别（SE_MD/SE_JSON/SE_CSV/SE_CODE/SE_SVG）统一生效，
    因为它们都走 csv_eval → csv2triples → normalize_triple_key 同一条链路。
    """
    if not text:
        return text
    for fw, hw in _FULLWIDTH_TO_HALFWIDTH.items():
        if fw in text:
            text = text.replace(fw, hw)
    return text


# LaTeX 上标/下标映射
_LATEX_SUPERSCRIPTS = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "n": "ⁿ",
    "+": "⁺",
    "-": "⁻",
}


def _normalize_latex(text: str) -> str:
    """将 LaTeX 数学公式归一化为 Unicode 等价形式。

    常见场景：ref 中的列头使用 LaTeX 格式（如 ``$ m^{2} $``），
    而 pred 使用普通 Unicode（如 ``m²``）。归一化后两者一致。

    处理规则：
      1) 去掉 ``$ ... $`` 包裹（行内公式标记）
      2) ``\\text{...}`` → 直接取花括号内的文本
      3) ``\\mathrm{...}`` → 直接取花括号内的文本
      4) ``^{2}`` → ``²``（上标数字）
      5) ``_{2}`` → ``₂``（下标数字，暂不处理）
      6) 清理多余空格
    """
    if "$" not in text and "\\" not in text:
        return text

    # 去掉 $ ... $ 包裹
    result = re.sub(r"\$\s*(.*?)\s*\$", r"\1", text)

    # \text{...} → 内容
    result = re.sub(r"\\text\{([^}]*)\}", r"\1", result)
    # \mathrm{...} → 内容
    result = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", result)
    # \textbf{...} → 内容
    result = re.sub(r"\\textbf\{([^}]*)\}", r"\1", result)

    # ^{N} → 上标 Unicode（仅处理单个数字）
    def _sup_repl(m: re.Match) -> str:
        content = m.group(1)
        return "".join(_LATEX_SUPERSCRIPTS.get(c, c) for c in content)

    result = re.sub(r"\^\{([^}]*)\}", _sup_repl, result)
    # ^N（无花括号，单字符上标）
    result = re.sub(r"\^(\d)", lambda m: _LATEX_SUPERSCRIPTS.get(m.group(1), m.group(1)), result)

    # 清理多余空格
    result = re.sub(r"\s+", " ", result).strip()

    return result


# ============================================================
# 箱线图统计量归一化映射
# ============================================================
# 将各种表述统一为 "-最小值", "-Q1", "-中位数", "-Q3", "-最大值" 的后缀形式
# 例如 "工资下四分位数(Q1)" -> "工资-Q1"，"salary-Q1" -> "salary-Q1"（已经是目标格式）

_BOXPLOT_STAT_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Q1 / 下四分位数 / 第一四分位数（带或不带 Q1 后缀均可）
    (re.compile(r"[_\-]?(?:下|第一)四分位数?\s*(?:[\(（]\s*Q1\s*[\)）])?", re.IGNORECASE), "-Q1"),
    (re.compile(r"[_\-]?Q1\b", re.IGNORECASE), "-Q1"),
    # Q2 / 中位数 / 第二四分位数（注意：中位数可能不带 Q2 标记）
    (re.compile(r"[_\-]?第二四分位数?\s*(?:[\(（]\s*Q2\s*[\)）])?", re.IGNORECASE), "-中位数"),
    (re.compile(r"[_\-]?中位数\s*(?:[\(（]\s*Q2\s*[\)）])?", re.IGNORECASE), "-中位数"),
    (re.compile(r"[_\-]?Q2\b", re.IGNORECASE), "-中位数"),
    # Q3 / 上四分位数 / 第三四分位数（带或不带 Q3 后缀均可）
    (re.compile(r"[_\-]?(?:上|第三)四分位数?\s*(?:[\(（]\s*Q3\s*[\)）])?", re.IGNORECASE), "-Q3"),
    (re.compile(r"[_\-]?Q3\b", re.IGNORECASE), "-Q3"),
    # 中位数 / Median（英文）
    (re.compile(r"[_\-]?\bMedian\b", re.IGNORECASE), "-中位数"),
    # 最小值 / Min
    (re.compile(r"[_\-]?最小值", re.IGNORECASE), "-最小值"),
    (re.compile(r"[_\-]?\bMin\b", re.IGNORECASE), "-最小值"),
    # 最大值 / Max
    (re.compile(r"[_\-]?最大值", re.IGNORECASE), "-最大值"),
    (re.compile(r"[_\-]?\bMax\b", re.IGNORECASE), "-最大值"),
]


def normalize_triple_key(text: str, norm_logs: list[str] | None = None) -> str:
    """归一化三元组的 entity/header 文本

    主要处理箱线图场景下统计量表头的各种表述差异，将其统一为标准后缀形式。

    归一化规则：
        1. 箱线图统计量：将 "下四分位数(Q1)" / "第一四分位数" / "Q1" 等统一为 "-Q1" 后缀
           例如: "工资下四分位数(Q1)" -> "工资-Q1"
                 "第一四分位数"       -> "-Q1"
                 "salary-Q1"          -> "salary-Q1"（不变）
                 "最大群体规模-Q1"     -> "最大群体规模-Q1"（不变）
        2. 统计量在前、主题在后时，自动调换顺序为 "主题-统计量"
           例如: "第一四分位数-配送时长(分钟)" -> "配送时长(分钟)-Q1"
                 "最小值-工资"                -> "工资-最小值"
        3. 统计量被括号包裹时，自动去除括号
           例如: "len (Min)"  -> "len-最小值"
                 "len (Q1)"   -> "len-Q1"

    Args:
        text: 原始的 entity 或 header 文本
        norm_logs: 可选的日志列表，记录归一化前后的变化

    Returns:
        归一化后的文本
    """
    original = text = text.strip()
    if not text:
        return text

    # ---- 规则 0: 全角 → 半角标点归一化（：（），；％ -> :(),;% ）----
    text = _normalize_fullwidth(text)

    # ---- 规则 0.5: LaTeX 公式归一化 ----
    # 将 LaTeX 格式的数学表达式转为 Unicode 等价形式
    # 如 "$ m^{2} $" → "m²", "$ \text{元}/m^{2} $" → "元/m²"
    text = _normalize_latex(text)

    # ---- 规则 0.6: Unicode 字符归一化 ----
    # µ (U+00B5 MICRO SIGN) → μ (U+03BC GREEK SMALL LETTER MU)
    text = text.replace("\u00b5", "\u03bc")

    # ---- 规则 1: 箱线图统计量归一化 ----
    for pattern, replacement in _BOXPLOT_STAT_PATTERNS:
        match = pattern.search(text)
        if match:
            # 取匹配位置之前的部分作为前缀（主题在统计量前面的情况）
            # 清理尾部的分隔符和括号，如 "len (" -> "len"
            prefix = text[: match.start()].rstrip("-_— ").rstrip(" ").rstrip("(（").rstrip(" ")
            # 取匹配位置之后的部分作为后缀（主题在统计量后面的情况）
            # 清理头部的分隔符和括号，如 ")" -> ""
            suffix = text[match.end() :].lstrip("-_— ").lstrip(" ").lstrip(")）").lstrip(" ")

            if prefix and suffix:
                # 两边都有内容，统计量在中间，保持 prefix-统计量-suffix
                text = prefix + replacement + "-" + suffix
            elif prefix:
                # 标准情况：主题在前，统计量在后 -> "主题-统计量"
                text = prefix + replacement
            elif suffix:
                # 统计量在前，主题在后 -> 调换为 "主题-统计量"
                text = suffix + replacement
            else:
                # 只有统计量本身
                text = replacement

            break  # 只匹配第一个规则

    if text != original:
        logger.info(f"箱线图归一化: '{original}' -> '{text}'")
        if norm_logs is not None:
            norm_logs.append(f"归一化: '{original}' -> '{text}'")

    return text


def _is_empty_header(header: list[str]) -> bool:
    """判断 header 是否为无表头表格。

    以下两种情况均视为无表头：
    1. 所有列都为空（纯空表头）
    2. 仅第一列有值、其余列全为空（第一列只是行标签的列名，无实际意义）
    """
    if all(h.strip() == "" for h in header):
        return True
    # 第一列有值但其余列全空 → 也视为无表头
    if len(header) >= 2 and header[0].strip() != "" and all(h.strip() == "" for h in header[1:]):
        return True
    return False


def csv2triples(
    csv: str,
    separator="\\t",
    delimiter="\\n",
    norm_logs: list[str] | None = None,
) -> list[tuple[str, str, str]]:
    """
    将 CSV 字符串（以 \\t 分隔列，\\n 分隔行）解析为三元组 (entity, header, value)，其中 entity 和 header 会排序以消除顺序影响。

    特殊处理：
        - 如果 header 全为空，或仅第一列有值其余列全空（伪表头），则视为无表头表格，
          所有行（包括第一行）都按 key-value 对处理，三元组为 (col0, "", col1)，
          适用于两列无表头的 Markdown 表格。

    Args:
        csv: CSV 格式字符串
        separator: 列分隔符
        delimiter: 行分隔符
        norm_logs: 可选的日志列表，记录归一化操作
    """
    lines: list[str] = csv.strip().split(delimiter)
    header: list[str] = lines[0].split(separator)

    # 检测空表头：如果 header 全为空，则为无表头表格
    if _is_empty_header(header):
        logger.info("检测到空表头，按无表头模式解析")
        triples: list[tuple[str, str, str]] = []
        # 所有行（包括第一行）都作为数据行处理
        for line in lines:
            if not line:
                continue
            values: list[str] = line.split(separator)
            # 跳过全空行（即空表头行本身）
            if all(v.strip() == "" for v in values):
                continue
            # 跳过"仅第一列有值、其余列全空"的行（即伪表头行，如 "Platform\t"）
            if len(values) >= 2 and values[0].strip() != "" and all(v.strip() == "" for v in values[1:]):
                continue
            if len(values) >= 2:
                entity = values[0].strip()
                entity = normalize_triple_key(entity, norm_logs=norm_logs)
                # 对后续每一列都生成一个三元组
                # 不使用列索引作为 header，统一用空字符串，使三元组与列顺序无关
                for col_idx in range(1, len(values)):
                    value = _normalize_fullwidth(values[col_idx].strip()).replace("%", "").replace("$", "")
                    triples.append((entity, "", value))
            elif len(values) == 1 and values[0].strip():
                key = normalize_triple_key(values[0].strip(), norm_logs=norm_logs)
                triples.append((key, "", ""))
        return triples

    # 正常有表头模式
    triples: list[tuple[str, str, str]] = []
    for line in lines[1:]:
        if not line:
            continue
        values: list[str] = line.split(separator)
        entity: str = values[0]
        for i in range(1, len(values)):
            if i >= len(header):
                break
            # triples.append((entity, header[i], values[i]))
            # ---------------------------------------------------------
            value = _normalize_fullwidth(values[i].strip()).replace("%", "").replace("$", "")
            # 先归一化 entity 和 header，再排序（确保归一化后排序一致）
            norm_entity = normalize_triple_key(entity.strip(), norm_logs=norm_logs)
            norm_header = normalize_triple_key(header[i].strip(), norm_logs=norm_logs)
            key0, key1 = sorted([norm_entity, norm_header])
            triples.append((key0, key1, value))
            # ---------------------------------------------------------
    return triples


def process_triplets(triplets: list[tuple[str, str, str]]) -> list[tuple]:
    """
    标准化三元组：全部转小写，数值型的 value 转为 float
    """
    new_triplets = []
    for triplet in triplets:
        triplet_temp = []
        if len(triplet) > 2:
            if is_int(triplet[2]) or is_float(triplet[2]):
                triplet_temp = (triplet[0].lower(), triplet[1].lower(), float(triplet[2]))
            else:
                triplet_temp = (triplet[0].lower(), triplet[1].lower(), triplet[2].lower())
        else:
            triplet_temp = (triplet[0].lower(), triplet[1].lower(), "no meaning")
        new_triplets.append(triplet_temp)
    return new_triplets


def intersection_with_tolerance(
    a: list[tuple[str, str, Any]],
    b: list[tuple[str, str, Any]],
    tol_word: int,
    tol_num: float,
):
    """
    带容差的交集：文本部分用 Levenshtein 编辑距离容差（tol_word），数值部分用相对误差容差（tol_num）
    """
    a, b, c = set(a), set(b), set()
    for elem1 in a:
        for elem2 in b:
            if is_float(elem1[-1]) and is_float(elem2[-1]):
                if (Levenshtein.distance("".join(elem1[:-1]), "".join(elem2[:-1])) <= tol_word) and (
                    abs(elem1[-1] - elem2[-1]) / (elem2[-1] + 0.000001) <= tol_num
                ):
                    c.add(elem1)
            else:
                if Levenshtein.distance("".join([str(i) for i in elem1]), "".join([str(j) for j in elem2])) <= tol_word:
                    c.add(elem1)
    return list(c)


def union_with_tolerance(a: list[tuple[str, str, Any]], b: list[tuple[str, str, Any]], tol_word: int, tol_num: float):
    """
    基于容差交集计算的并集
    """
    c = set(a) | set(b)
    d = set(a) & set(b)
    e = intersection_with_tolerance(a, b, tol_word, tol_num)
    f = set(e)
    g = c - (f - d)
    return list(g)


# ============================================================
# 无表头对齐后处理
# ============================================================


# ============================================================
# 需要后处理过滤的三元组类别（关键词 + 标签）
# ============================================================
# 每个元素为 (关键词列表, 日志标签)，在 csv_eval 中依次过滤
_EXTRA_TRIPLE_FILTERS: list[tuple[list[str], str]] = [
    (["异常值", "outlier", "离群"], "异常值"),
    (["离散点", "离散值", "discrete", "scatter"], "离散点"),
]


def _make_keyword_triple_checker(keywords: list[str]) -> callable:
    """工厂函数：根据关键词列表生成三元组判断函数

    返回的函数检查三元组的前两个 key 是否包含任一关键词。

    Args:
        keywords: 关键词列表

    Returns:
        判断函数 (triple) -> bool
    """

    def _checker(triple: tuple) -> bool:
        text = str(triple[0]) + str(triple[1])
        return any(kw in text for kw in keywords)

    return _checker


def _postprocess_extra_triples(
    pred_triple_list: list[list[tuple]],
    label_triple_list: list[list[tuple]],
    is_target_triple: callable,
    tag: str,
    logs: list[str] | None = None,
    force_filter: bool = True,
) -> tuple[list[list[tuple]], list[list[tuple]]]:
    """通用后处理：如果 label 中完全没有某类三元组，则去掉 pred 中对应的三元组

    规则：
        - 对每一对 (pred, label)，分别统计目标类三元组数量
        - 如果 label 目标类数量为 0 且 pred 目标类数量 > 0：
            - 按行（entity）分组统计 pred 中目标类三元组数量，如果每个 entity 去掉的数量一致，
              则执行过滤并报 info
            - 如果每个 entity 去掉的数量不一致：
              - force_filter=True（默认）：仍然执行过滤，报 warning 记录数量分布
              - force_filter=False：不做任何修改，报 warning
        - 其他情况不做处理

    Args:
        pred_triple_list: 预测三元组列表
        label_triple_list: 标注三元组列表
        is_target_triple: 判断函数，接收一个三元组，返回是否属于目标类别
        tag: 日志标签（如 "离散点"）
        logs: 可选的日志列表
        force_filter: 当各 entity 的目标类三元组数量不一致时，是否仍然强制过滤（默认 True）

    Returns:
        处理后的 (pred_triple_list, label_triple_list)
    """
    new_pred_list = []
    for idx, (pred, label) in enumerate(zip(pred_triple_list, label_triple_list)):
        label_target_count = sum(1 for t in label if is_target_triple(t))
        pred_target_triples = [t for t in pred if is_target_triple(t)]
        pred_target_count = len(pred_target_triples)

        if label_target_count == 0 and pred_target_count > 0:
            from collections import Counter

            entity_target_counts = Counter()
            for t in pred_target_triples:
                # 判断哪个 key 包含目标关键词，另一个就是 entity
                # 构造只含单个 key 的伪三元组来检测
                if is_target_triple((str(t[0]), "", "")):
                    entity = str(t[1])
                else:
                    entity = str(t[0])
                entity_target_counts[entity] += 1

            unique_counts = set(entity_target_counts.values())
            if len(unique_counts) <= 1:
                filtered_pred = [t for t in pred if not is_target_triple(t)]
                msg = (
                    f"移除{pred_target_count}条{tag}三元组"
                    f"({len(entity_target_counts)}个entity×{list(unique_counts)[0] if unique_counts else 0})"
                )
                logger.info(f"[样本 {idx}] {msg}")
                if logs is not None:
                    logs.append(msg)
                new_pred_list.append(filtered_pred)
            else:
                if force_filter:
                    filtered_pred = [t for t in pred if not is_target_triple(t)]
                    msg = f"{tag}数量不一致({dict(entity_target_counts)})，强制移除{pred_target_count}条{tag}三元组"
                    logger.warning(f"[样本 {idx}] {msg}")
                    if logs is not None:
                        logs.append(msg)
                    new_pred_list.append(filtered_pred)
                else:
                    msg = f"{tag}数量不一致({dict(entity_target_counts)})，跳过过滤"
                    logger.warning(f"[样本 {idx}] {msg}")
                    if logs is not None:
                        logs.append(msg)
                    new_pred_list.append(pred)
        else:
            new_pred_list.append(pred)

    return new_pred_list, label_triple_list


def csv_eval(predictions: list[str], references: list[str], easy: Literal[0, 1], separator="\\t", delimiter="\\n"):
    predictions = np.asarray(predictions)
    labels = np.asarray(references)

    # 收集后处理日志（传回给调用方，而非仅 logger 输出）
    eval_logs: list[str] = []

    # ---- 0. CSV 层面的无表头对齐 ----
    # 三个分支：
    #   A. ref 无表头 + pred 有表头 → 去掉 pred 表头
    #   B. pred 无表头 + ref 有表头 → 把 ref 也归一化为空表头
    #   C. pred 比 ref 多一列"行头列"（ref 首列为空列名，pred 首列有列名，如 Year/date）
    #      → 去掉 pred 首列（把首列当作行头，与 ref 对齐）
    # 目的：让两边都以兼容模式比较，避免结构性错位导致 0 分。
    #
    # 解析注意：csv 格式形如 " \t A \t B \n ..."（分隔符前后带空格），首单元为空时
    # 字符串以 ' \t ' 开头（首字符是空格）。直接 `.strip().split(sep)` 会吃掉
    # 首空格，导致首 cell 的"空 token"丢失。这里用 `_parse_rows` 用 rstrip 只去
    # 末尾换行，首空单元被保留。
    def _parse_rows(csv_str: str) -> list[list[str]]:
        if not csv_str:
            return []
        trimmed = csv_str.rstrip("\n\r")
        if not trimmed.strip():
            return []
        return [ln.split(separator) for ln in trimmed.split(delimiter)]

    aligned_predictions = []
    aligned_labels = []
    for pred_csv, ref_csv in zip(predictions, labels):
        pred_rows = _parse_rows(pred_csv)
        ref_rows = _parse_rows(ref_csv)

        ref_header = ref_rows[0] if ref_rows else []
        pred_header = pred_rows[0] if pred_rows else []
        ref_headerless = _is_empty_header(ref_header) if ref_header else False
        pred_headerless = _is_empty_header(pred_header) if pred_header else False

        if ref_headerless and not pred_headerless and len(pred_rows) > 1:
            # 分支 A：ref 无表头，pred 有表头 -> 去掉 pred 的表头行
            num_cols = len(pred_header)
            empty_header = separator.join([" "] * num_cols)
            rest = pred_csv.rstrip("\n\r").split(delimiter)[1:]
            new_pred_csv = empty_header + delimiter + delimiter.join(rest)
            msg = "去掉pred表头行(ref无表头)"
            logger.info(msg)
            eval_logs.append(msg)
            aligned_predictions.append(new_pred_csv)
            aligned_labels.append(ref_csv)
        elif pred_headerless and not ref_headerless and len(ref_rows) > 1:
            # 分支 B：pred 无表头，ref 有表头 -> 把 ref 也改成空表头
            pred_data_rows = len(pred_rows) - 1
            ref_data_rows = len(ref_rows) - 1
            pred_ncols = len(pred_rows[1]) if len(pred_rows) > 1 else len(pred_header)
            ref_ncols = len(ref_rows[1]) if len(ref_rows) > 1 else len(ref_header)

            row_diff_ok = abs(pred_data_rows - ref_data_rows) <= 1
            col_match_ok = pred_ncols == ref_ncols and ref_ncols >= 2

            if row_diff_ok and col_match_ok:
                num_cols = len(ref_header)
                empty_header = separator.join([" "] * num_cols)
                rest = ref_csv.rstrip("\n\r").split(delimiter)[1:]
                new_ref_csv = empty_header + delimiter + delimiter.join(rest)
                msg = "去掉ref表头行(pred无表头, 行列数一致)"
                logger.info(msg)
                eval_logs.append(msg)
                aligned_predictions.append(pred_csv)
                aligned_labels.append(new_ref_csv)
            else:
                aligned_predictions.append(pred_csv)
                aligned_labels.append(ref_csv)
        else:
            # 分支 C：pred 有表头 + ref 有表头，但 pred 比 ref 多一列"行头列"。
            # 两种常见情况都覆盖：
            #   C1) pred 首列是"有名"的行头列（如 Year/Month/date/sex/Position），
            #       ref 首列为空列名：典型 pred "' \t Year \t A \t B'"，ref "' \t A \t B'"
            #   C2) pred 首列是 pandas 自动生成的"无名行号列"（数据都是 1,2,3,... 递增），
            #       典型 pred "' \t Year \t A \t B \n 1 \t 2013 \t ...'"，ref "' \t A \t B \n 2013 \t ...'"
            # 对这两种情况都：去掉 pred 首列（含表头行与每行首个单元），再做 triples 对比。
            try_branch_c = not pred_headerless and not ref_headerless and len(pred_rows) > 1 and len(ref_rows) > 1
            if try_branch_c:
                pred_data_rows = len(pred_rows) - 1
                ref_data_rows = len(ref_rows) - 1
                pred_ncols = len(pred_rows[1])
                ref_ncols = len(ref_rows[1])

                ref_first_col_empty = len(ref_header) > 0 and ref_header[0].strip() == ""

                row_diff_ok = abs(pred_data_rows - ref_data_rows) <= 1
                col_match_ok = (pred_ncols == ref_ncols + 1) and ref_ncols >= 2

                # 分支 C1：pred 首列列头非空（如 "Year"）
                pred_first_col_named = len(pred_header) > 0 and pred_header[0].strip() != ""

                # 分支 C2：pred 首列是 1-based 递增整数（pandas 的自动行号）
                def _is_auto_index_column() -> bool:
                    if len(pred_rows) < 2:
                        return False
                    # pred 首列列头为空
                    if len(pred_header) == 0 or pred_header[0].strip() != "":
                        return False
                    # 收集数据行首列
                    first_col_vals: list[str] = []
                    for r in pred_rows[1:]:
                        if not r:
                            continue
                        first_col_vals.append(r[0].strip())
                    if not first_col_vals:
                        return False
                    # 所有都是整数，且是 1,2,3,... 或 0,1,2,... 递增
                    try:
                        ints = [int(v) for v in first_col_vals]
                    except ValueError:
                        return False
                    if len(ints) < 2:
                        return False
                    # 允许 0-based 或 1-based
                    start = ints[0]
                    if start not in (0, 1):
                        return False
                    return ints == list(range(start, start + len(ints)))

                trigger_c1 = row_diff_ok and col_match_ok and ref_first_col_empty and pred_first_col_named
                trigger_c2 = row_diff_ok and col_match_ok and _is_auto_index_column()

                if trigger_c1 or trigger_c2:
                    raw_pred_lines = pred_csv.rstrip("\n\r").split(delimiter)
                    new_pred_lines = []
                    for ln in raw_pred_lines:
                        cells = ln.split(separator)
                        if len(cells) <= 1:
                            new_pred_lines.append(ln)
                        else:
                            new_pred_lines.append(separator.join(cells[1:]))
                    new_pred_csv = delimiter.join(new_pred_lines)
                    tag = "有名首列" if trigger_c1 else "pandas自动行号"
                    msg = f"去掉pred首列({tag}, 比 ref 多一列)"
                    logger.info(msg)
                    eval_logs.append(msg)
                    aligned_predictions.append(new_pred_csv)
                    aligned_labels.append(ref_csv)
                    continue

                # 分支 D：两边都是"2 列表格"（1 个行标签列 + 1 个数据列），
                # 即两边都只有一个非空 col_key，但这两个 col_key 的名字完全不同。
                # 例如：
                #   pred:  '  \t Values \n 2015 \t 157.4 \n 2016 \t 168.0'
                #   ref :  '  \t 北京新经济指数 \n 2015 \t 157.4 \n 2016 \t 168.9'
                # 列名对不上 → 三元组 (2015, Values, 157.4) 与 (2015, 北京新经济指数, 157.4)
                # 不匹配 → 0 分。
                #
                # 对此种"每边只有一个 col_key"的情况，把两边都改成空表头，
                # 让 csv2triples 进入无表头模式（triple 形如 (row_key, "", value)），
                # 仅比较行标签 + 数值。
                #
                # 严格限制：
                #   * pred/ref 都是 2 列（pred_ncols == ref_ncols == 2）
                #   * 行数大致匹配（|pred_data_rows - ref_data_rows| <= 1）
                #   * 两边都不在 headerless 状态（已由 try_branch_c 保证）
                #   * 两边首列列头相同语义地位（均非空或均为空），避免错位
                if pred_ncols == 2 and ref_ncols == 2 and row_diff_ok:
                    pred_first_empty = len(pred_header) > 0 and pred_header[0].strip() == ""
                    # 只要两边都有 1 个非空 col_key（第 2 列列头非空）
                    pred_col_named = len(pred_header) >= 2 and pred_header[1].strip() != ""
                    ref_col_named = len(ref_header) >= 2 and ref_header[1].strip() != ""
                    # 并且两边首列是否有名保持一致（都为空 or 都非空），减少错位
                    # 放宽：也允许 pred 首列非空但 ref 首列为空的情况
                    # （模型把行标签列的列头也填了，如 "Category | Value" vs " | Sales"）
                    first_col_sym = pred_first_empty == ref_first_col_empty
                    first_col_pred_named_ref_empty = (not pred_first_empty) and ref_first_col_empty

                    if pred_col_named and ref_col_named and (first_col_sym or first_col_pred_named_ref_empty):
                        empty_header_2 = separator.join([" "] * 2)
                        pred_rest = pred_csv.rstrip("\n\r").split(delimiter)[1:]
                        ref_rest = ref_csv.rstrip("\n\r").split(delimiter)[1:]
                        new_pred_csv = empty_header_2 + delimiter + delimiter.join(pred_rest)
                        new_ref_csv = empty_header_2 + delimiter + delimiter.join(ref_rest)
                        msg = (
                            f"抹空表头(2列表格, pred_col='{pred_header[1].strip()}' "
                            f"vs ref_col='{ref_header[1].strip()}')"
                        )
                        logger.info(msg)
                        eval_logs.append(msg)
                        aligned_predictions.append(new_pred_csv)
                        aligned_labels.append(new_ref_csv)
                        continue

            aligned_predictions.append(pred_csv)
            aligned_labels.append(ref_csv)

    predictions = np.asarray(aligned_predictions)
    labels = np.asarray(aligned_labels)

    # ---- 1. 解析三元组（只做一次，与容差无关） ----
    pred_triple_list: list[list[tuple]] = []
    for it in predictions:
        pred_triple_temp = csv2triples(it, separator=separator, delimiter=delimiter, norm_logs=eval_logs)
        pred_triple_list.append(process_triplets(pred_triple_temp))

    label_triple_list: list[list[tuple]] = []
    for it in labels:
        label_triple_temp = csv2triples(it, separator=separator, delimiter=delimiter, norm_logs=eval_logs)
        label_triple_list.append(process_triplets(label_triple_temp))

    # ---- 1.2 后处理：表头公共子串对齐 ----
    # 场景 A（公共前缀 / 箱线图）：ref 表头为 "检测结果-最小值", "检测结果-Q1" 等，
    #       pred 归一化后为 "-最小值", "-Q1" 等（缺少公共前缀）。
    # 场景 B（公共后缀）：ref 表头为 "MSFT-Monthly return (%)", "V-Monthly return (%)" 等，
    #       pred 表头为 "msft", "v" 等（缺少公共后缀）。
    # 统一处理：从 ref 的 key 中提取公共前缀/后缀，补到 pred 对应的 key 上。
    _BOXPLOT_SUFFIXES = {"-最小值", "-q1", "-中位数", "-q3", "-最大值"}

    for idx in range(len(pred_triple_list)):
        pred_triples = pred_triple_list[idx]
        label_triples = label_triple_list[idx]
        changed_triples = False

        # ---- 场景 A：箱线图公共前缀对齐 ----
        ref_prefixes: set[str] = set()
        for t in label_triples:
            for key in (t[0], t[1]):
                for suffix in _BOXPLOT_SUFFIXES:
                    if key.endswith(suffix) and len(key) > len(suffix):
                        ref_prefixes.add(key[: -len(suffix)])

        if len(ref_prefixes) == 1:
            common_prefix = ref_prefixes.pop()
            has_bare_suffix = any(t[0] in _BOXPLOT_SUFFIXES or t[1] in _BOXPLOT_SUFFIXES for t in pred_triples)
            if has_bare_suffix:
                new_pred = []
                for k0, k1, v in pred_triples:
                    c = False
                    if k0 in _BOXPLOT_SUFFIXES:
                        k0 = common_prefix + k0
                        c = True
                    if k1 in _BOXPLOT_SUFFIXES:
                        k1 = common_prefix + k1
                        c = True
                    if c:
                        k0, k1 = sorted([k0, k1])
                    new_pred.append((k0, k1, v))
                pred_triples = new_pred
                changed_triples = True
                msg = f"箱线图表头前缀对齐: pred补上公共前缀'{common_prefix}'"
                logger.info(msg)
                eval_logs.append(msg)

        # ---- 场景 B：通用公共后缀对齐 ----
        # 收集 ref 中所有非数值、非空的 key（排除纯数字行标签）
        ref_keys: set[str] = set()
        for t in label_triples:
            for key in (t[0], t[1]):
                if key and not key.replace(".", "").replace("-", "").isdigit():
                    ref_keys.add(key)
        # 找 ref key 中的最长公共后缀（以 "-" 或分隔符为边界）
        if len(ref_keys) >= 2:
            ref_key_list = sorted(ref_keys)
            # 逐字符从尾部比较，找最长公共后缀
            min_len = min(len(k) for k in ref_key_list)
            common_suffix_len = 0
            for i in range(1, min_len + 1):
                if all(k[-i] == ref_key_list[0][-i] for k in ref_key_list):
                    common_suffix_len = i
                else:
                    break
            common_suffix = ref_key_list[0][-common_suffix_len:] if common_suffix_len > 0 else ""
            # 后缀必须以分隔符开头（"-"、"_"、" "），确保是有意义的子串边界
            common_suffix = common_suffix.lstrip()
            if common_suffix and common_suffix[0] in ("-", "_"):
                # 提取 ref 中去掉后缀后的前缀部分
                ref_stems = {k[: len(k) - common_suffix_len].rstrip("-_ ") for k in ref_keys}
                # 检查 pred 中是否存在这些 stem 但缺少后缀
                pred_keys: set[str] = set()
                for t in pred_triples:
                    for key in (t[0], t[1]):
                        if key and not key.replace(".", "").replace("-", "").isdigit():
                            pred_keys.add(key)
                # pred 的 key 恰好是 ref 去掉后缀后的 stem（且 pred 中没有带后缀的完整 key）
                if ref_stems and pred_keys and pred_keys <= ref_stems and not (pred_keys & ref_keys):
                    new_pred = []
                    for k0, k1, v in pred_triples:
                        c = False
                        if k0 in ref_stems and k0 not in ref_keys:
                            k0 = k0 + common_suffix
                            c = True
                        if k1 in ref_stems and k1 not in ref_keys:
                            k1 = k1 + common_suffix
                            c = True
                        if c:
                            k0, k1 = sorted([k0, k1])
                        new_pred.append((k0, k1, v))
                    pred_triples = new_pred
                    changed_triples = True
                    msg = f"表头后缀对齐: pred补上公共后缀'{common_suffix}'"
                    logger.info(msg)
                    eval_logs.append(msg)

        # ---- 场景 C：通用公共前缀对齐 ----
        # ref 列头为 "Fluorescence intensity(a.u.)-10μM", "Fluorescence intensity(a.u.)-5μM" 等，
        # pred 列头为 "10μm", "5μm" 等（缺少公共前缀）。
        # 从 ref 的 key 中提取公共前缀（以 "-" 为边界），检查 pred 的 key 是否是去掉前缀后的部分。
        if not changed_triples:
            ref_keys_c: set[str] = set()
            for t in label_triples:
                for key in (t[0], t[1]):
                    if key and not key.replace(".", "").replace("-", "").isdigit():
                        ref_keys_c.add(key)
            if len(ref_keys_c) >= 2:
                ref_key_list_c = sorted(ref_keys_c)
                # 找公共前缀（逐字符从头比较）
                min_len_c = min(len(k) for k in ref_key_list_c)
                common_prefix_len = 0
                for i in range(min_len_c):
                    if all(k[i] == ref_key_list_c[0][i] for k in ref_key_list_c):
                        common_prefix_len = i + 1
                    else:
                        break
                common_prefix = ref_key_list_c[0][:common_prefix_len] if common_prefix_len > 0 else ""
                # 前缀必须以分隔符结尾（"-"、"_"、" "），确保是有意义的子串边界
                common_prefix = common_prefix.rstrip()
                if common_prefix and common_prefix[-1] in ("-", "_"):
                    # 提取 ref 中去掉前缀后的后缀部分
                    ref_suffixes = {k[common_prefix_len:].lstrip("-_ ") for k in ref_keys_c}
                    # 检查 pred 中是否存在这些 suffix 但缺少前缀
                    pred_keys_c: set[str] = set()
                    for t in pred_triples:
                        for key in (t[0], t[1]):
                            if key and not key.replace(".", "").replace("-", "").isdigit():
                                pred_keys_c.add(key)
                    # pred 的 key 恰好是 ref 去掉前缀后的 suffix
                    if ref_suffixes and pred_keys_c and pred_keys_c <= ref_suffixes and not (pred_keys_c & ref_keys_c):
                        new_pred = []
                        for k0, k1, v in pred_triples:
                            c = False
                            if k0 in ref_suffixes and k0 not in ref_keys_c:
                                k0 = common_prefix + k0
                                c = True
                            if k1 in ref_suffixes and k1 not in ref_keys_c:
                                k1 = common_prefix + k1
                                c = True
                            if c:
                                k0, k1 = sorted([k0, k1])
                            new_pred.append((k0, k1, v))
                        pred_triples = new_pred
                        changed_triples = True
                        msg = f"表头前缀对齐: pred补上公共前缀'{common_prefix}'"
                        logger.info(msg)
                        eval_logs.append(msg)

        if changed_triples:
            pred_triple_list[idx] = pred_triples

    # ---- 1.5 后处理：过滤 label 中不存在的异常值 / 离散点等三元组 ----
    if POST_PROCESS_TRIPLES:
        for keywords, tag in _EXTRA_TRIPLE_FILTERS:
            checker = _make_keyword_triple_checker(keywords)
            pred_triple_list, label_triple_list = _postprocess_extra_triples(
                pred_triple_list, label_triple_list, checker, tag, logs=eval_logs
            )

    # ---- 2. 定义容差参数（每种容差只需计算一次 sim_list） ----
    tolerance_params: dict[str, tuple[int, float]] = {
        "strict": (0, 0 if easy == 1 else 0.1),
        "slight": (2, 0.05 if easy == 1 else 0.3),
        "high": (5, 0.1 if easy == 1 else 0.5),
    }

    def _compute_sim_list(tol_word: int, tol_num: float) -> list[float]:
        """给定容差参数，计算每条数据的相似度（交集/并集）"""
        sim_list: list[float] = []
        for pred, label in zip(pred_triple_list, label_triple_list):
            intersection = intersection_with_tolerance(pred, label, tol_word=tol_word, tol_num=tol_num)
            union = union_with_tolerance(pred, label, tol_word=tol_word, tol_num=tol_num)
            sim = len(intersection) / len(union) if len(union) > 0 else 0.0
            sim_list.append(sim)
        return sim_list

    # ---- 3. 每种容差只调用一次 _compute_sim_list ----
    sim_lists: dict[str, list[float]] = {}
    for tol_name, (tol_word, tol_num) in tolerance_params.items():
        sim_lists[tol_name] = _compute_sim_list(tol_word, tol_num)

    # ---- 4. 基于已缓存的 sim_list，快速计算各阈值下的 AP ----
    def _get_ap(sim_list: list[float], sim_threshold: float) -> float:
        """AP@θ = 相似度 ≥ θ 的样本数 / 总样本数"""
        return len([s for s in sim_list if s >= sim_threshold]) / len(sim_list)

    # mAP: 在 0.5~0.95（步长0.05，共10个阈值）上取平均
    map_strict = 0.0
    map_slight = 0.0
    map_high = 0.0
    for sim_threshold in np.arange(0.5, 1, 0.05):
        map_strict += _get_ap(sim_lists["strict"], sim_threshold) / 10
        map_slight += _get_ap(sim_lists["slight"], sim_threshold) / 10
        map_high += _get_ap(sim_lists["high"], sim_threshold) / 10

    # 单阈值 AP
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
    eval_logs = dedup_logs(eval_logs)

    return scores, eval_logs

