"""Markdown 多级无序列表的评估模块

实现基于路径集合的相似度匹配（Path-based Set Similarity）：
1. 将 Markdown 无序列表解析为树结构
2. 将树展平为"从根节点到每个节点"的路径集合（包含中间节点，不只是叶子）
3. 用 Levenshtein Ratio 计算两两路径相似度，构建得分矩阵
4. 用匈牙利算法求最大二分图匹配
5. 最终得分 = 匹配对的相似度总和 / max(pred路径数, ref路径数)
"""

import logging
import re
from typing import Literal

import Levenshtein
import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# ============================================================
# Markdown 无序列表检测
# ============================================================


def is_markdown_list(text: str) -> bool:
    """判断文本是否为 Markdown 多级无序列表格式

    检测规则：
        - 至少有 2 行以 '- ' 或 '  - '（缩进 + '-'）开头
        - 不是 Markdown 表格（不含 | 分隔符行）

    Args:
        text: 待检测的文本

    Returns:
        是否为 Markdown 无序列表
    """
    if not text or not text.strip():
        return False

    # 预处理：去掉可能的 <pFig>...</pFig><quad>...</quad> 标签行
    lines = text.strip().split("\n")
    list_line_count = 0

    for line in lines:
        stripped = line.rstrip()
        # 跳过空行
        if not stripped:
            continue
        # 跳过 HTML 标签行（如 <pFig>...</pFig>）
        if re.match(r"^\s*<[^>]+>", stripped):
            continue
        # 检测无序列表行：可能有前导空格/制表符，然后是 '- '
        if re.match(r"^(\s*)- ", stripped):
            list_line_count += 1

    return list_line_count >= 2


# ============================================================
# Markdown 无序列表解析
# ============================================================


def _parse_indent_level(line: str) -> tuple[int, str]:
    """解析一行的缩进级别和内容

    支持空格缩进（每 2 个空格一级）和制表符缩进。

    Args:
        line: 原始行文本

    Returns:
        (缩进级别, 去掉缩进和 '- ' 前缀后的内容)
    """
    # 计算前导空格数
    stripped = line.rstrip()
    content_start = 0
    spaces = 0
    for ch in stripped:
        if ch == " ":
            spaces += 1
            content_start += 1
        elif ch == "\t":
            spaces += 2  # 制表符算 2 个空格
            content_start += 1
        else:
            break

    # 去掉 '- ' 前缀
    rest = stripped[content_start:]
    if rest.startswith("- "):
        content = rest[2:]
    elif rest.startswith("-"):
        content = rest[1:].lstrip()
    else:
        content = rest

    # 缩进级别：每 2 个空格一级
    level = spaces // 2
    return level, content.strip()


def parse_markdown_list(text: str) -> list[dict]:
    """将 Markdown 多级无序列表解析为树结构

    树结构用嵌套字典表示：
        {"name": "节点名", "children": [子节点...]}

    Args:
        text: Markdown 无序列表文本

    Returns:
        根节点列表（可能有多个顶级节点）
    """
    lines = text.strip().split("\n")

    # 过滤掉空行和标签行，只保留列表行
    list_lines: list[tuple[int, str]] = []
    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            continue
        # 跳过 HTML 标签行
        if re.match(r"^\s*<[^>]+>", stripped):
            continue
        # 只处理列表行
        if re.match(r"^(\s*)- ", stripped) or re.match(r"^(\s*)-\S", stripped):
            level, content = _parse_indent_level(stripped)
            if content:
                list_lines.append((level, content))

    if not list_lines:
        return []

    # 构建树：使用栈来跟踪当前的父节点链
    roots: list[dict] = []
    # stack: [(level, node_dict), ...]
    stack: list[tuple[int, dict]] = []

    for level, content in list_lines:
        node = {"name": content, "children": []}

        # 弹出栈中级别 >= 当前级别的节点（找到父节点）
        while stack and stack[-1][0] >= level:
            stack.pop()

        if stack:
            # 有父节点，添加为子节点
            parent = stack[-1][1]
            parent["children"].append(node)
        else:
            # 没有父节点，是根节点
            roots.append(node)

        stack.append((level, node))

    return roots


# ============================================================
# 树归一化：单子节点合并 / 分隔符拆分
# ============================================================

# 用于检测节点名称中的分隔符（——、—、->、--、-、:、：等）
# 注意：只匹配作为"连接符"使用的分隔符，前后都有非空内容
# 注意：匹配顺序很重要，长的分隔符要放在短的前面（如 -> 在 - 前面）
_SEPARATOR_PATTERN = re.compile(
    r"(?<=\S)"  # 前面有非空字符
    r"\s*"  # 可选空格
    r"(?:——|—|->|-->|--|-|:|：)"  # 分隔符（含箭头）
    r"\s*"  # 可选空格
    r"(?=\S)"  # 后面有非空字符
)


def normalize_tree(roots: list[dict]) -> list[dict]:
    """归一化树结构，使单子节点合并和分隔符拆分的写法等效

    归一化规则（双向）：
        1. 合并：如果一个节点只有一个子节点，将父子合并为一个节点
           （名称用 "——" 连接），子节点的 children 提升为合并节点的 children
           例如：A -> B -> [C, D]  变为  A——B -> [C, D]

        2. 拆分：如果一个节点名称中包含分隔符（——、—、-、-- 等），
           且该节点没有子节点或只有子节点，尝试拆分为父子关系
           例如：A——B -> [C, D]  保持不变（已经是合并形式）
                 A——B（无子节点）变为  A -> B

    为了使两种写法等效，统一策略是：**总是合并单子节点**。
    这样无论原始写法是分开的还是合并的，归一化后的树结构一致。

    具体流程：
        1. 先拆分：将含分隔符的节点名拆分为父子链
        2. 再合并：将只有一个子节点的节点与子节点合并

    Args:
        roots: 根节点列表

    Returns:
        归一化后的根节点列表（新的树，不修改原始数据）
    """
    # 深拷贝，避免修改原始数据
    import copy

    roots = copy.deepcopy(roots)

    # 第一步：拆分含分隔符的节点
    roots = [_split_separator_nodes(root) for root in roots]

    # 第二步：合并单子节点
    roots = [_merge_single_child_nodes(root) for root in roots]

    return roots


def _split_separator_nodes(node: dict) -> dict:
    """递归拆分含分隔符的节点名

    如果节点名包含分隔符（如 "A——B" 或 "A -> B"），拆分为父子链：
        A -> B -> [原有 children]

    支持的分隔符：——、—、->、-->、--、-、:、：
    支持多级拆分："A——B——C" -> A -> B -> C -> [原有 children]

    Args:
        node: 树节点

    Returns:
        拆分后的（可能是新的）根节点
    """
    # 先递归处理子节点
    node["children"] = [_split_separator_nodes(child) for child in node["children"]]

    # 尝试拆分当前节点名
    name = node["name"]
    parts = _SEPARATOR_PATTERN.split(name)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) <= 1:
        # 没有分隔符或拆分后只有一部分，不需要拆分
        return node

    # 拆分为链：parts[0] -> parts[1] -> ... -> parts[-1] -> [原有 children]
    # 从最后一个 part 开始构建
    current = {"name": parts[-1], "children": node["children"]}
    for i in range(len(parts) - 2, 0, -1):
        current = {"name": parts[i], "children": [current]}

    # 最顶层节点
    return {"name": parts[0], "children": [current]}


def _merge_single_child_nodes(node: dict) -> dict:
    """递归合并只有一个子节点的节点

    如果一个节点只有一个子节点，将父子合并：
        名称 = "父名——子名"，children = 子节点的 children

    持续合并直到节点有 0 个或 2+ 个子节点。

    Args:
        node: 树节点

    Returns:
        合并后的节点
    """
    # 先递归处理子节点
    node["children"] = [_merge_single_child_nodes(child) for child in node["children"]]

    # 合并单子节点
    while len(node["children"]) == 1:
        child = node["children"][0]
        node["name"] = node["name"] + "——" + child["name"]
        node["children"] = child["children"]

    return node


# ============================================================
# 树 → 路径集合（包含中间节点路径）
# ============================================================


def tree_to_paths(roots: list[dict]) -> list[tuple[str, ...]]:
    """将树结构展平为路径集合

    包含从根到每个节点（包括中间节点）的路径，不只是叶子节点。

    例如对于树：
        - A
          - B
            - C
          - D

    生成的路径集合为：
        ("A",)
        ("A", "B")
        ("A", "B", "C")
        ("A", "D")

    Args:
        roots: 根节点列表

    Returns:
        路径元组列表，每个路径是从根到某节点的名称元组
    """
    paths: list[tuple[str, ...]] = []

    def _dfs(node: dict, current_path: tuple[str, ...]):
        new_path = current_path + (node["name"],)
        paths.append(new_path)
        for child in node["children"]:
            _dfs(child, new_path)

    for root in roots:
        _dfs(root, ())

    return paths


# ============================================================
# 路径相似度计算
# ============================================================


def _path_similarity(path_a: tuple[str, ...], path_b: tuple[str, ...]) -> float:
    """计算两条路径的相似度

    策略：
        1. 将路径拼接为字符串（用 " -> " 连接）
        2. 用 Levenshtein Ratio 计算归一化相似度（0~1）

    Args:
        path_a: 路径 A
        path_b: 路径 B

    Returns:
        相似度分数（0~1）
    """
    str_a = " -> ".join(path_a).lower()
    str_b = " -> ".join(path_b).lower()
    return Levenshtein.ratio(str_a, str_b)


# ============================================================
# 匈牙利算法匹配打分
# ============================================================


def _hungarian_matching_score(pred_paths: list[tuple[str, ...]], ref_paths: list[tuple[str, ...]]) -> float:
    """使用匈牙利算法计算路径集合的最优匹配相似度

    构建相似度矩阵，用匈牙利算法求最大匹配，
    最终得分 = 匹配对的相似度总和 / max(pred路径数, ref路径数)

    Args:
        pred_paths: 预测路径集合
        ref_paths: 参考路径集合

    Returns:
        匹配相似度分数（0~1）
    """
    if not pred_paths and not ref_paths:
        return 1.0
    if not pred_paths or not ref_paths:
        return 0.0

    n_pred = len(pred_paths)
    n_ref = len(ref_paths)

    # 构建相似度矩阵
    sim_matrix = np.zeros((n_pred, n_ref))
    for i, p_path in enumerate(pred_paths):
        for j, r_path in enumerate(ref_paths):
            sim_matrix[i, j] = _path_similarity(p_path, r_path)

    # 匈牙利算法求最大匹配（linear_sum_assignment 求最小，所以用 cost = 1 - sim）
    cost_matrix = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 计算匹配对的相似度总和
    matched_sim_sum = sim_matrix[row_ind, col_ind].sum()

    # 归一化：除以 max(pred路径数, ref路径数)
    score = matched_sim_sum / max(n_pred, n_ref)
    return score


# ============================================================
# 带容差的匹配打分（对应 SCRM 的 strict/slight/high 三档）
# ============================================================


def _hungarian_matching_score_with_threshold(
    pred_paths: list[tuple[str, ...]],
    ref_paths: list[tuple[str, ...]],
    sim_threshold: float = 0.0,
) -> float:
    """带相似度阈值的匈牙利匹配打分

    只有相似度 >= sim_threshold 的匹配对才计入得分。

    Args:
        pred_paths: 预测路径集合
        ref_paths: 参考路径集合
        sim_threshold: 相似度阈值，低于此值的匹配对不计入

    Returns:
        匹配相似度分数（0~1）
    """
    if not pred_paths and not ref_paths:
        return 1.0
    if not pred_paths or not ref_paths:
        return 0.0

    n_pred = len(pred_paths)
    n_ref = len(ref_paths)

    # 构建相似度矩阵
    sim_matrix = np.zeros((n_pred, n_ref))
    for i, p_path in enumerate(pred_paths):
        for j, r_path in enumerate(ref_paths):
            sim_matrix[i, j] = _path_similarity(p_path, r_path)

    # 匈牙利算法求最大匹配
    cost_matrix = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 只计入相似度 >= 阈值的匹配对
    matched_sim_sum = 0.0
    for r, c in zip(row_ind, col_ind):
        if sim_matrix[r, c] >= sim_threshold:
            matched_sim_sum += sim_matrix[r, c]

    score = matched_sim_sum / max(n_pred, n_ref)
    return score


# ============================================================
# 主评估函数
# ============================================================


def tree_eval(
    predictions: list[str],
    references: list[str],
    easy: Literal[0, 1],
) -> tuple:
    """对 Markdown 无序列表进行路径集合匹配评估

    与 csv_eval 输出格式完全一致，返回 13 个指标。

    评估流程：
        1. 解析 pred 和 ref 的 Markdown 无序列表为树结构
        2. 将树展平为路径集合（包含中间节点路径）
        3. 对每种容差级别，用匈牙利算法计算匹配相似度
        4. 计算 mAP 和各阈值下的 AP

    容差映射（对应 SCRM 的 strict/slight/high）：
        - strict: 路径相似度阈值 = 1.0（完全匹配）
        - slight: 路径相似度阈值 = 0.8（允许小差异）
        - high:   路径相似度阈值 = 0.5（允许较大差异）

    Args:
        predictions: 预测文本列表（Markdown 无序列表）
        references: 参考文本列表（Markdown 无序列表）
        easy: 难度级别（1=简单，0=困难）

    Returns:
        与 csv_eval 相同的 13 元组
    """
    # ---- 1. 解析路径集合 ----
    pred_paths_list: list[list[tuple[str, ...]]] = []
    ref_paths_list: list[list[tuple[str, ...]]] = []

    for pred_text, ref_text in zip(predictions, references):
        # 解析 pred
        pred_roots = parse_markdown_list(pred_text)
        pred_roots = normalize_tree(pred_roots)
        pred_paths = tree_to_paths(pred_roots)

        # 解析 ref
        ref_roots = parse_markdown_list(ref_text)
        ref_roots = normalize_tree(ref_roots)
        ref_paths = tree_to_paths(ref_roots)

        pred_paths_list.append(pred_paths)
        ref_paths_list.append(ref_paths)

        logger.info(f"Pred 路径数: {len(pred_paths)}, Ref 路径数: {len(ref_paths)}")
        if logger.isEnabledFor(logging.DEBUG):
            for p in pred_paths:
                logger.debug(f"  Pred 路径: {' -> '.join(p)}")
            for p in ref_paths:
                logger.debug(f"  Ref 路径: {' -> '.join(p)}")

    # ---- 2. 定义容差参数 ----
    # 对于树匹配，容差体现在路径相似度的阈值上
    # strict: 要求路径几乎完全一致
    # slight: 允许小差异（如标点、空格）
    # high:   允许较大差异（如同义词、缩写）
    tolerance_params: dict[str, float] = {
        "strict": 1.0 if easy == 1 else 0.95,
        "slight": 0.85 if easy == 1 else 0.75,
        "high": 0.6 if easy == 1 else 0.5,
    }

    # ---- 3. 计算每种容差下的相似度列表 ----
    def _compute_sim_list(path_sim_threshold: float) -> list[float]:
        """给定路径相似度阈值，计算每条数据的整体匹配分数"""
        sim_list: list[float] = []
        for pred_paths, ref_paths in zip(pred_paths_list, ref_paths_list):
            score = _hungarian_matching_score_with_threshold(pred_paths, ref_paths, sim_threshold=path_sim_threshold)
            sim_list.append(score)
        return sim_list

    sim_lists: dict[str, list[float]] = {}
    for tol_name, threshold in tolerance_params.items():
        sim_lists[tol_name] = _compute_sim_list(threshold)

    # ---- 4. 计算 mAP 和 AP ----
    def _get_ap(sim_list: list[float], sim_threshold: float) -> float:
        """AP@θ = 相似度 ≥ θ 的样本数 / 总样本数"""
        if not sim_list:
            return 0.0
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
    return scores, []
