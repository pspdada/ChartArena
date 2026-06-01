r"""Mermaid 流程图解析与评估模块

功能：
1. 检测文本是否为 mermaid 流程图格式
2. 将 mermaid 流程图解析为有向图（节点 + 带 edge_label 的边）
3. 将 mermaid 流程图转换为 Markdown 无序列表（用于与思维导图统一评分）
4. 两个 mermaid 流程图之间的相似度评估（基于边集合 + 匈牙利匹配）

评估思路（流程图 vs 流程图）：
    - 将两个 mermaid 解析为有向图，每条边表示为 (src_label, dst_label, edge_label)
    - 边相似度 = (sim(src) + sim(dst) + sim(edge_label)) / 3
        * sim 使用 Levenshtein Ratio（大小写不敏感）
        * 两条边 edge_label 都为空时 sim(edge_label)=1（不惩罚）
    - 节点相似度：用 Levenshtein Ratio 比较节点 label
    - 综合得分 = 0.6 * 边匹配分 + 0.4 * 节点匹配分（边更重要，因为它包含结构信息）
    - 用匈牙利算法求最优匹配

支持的边语法（含 pipe 形式的 edge_label，属于 mermaid 标准语法）：
    A --> B             A ==> B             A -.-> B         A --- B
    A -->|text| B       A ==>|text| B       A -.->|text| B
    A -- text --> B     A == text ==> B
    A -- "text" --> B   A == "text" ==>
    A --> B --> C       (链式)
    A -->|是| B -->|否| C (链式 + 标签)

支持的节点形状（形状本身不参与打分，只用于 label 提取）：
    A[label]            方形
    A(label)            圆角
    A((label))          圆形（双圆括号）
    A([label])          体育场/跑道
    A[[label]]          子程序
    A[(label)]          圆柱体/数据库
    A{label}            菱形
    A{{label}}          六边形
    A>label]            旗帜
    A[/label/]          平行四边形
    A[\label\]          反向平行四边形
    均支持 "label" 引号 / 无引号两种写法。
"""

import logging
import re
from collections import defaultdict
from typing import Literal

import Levenshtein
import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# ============================================================
# Mermaid 检测
# ============================================================


def is_mermaid(text: str) -> bool:
    """判断文本是否为 mermaid 流程图格式

    检测规则：
        - 包含 flowchart / graph 关键词（TD/LR/TB/RL/BT 等方向）
        - 包含 --> / --- / ==> 等箭头连接符

    Args:
        text: 待检测的文本

    Returns:
        是否为 mermaid 流程图
    """
    if not text or not text.strip():
        return False

    # 去掉 ```mermaid ... ``` 包裹
    cleaned = _strip_mermaid_fence(text)

    lines = cleaned.strip().split("\n")
    has_graph_decl = False
    has_arrow = False

    for line in lines:
        stripped = line.strip()
        # 检测 flowchart / graph 声明
        if re.match(r"^(flowchart|graph)\s+(TD|TB|LR|RL|BT)\b", stripped, re.IGNORECASE):
            has_graph_decl = True
        # 检测箭头连接
        if re.search(r"-->|---|\.\->|==>|--\s*\w+\s*-->", stripped):
            has_arrow = True

    return has_graph_decl and has_arrow


def _strip_mermaid_fence(text: str) -> str:
    """去掉 ```mermaid ... ``` 代码块包裹

    Args:
        text: 可能包含代码块标记的文本

    Returns:
        去掉代码块标记后的文本
    """
    text = text.strip()
    # 去掉开头的 ```mermaid
    text = re.sub(r"^```\s*mermaid\s*\n?", "", text, flags=re.IGNORECASE)
    # 去掉结尾的 ```
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


# ============================================================
# Mermaid 解析
# ============================================================


def _clean_node_label(label: str) -> str:
    """对 mermaid 节点 label 做温和归一化，用于节点/边匹配。

    只做两类"明显是书写差异"的规整：
      1. 换行符号类：`<br>` / `<br/>` / `<BR>` / 字面 `\\n` / 真实换行 / 制表符 → 单空格；
         （mermaid 里的 `<br>` 和 PlantUML 的 `\\n` 语义都是"换行"，不应拉开节点 label 距离）
      2. 连续空白 → 单空格，首尾空白 strip。

    ⚠️ 不做大小写归一、不删标点 —— strict 模式仍保持对 label 内容的严格区分。
    """
    if not label:
        return ""
    # <br> / <br/> / <br /> 各种写法 → 空格（大小写不敏感）
    s = re.sub(r"<\s*br\s*/?\s*>", " ", label, flags=re.IGNORECASE)
    # 字面换行转义 `\n` / `\r` / `\l`
    s = re.sub(r"\\[nrl]", " ", s)
    # 真实换行 / 制表符
    s = re.sub(r"[\n\r\t]+", " ", s)
    # 连续空白合并
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _parse_node_def(token: str) -> tuple[str, str]:
    """解析节点定义，提取 ID 和 label

    支持的 mermaid 节点形状（修 Bug ②：新增 (( )) / [[ ]] / [( )] 等）：
        A["label"]   A[label]            # 方形
        A("label")   A(label)            # 圆角
        A(("label")) A((label))          # 圆形（新增：双圆括号）
        A((["label"])) A(([label]))      # 体育场/跑道（新增）
        A[["label"]] A[[label]]          # 子程序（新增）
        A[("label")] A[(label)]          # 圆柱体/数据库（新增）
        A{"label"}   A{label}            # 菱形
        A{{"label"}} A{{label}}          # 六边形
        A>"label"]   A>label]            # 不对称（旗帜）
        A[/"label"/] A[/label/]          # 平行四边形
        A[\\"label"\\] A[\\label\\]      # 反向平行四边形
        纯 ID（无括号）

    Args:
        token: 节点定义字符串

    Returns:
        (node_id, label)
    """
    token = token.strip()
    if not token:
        return "", ""

    # 通用模式：ID + 括号包裹的内容。
    # ⚠️ 匹配顺序很关键：更长/更嵌套的形状（如 (( )) / ([ ]) / {{ }} / [[ ]] / [( )]）
    #   必须放在更短形状（(...) / [...] / {...}）前面，避免被短形状贪婪先匹配掉。
    m = re.match(
        r"^([A-Za-z_]\w*)\s*"  # 节点 ID
        r"(?:"
        # ---- 带双引号的形状 ----
        r'\(\(\s*"([^"]*)"\s*\)\)'  # (( "label" ))   圆形
        r'|\(\s*\[\s*"([^"]*)"\s*\]\s*\)'  # (( "label" ))   (体育场) —— 原有
        r'|\[\s*\[\s*"([^"]*)"\s*\]\s*\]'  # [[ "label" ]]   子程序
        r'|\[\s*\(\s*"([^"]*)"\s*\)\s*\]'  # [( "label" )]   圆柱体
        r'|\{\s*\{\s*"([^"]*)"\s*\}\s*\}'  # {{ "label" }}   六边形
        r'|\[\s*"([^"]*)"\s*\]'  # [ "label" ]     方形
        r'|\(\s*"([^"]*)"\s*\)'  # ( "label" )     圆角
        r'|\{\s*"([^"]*)"\s*\}'  # { "label" }     菱形
        r'|>\s*"([^"]*)"\s*\]'  # > "label" ]     旗帜
        r'|\[\s*/\s*"([^"]*)"\s*/\s*\]'  # [ /"label"/ ]   平行四边形
        r'|\[\s*\\\s*"([^"]*)"\s*\\\s*\]'  # [ \"label"\ ]   反向平行四边形
        # ---- 无引号的形状 ----
        r"|\(\(\s*([^)]*?)\s*\)\)"  # (( label ))     圆形
        r"|\(\s*\[\s*([^\]]*?)\s*\]\s*\)"  # ([ label ])     体育场
        r"|\[\s*\[\s*([^\]]*?)\s*\]\s*\]"  # [[ label ]]     子程序
        r"|\[\s*\(\s*([^)]*?)\s*\)\s*\]"  # [( label )]     圆柱体
        r"|\{\s*\{\s*([^}]*?)\s*\}\s*\}"  # {{ label }}     六边形
        r"|\[\s*([^\]]*)\s*\]"  # [ label ]       方形
        r"|\(\s*([^)]*)\s*\)"  # ( label )       圆角
        r"|\{\s*([^}]*)\s*\}"  # { label }       菱形
        r"|>\s*([^\]]*)\s*\]"  # > label ]       旗帜
        r")?$",
        token,
    )

    if m:
        node_id = m.group(1)
        # 找到第一个非 None 的捕获组作为 label
        label = None
        if m.lastindex and m.lastindex >= 2:
            for i in range(2, m.lastindex + 1):
                if m.group(i) is not None:
                    label = m.group(i).strip()
                    break
        if label is None:
            label = node_id  # 没有括号，label 就是 ID 本身
        return node_id, _clean_node_label(label)

    # 纯 ID（无括号）
    m_id = re.match(r"^([A-Za-z_]\w*)$", token)
    if m_id:
        return m_id.group(1), m_id.group(1)

    return token, _clean_node_label(token)


def parse_mermaid(text: str) -> tuple[dict[str, str], list[tuple[str, str]], list[tuple[str, str, str]]]:
    """解析 mermaid 流程图为有向图

    Args:
        text: mermaid 流程图文本

    Returns:
        (nodes, edges, labeled_edges)
        - nodes: {node_id: label}
        - edges: [(src_id, dst_id), ...]
        - labeled_edges: [(src_id, dst_id, edge_label), ...] 带标签的边
    """
    cleaned = _strip_mermaid_fence(text)
    lines = cleaned.strip().split("\n")

    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    labeled_edges: list[tuple[str, str, str]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # 跳过 flowchart/graph 声明行
        if re.match(r"^(flowchart|graph)\s+(TD|TB|LR|RL|BT)\b", stripped, re.IGNORECASE):
            continue
        # 跳过 subgraph / end / style / classDef 等
        if re.match(r"^(subgraph|end|style|classDef|class|click|linkStyle)\b", stripped, re.IGNORECASE):
            continue
        # 跳过注释
        if stripped.startswith("%%"):
            continue

        # 尝试解析连接关系: A --> B, A -- text --> B, A ==> B 等
        # 支持链式连接: A --> B --> C
        edge_match = _parse_edge_line(stripped)
        if edge_match:
            for src_token, dst_token, edge_label in edge_match:
                src_id, src_label = _parse_node_def(src_token)
                dst_id, dst_label = _parse_node_def(dst_token)
                if src_id:
                    nodes.setdefault(src_id, src_label)
                if dst_id:
                    nodes.setdefault(dst_id, dst_label)
                if src_id and dst_id:
                    edges.append((src_id, dst_id))
                    labeled_edges.append((src_id, dst_id, edge_label or ""))
            continue

        # 尝试解析纯节点定义行: A["label"]
        node_id, label = _parse_node_def(stripped)
        if node_id:
            nodes.setdefault(node_id, label)

    return nodes, edges, labeled_edges


def _parse_edge_line(line: str) -> list[tuple[str, str, str]] | None:
    """解析一行中的边连接关系

    支持：
        A --> B
        A -- "text" --> B
        A -- text --> B
        A -->|text| B               ← 新增：pipe 形式边标签（mermaid 常见）
        A ==>|text| B
        A -.->|text| B
        A ==> B
        A -.-> B
        A --- B
        A --> B --> C               (链式)
        A -->|是| B -->|否| C      (链式 + 带标签)

    Args:
        line: 一行 mermaid 文本

    Returns:
        [(src_token, dst_token, edge_label), ...] 或 None（非边定义行）
    """
    # 箭头模式：
    # - 优先匹配带 edge_label 的长模式（|label|、"label" 或 -- label -->）
    # - 其次匹配无标签的 -->/==>/-.-> 等
    # 注意 pipe 形式必须放在最前，否则 --> 会先把 `A-->` 吃掉，`|是| B` 留下来被当节点。
    arrow_patterns = [
        # ---- pipe 形式（新增，修 Bug ①） ----
        r'-->\s*\|\s*"([^"]*)"\s*\|',  # --> |"text"|
        r"-->\s*\|([^|]*)\|",  # --> |text|
        r'==>\s*\|\s*"([^"]*)"\s*\|',  # ==> |"text"|
        r"==>\s*\|([^|]*)\|",  # ==> |text|
        r'-\.->\s*\|\s*"([^"]*)"\s*\|',  # -.-> |"text"|
        r"-\.->\s*\|([^|]*)\|",  # -.-> |text|
        # ---- 传统带 edge_label 形式 ----
        r'==\s*"([^"]*)"\s*==>',  # == "text" ==>
        r'--\s*"([^"]*)"\s*-->',  # -- "text" -->
        r"--\s*([^->\s][^->]*?)\s*-->",  # -- text -->
        r"==\s*([^=>\s][^=>]*?)\s*==>",  # == text ==>
        # ---- 无标签 ----
        r"==>",
        r"-->",
        r"-\.->",
        r"---",
    ]

    # 检查是否包含任何箭头
    if not re.search(r"-->|---|==>|-\.->|\.\->", line):
        return None

    results: list[tuple[str, str, str]] = []

    # 逐步扫描：每一轮找到最靠左的箭头，截出前面的 node token + 该箭头的 label，
    # 然后从箭头末尾继续。这样 `A -->|是| B -->|否| C` 也能被正确切分。
    parts: list[str] = []
    labels: list[str] = []
    remaining = line

    # 防止病态正则导致死循环：限制最多匹配 256 次
    safety = 256
    while remaining and safety > 0:
        safety -= 1
        best_match = None
        best_pos = len(remaining)
        best_label = ""

        for pattern in arrow_patterns:
            m = re.search(pattern, remaining)
            if m and m.start() < best_pos:
                best_match = m
                best_pos = m.start()
                # 提取边标签（若该 pattern 有捕获组）
                best_label = ""
                for g in m.groups():
                    if g is not None:
                        best_label = g.strip()
                        break

        if best_match:
            before = remaining[: best_match.start()].strip()
            if before:
                parts.append(before)
            labels.append(best_label)
            remaining = remaining[best_match.end() :].strip()
        else:
            if remaining.strip():
                parts.append(remaining.strip())
            break

    # 构建边
    if len(parts) >= 2:
        for i in range(len(parts) - 1):
            edge_label = labels[i] if i < len(labels) else ""
            results.append((parts[i], parts[i + 1], edge_label))

    return results if results else None


# ============================================================
# Mermaid → Markdown 无序列表转换
# ============================================================


def mermaid_to_markdown_list(text: str) -> str:
    """将 mermaid 流程图转换为 Markdown 多级无序列表

    转换策略：
        1. 解析 mermaid 为有向图（节点 + 边）
        2. 找到根节点（入度为 0 的节点）
        3. 从根节点开始 DFS，生成缩进的无序列表

    注意：
        - 如果有环，会在访问过的节点处停止（避免无限递归）
        - 如果有多个根节点，依次输出
        - 孤立节点（无边连接）作为顶级节点输出

    Args:
        text: mermaid 流程图文本

    Returns:
        Markdown 多级无序列表文本
    """
    nodes, edges, _ = parse_mermaid(text)

    if not nodes:
        return ""

    # 构建邻接表和入度表
    children: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = defaultdict(int)

    for node_id in nodes:
        in_degree.setdefault(node_id, 0)

    for src, dst in edges:
        children[src].append(dst)
        in_degree[dst] = in_degree.get(dst, 0) + 1

    # 找根节点（入度为 0）
    roots = [nid for nid in nodes if in_degree.get(nid, 0) == 0]

    # 如果没有根节点（全是环），取第一个节点
    if not roots:
        roots = [list(nodes.keys())[0]]

    # DFS 生成无序列表
    lines: list[str] = []
    visited: set[str] = set()

    def _dfs(node_id: str, depth: int):
        if node_id in visited:
            return
        visited.add(node_id)
        indent = "  " * depth
        label = nodes.get(node_id, node_id)
        lines.append(f"{indent}- {label}")
        for child_id in children.get(node_id, []):
            _dfs(child_id, depth + 1)

    for root_id in roots:
        _dfs(root_id, 0)

    # 处理孤立节点（未被访问到的）
    for node_id in nodes:
        if node_id not in visited:
            label = nodes.get(node_id, node_id)
            lines.append(f"- {label}")

    return "\n".join(lines)


# ============================================================
# 流程图评估（有向图相似度）
# ============================================================


def _edge_similarity(edge_a: tuple, edge_b: tuple) -> float:
    """计算两条有向边的相似度（考虑 src / dst / edge_label 三者）

    修 Bug ④ + B5：
    - 边视为 (src_label, dst_label, edge_label) 三元组，三者地位相等；
    - src / dst / edge_label 分别做 Levenshtein ratio 再平均，避免
      把 "src -> dst" 拼成整串后短 label 的差异被稀释；
    - 若两边 edge_label 都为空：edge_label 部分视为 1.0（不惩罚）；
      若仅一边为空：按正常 ratio（空串对非空串得 0）处理。

    Args:
        edge_a: (src_label, dst_label[, edge_label]) 2 元或 3 元
        edge_b: (src_label, dst_label[, edge_label])

    Returns:
        相似度分数（0~1）
    """
    src_a, dst_a = edge_a[0], edge_a[1]
    src_b, dst_b = edge_b[0], edge_b[1]
    lab_a = edge_a[2] if len(edge_a) >= 3 else ""
    lab_b = edge_b[2] if len(edge_b) >= 3 else ""

    # 比较前统一做一次换行类符号归一化（`<br>` / `\n` / 连续空白 → 单空格），
    # 避免 pred/ref 在书写习惯上的差异拉低相似度。
    src_a = _clean_node_label(src_a or "")
    dst_a = _clean_node_label(dst_a or "")
    src_b = _clean_node_label(src_b or "")
    dst_b = _clean_node_label(dst_b or "")
    lab_a = _clean_node_label(lab_a or "")
    lab_b = _clean_node_label(lab_b or "")

    src_sim = Levenshtein.ratio(src_a.lower(), src_b.lower())
    dst_sim = Levenshtein.ratio(dst_a.lower(), dst_b.lower())
    if not lab_a and not lab_b:
        lab_sim = 1.0  # 两边都没有 edge label，不作惩罚
    else:
        lab_sim = Levenshtein.ratio(lab_a.lower(), lab_b.lower())

    return (src_sim + dst_sim + lab_sim) / 3.0


def _node_similarity(label_a: str, label_b: str) -> float:
    """计算两个节点 label 的相似度

    Args:
        label_a: 节点 A 的 label
        label_b: 节点 B 的 label

    Returns:
        相似度分数（0~1）
    """
    # 比较前先做一次换行类符号归一化，保护 strict 匹配的“内容一致性”语义
    # 不被 `<br>` / `\n` / 连续空白 这些书写差异干扰。
    return Levenshtein.ratio(_clean_node_label(label_a).lower(), _clean_node_label(label_b).lower())


def _hungarian_matching_score(
    pred_items: list,
    ref_items: list,
    sim_func: callable,
    sim_threshold: float = 0.0,
) -> float:
    """通用匈牙利匹配打分

    Args:
        pred_items: 预测项列表
        ref_items: 参考项列表
        sim_func: 相似度函数 (a, b) -> float
        sim_threshold: 相似度阈值，低于此值的匹配对不计入

    Returns:
        匹配相似度分数（0~1）
    """
    if not pred_items and not ref_items:
        return 1.0
    if not pred_items or not ref_items:
        return 0.0

    n_pred = len(pred_items)
    n_ref = len(ref_items)

    # 构建相似度矩阵
    sim_matrix = np.zeros((n_pred, n_ref))
    for i, p in enumerate(pred_items):
        for j, r in enumerate(ref_items):
            sim_matrix[i, j] = sim_func(p, r)

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


def flowchart_similarity(
    pred_text: str,
    ref_text: str,
    edge_weight: float = 0.6,
    node_weight: float = 0.4,
    sim_threshold: float = 0.0,
) -> float:
    """计算两个流程图的相似度

    综合得分 = edge_weight * 边匹配分 + node_weight * 节点匹配分

    Args:
        pred_text: 预测的 mermaid 文本
        ref_text: 参考的 mermaid 文本
        edge_weight: 边匹配权重
        node_weight: 节点匹配权重
        sim_threshold: 相似度阈值

    Returns:
        综合相似度分数（0~1）
    """
    # 使用 labeled_edges，把 edge_label（例如 `是 / 否 / Yes / No`）也纳入相似度。
    pred_nodes, _, pred_labeled_edges = parse_mermaid(pred_text)
    ref_nodes, _, ref_labeled_edges = parse_mermaid(ref_text)

    # 将边的 ID 转换为 label，得到 (src_label, dst_label, edge_label) 三元组
    pred_edge_labels = [(pred_nodes.get(s, s), pred_nodes.get(d, d), lab or "") for s, d, lab in pred_labeled_edges]
    ref_edge_labels = [(ref_nodes.get(s, s), ref_nodes.get(d, d), lab or "") for s, d, lab in ref_labeled_edges]

    # 节点 label 列表
    pred_node_labels = list(pred_nodes.values())
    ref_node_labels = list(ref_nodes.values())

    # 边匹配分
    edge_score = _hungarian_matching_score(pred_edge_labels, ref_edge_labels, _edge_similarity, sim_threshold)

    # 节点匹配分
    node_score = _hungarian_matching_score(pred_node_labels, ref_node_labels, _node_similarity, sim_threshold)

    # 如果没有边（只有节点），全部权重给节点
    if not pred_edge_labels and not ref_edge_labels:
        return node_score

    return edge_weight * edge_score + node_weight * node_score


# ============================================================
# 主评估函数
# ============================================================


def flowchart_eval(
    predictions: list[str],
    references: list[str],
    easy: Literal[0, 1],
) -> tuple:
    """对 mermaid 流程图进行有向图相似度评估

    与 csv_eval / tree_eval 输出格式完全一致，返回 13 个指标 + eval_logs。

    容差映射：
        - strict: sim_threshold = 1.0 / 0.95
        - slight: sim_threshold = 0.85 / 0.75
        - high:   sim_threshold = 0.6 / 0.5

    Args:
        predictions: 预测文本列表（mermaid 流程图）
        references: 参考文本列表（mermaid 流程图）
        easy: 难度级别（1=简单，0=困难）

    Returns:
        (scores_tuple, eval_logs)
    """
    eval_logs: list[str] = []

    # 容差参数（与 tree_eval 一致）
    tolerance_params: dict[str, float] = {
        "strict": 1.0 if easy == 1 else 0.95,
        "slight": 0.85 if easy == 1 else 0.75,
        "high": 0.6 if easy == 1 else 0.5,
    }

    def _compute_sim_list(sim_threshold: float) -> list[float]:
        sim_list: list[float] = []
        for pred_text, ref_text in zip(predictions, references):
            score = flowchart_similarity(pred_text, ref_text, sim_threshold=sim_threshold)
            sim_list.append(score)
        return sim_list

    sim_lists: dict[str, list[float]] = {}
    for tol_name, threshold in tolerance_params.items():
        sim_lists[tol_name] = _compute_sim_list(threshold)

    # 计算 mAP 和 AP
    def _get_ap(sim_list: list[float], sim_threshold: float) -> float:
        if not sim_list:
            return 0.0
        return len([s for s in sim_list if s >= sim_threshold]) / len(sim_list)

    map_strict = 0.0
    map_slight = 0.0
    map_high = 0.0
    for sim_threshold in np.arange(0.5, 1, 0.05):
        map_strict += _get_ap(sim_lists["strict"], sim_threshold) / 10
        map_slight += _get_ap(sim_lists["slight"], sim_threshold) / 10
        map_high += _get_ap(sim_lists["high"], sim_threshold) / 10

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
    return scores, eval_logs
