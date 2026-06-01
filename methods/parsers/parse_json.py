"""JSON → 内部 CSV / Markdown 列表；邻接表 CSV → Markdown 列表。"""

import csv
import io
import json
import re
from typing import Any

from ..context import strip_code_fence


def _safe_json_loads(text: str) -> Any | None:
    """安全解析 JSON，失败返回 None"""
    if not text or not text.strip():
        return None
    t = strip_code_fence(text, "json").strip()
    try:
        return json.loads(t)
    except Exception:
        # 尝试从文本中截取 {...} 或 [...] 段
        m = re.search(r"(\{.*\}|\[.*\])", t, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
    return None


def json_to_internal_csv(text: str) -> str:
    """将数值类图表的 JSON 输出转换为内部 CSV (\\t/\\n 分隔) 格式。

    支持两种常见结构：

    1) 两层嵌套 (类别->维度->值)：
        {"values": {
            "类别A": {"维度1": v1, "维度2": v2},
            "类别B": {"维度1": v3}
        }}
        -> 行=维度，列=类别

    2) 三层嵌套 (外层分组->行键->列键->值)，常见于箱线图：
        {"values": {
            "最大群体规模": {
                "两百五十万年前之前": {"最小值": 0, "中位数": 7},
                "两百五十万年前之后": {"最小值": 0, "中位数": 14}
            }
        }}
        -> 行=外层键下的中层键，列="外层键-内层键"

    若无法识别，返回空字符串。
    """
    obj = _safe_json_loads(text)
    if obj is None:
        return ""

    # 定位 values 字典
    values: dict | None = None
    if isinstance(obj, dict):
        if "values" in obj and isinstance(obj["values"], dict):
            values = obj["values"]
        else:
            if obj and all(isinstance(v, dict) for v in obj.values()):
                values = obj

    if not values or not isinstance(values, dict):
        # 兑底：扁平 kv 字典
        if isinstance(obj, dict) and obj:
            rows = [" \\t ".join([str(k).strip(), str(v).strip()]) for k, v in obj.items()]
            header = " \\t ".join(["key", "value"])
            return header + " \\n " + " \\n ".join(rows)
        return ""

    # 判断是两层还是三层
    # 三层：values[k1][k2] 仍然是 dict
    is_three_level = False
    for v1 in values.values():
        if isinstance(v1, dict):
            for v2 in v1.values():
                if isinstance(v2, dict):
                    is_three_level = True
                    break
            if is_three_level:
                break

    if is_three_level:
        # 三层嵌套：外层键=outer, 中层键=行, 内层键=列（附 outer-前缀）
        # 可能有多个 outer，合并所有 outer 的行/列
        # 输出表：行=所有 (outer, middle_key) 组合中的 middle_key（同名合并）
        # 列=所有 (outer, inner_key)
        row_keys: list[str] = []
        row_seen: set = set()
        col_keys: list[str] = []  # 格式: "outer-inner"
        col_seen: set = set()

        # data[(row_key, col_label)] = value
        data: dict[tuple, str] = {}

        for outer_key, outer_val in values.items():
            if not isinstance(outer_val, dict):
                continue
            for middle_key, middle_val in outer_val.items():
                if middle_key not in row_seen:
                    row_seen.add(middle_key)
                    row_keys.append(middle_key)
                if isinstance(middle_val, dict):
                    for inner_key, v in middle_val.items():
                        col_label = f"{outer_key}-{inner_key}"
                        if col_label not in col_seen:
                            col_seen.add(col_label)
                            col_keys.append(col_label)
                        data[(middle_key, col_label)] = str(v).strip()
                else:
                    # 并非三层，退化为用 outer_key 做列
                    col_label = str(outer_key)
                    if col_label not in col_seen:
                        col_seen.add(col_label)
                        col_keys.append(col_label)
                    data[(middle_key, col_label)] = str(middle_val).strip()

        if not row_keys or not col_keys:
            return ""

        header = [""] + col_keys
        rows_text = [" \\t ".join(c.strip() for c in header)]
        for rk in row_keys:
            cells = [str(rk).strip()]
            for ck in col_keys:
                cells.append(data.get((rk, ck), ""))
            rows_text.append(" \\t ".join(cells))
        return " \\n ".join(rows_text)

    # 两层嵌套：原逻辑
    dimensions: list[str] = []
    seen = set()
    for cat_values in values.values():
        if not isinstance(cat_values, dict):
            continue
        for dim in cat_values.keys():
            if dim not in seen:
                seen.add(dim)
                dimensions.append(dim)

    categories = [str(c) for c in values.keys()]
    if not dimensions or not categories:
        return ""

    header_cells = [""] + categories
    rows_text = [" \\t ".join(cell.strip() for cell in header_cells)]
    for dim in dimensions:
        row_cells = [str(dim).strip()]
        for cat in categories:
            cat_values = values.get(cat)
            if isinstance(cat_values, dict) and dim in cat_values:
                row_cells.append(str(cat_values[dim]).strip())
            else:
                row_cells.append("")
        rows_text.append(" \\t ".join(row_cells))

    return " \\n ".join(rows_text)


def json_tree_to_markdown_list(text: str) -> str:
    """将逻辑结构图的 JSON 嵌套树转换为 Markdown 多级无序列表。

    输入 JSON 结构期望：
        {"name": "根", "children": [
            {"name": "子1", "children": [...]},
            {"name": "子2"}
        ]}

    若根节点没有 `name`/`children` 结构，也兼容 {str: list|dict} 形式。
    """
    obj = _safe_json_loads(text)
    if obj is None:
        return ""

    lines: list[str] = []

    def _dump_node(node: Any, level: int) -> None:
        indent = "  " * level
        if isinstance(node, dict):
            # 标准 {name, children} 形式
            if "name" in node:
                name = str(node.get("name", "")).strip()
                lines.append(f"{indent}- {name}")
                children = node.get("children", [])
                if isinstance(children, list):
                    for child in children:
                        _dump_node(child, level + 1)
                elif isinstance(children, dict):
                    for k, v in children.items():
                        _dump_node({"name": k, "children": v}, level + 1)
            else:
                # 兼容 {str: sub} 形式
                for k, v in node.items():
                    lines.append(f"{indent}- {str(k).strip()}")
                    if isinstance(v, (dict, list)):
                        _dump_node(v, level + 1)
                    elif v is not None and str(v).strip():
                        lines.append(f"{indent}  - {str(v).strip()}")
        elif isinstance(node, list):
            for item in node:
                _dump_node(item, level)
        else:
            if node is not None and str(node).strip():
                lines.append(f"{indent}- {str(node).strip()}")

    _dump_node(obj, 0)
    return "\n".join(lines)


# ============================================================
# 邻接表 CSV（id,parent_id,name） → Markdown 无序列表
# ============================================================


def adjacency_csv_to_markdown_list(text: str) -> str:
    """将 `id,parent_id,name` 三列 CSV 转换为 Markdown 多级无序列表。

    约定：parent_id == 0 或 parent_id 为空 的节点为根节点。
    """
    if not text or not text.strip():
        return ""

    t = strip_code_fence(text, "csv").strip()

    try:
        reader = csv.DictReader(io.StringIO(t))
        fieldnames = [fn.strip().lower() for fn in (reader.fieldnames or [])]
        if not fieldnames or not {"id", "parent_id", "name"}.issubset(set(fieldnames)):
            # 尝试无表头：按顺序解析
            reader2 = csv.reader(io.StringIO(t))
            rows_raw = [r for r in reader2 if any(c.strip() for c in r)]
            # 跳过可能的表头
            if rows_raw and any(c.strip().lower() in {"id", "parent_id", "name"} for c in rows_raw[0]):
                rows_raw = rows_raw[1:]
            rows = []
            for r in rows_raw:
                if len(r) < 3:
                    continue
                rows.append({"id": r[0].strip(), "parent_id": r[1].strip(), "name": r[2].strip()})
        else:
            rows = []
            for r in reader:
                rows.append(
                    {
                        "id": str(r.get("id", "")).strip(),
                        "parent_id": str(r.get("parent_id", "")).strip(),
                        "name": str(r.get("name", "")).strip(),
                    }
                )
    except Exception:
        return ""

    if not rows:
        return ""

    # 构造邻接表
    id_to_node: dict[str, dict] = {r["id"]: r for r in rows if r["id"]}
    children_map: dict[str, list[str]] = {r["id"]: [] for r in rows if r["id"]}
    roots: list[str] = []

    for r in rows:
        nid = r["id"]
        pid = r["parent_id"]
        if not nid:
            continue
        if pid in ("", "0", "-1", "None", "null"):
            roots.append(nid)
        elif pid in id_to_node:
            children_map.setdefault(pid, []).append(nid)
        else:
            # 父节点未定义：作为根处理
            roots.append(nid)

    if not roots:
        # 若没识别出根，把没有对应 parent 的都作为根
        all_ids = set(id_to_node.keys())
        child_ids = set()
        for children in children_map.values():
            child_ids.update(children)
        roots = [nid for nid in all_ids if nid not in child_ids]

    lines: list[str] = []
    visited: set[str] = set()

    def _dfs(nid: str, level: int) -> None:
        if nid in visited:
            return
        visited.add(nid)
        node = id_to_node.get(nid)
        if not node:
            return
        indent = "  " * level
        lines.append(f"{indent}- {node['name']}")
        for child_id in children_map.get(nid, []):
            _dfs(child_id, level + 1)

    for root_id in roots:
        _dfs(root_id, 0)

    return "\n".join(lines)


# ============================================================
# Python 代码 → 内部 CSV / Markdown 无序列表
# ============================================================
