"""SE_CODE：Python 源码 → 内部 CSV / Markdown 列表。"""

import ast
import csv
import io
import re
from typing import Any

from ..context import get_chart_type


def _extract_python_code(text: str) -> str:
    """从模型输出中提取 Python 代码，关键辅助函数。

    优先匹配 ```python ... ```；其次任意围栏；都无则尝试提取
    TinyChart 等模型的 ``<step>...</step>`` agent 片段；最终兜底返回原文。
    """
    if not text:
        return ""
    t = text.strip()
    m = re.search(r"```[ \t]*(?:python|py|PYTHON)[ \t]*\n?(.*?)```", t, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```[a-zA-Z0-9_\-]*[ \t]*\n?(.*?)```", t, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # TinyChart agent 风格：<comment>...</comment>\n<step>Y=[...]</step>\n<step>X=['a','b']</step>
    # 此时原文含有 `<comment>` / `<step>` 这类非 Python 语法标签，ast.parse 必失败。
    # 抽取所有 <step>...</step> 里的内容，按行拼接成合法代码供下游 literal 收集。
    if "<step>" in t and "</step>" in t:
        steps = re.findall(r"<step>(.*?)</step>", t, flags=re.DOTALL)
        if steps:
            code = "\n".join(s.strip() for s in steps if s.strip())
            if code:
                return code
    return t


def _literal_to_python(node: ast.AST) -> Any:
    """将 ast 节点转为 Python 字面量值；不支持的返回 None。

    支持：Constant（str/int/float/bool/None）、List/Tuple/Dict 嵌套、UnaryOp (-x)、
    以及 np.array([...]) / numpy.array([...]) 类形式的 Call。
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _literal_to_python(node.operand)
        if isinstance(inner, (int, float)):
            return -inner if isinstance(node.op, ast.USub) else inner
        return None
    if isinstance(node, (ast.List, ast.Tuple)):
        out = []
        for elt in node.elts:
            v = _literal_to_python(elt)
            out.append(v)
        return out
    if isinstance(node, ast.Dict):
        out = {}
        for k_node, v_node in zip(node.keys, node.values):
            k = _literal_to_python(k_node) if k_node is not None else None
            v = _literal_to_python(v_node)
            if isinstance(k, (str, int, float, bool)):
                out[str(k)] = v
        return out
    if isinstance(node, ast.Call):
        # np.array([...])、list([...]) 等：尝试取第一个字面量参数
        func_name = ""
        f = node.func
        if isinstance(f, ast.Attribute):
            func_name = f.attr
        elif isinstance(f, ast.Name):
            func_name = f.id
        if func_name in {"array", "asarray", "list", "tuple", "Series"} and node.args:
            return _literal_to_python(node.args[0])
        return None
    if isinstance(node, ast.Name):
        # 没办法解析变量引用，返回 None
        return None
    # ListComp: 支持 [expr for var in range(...)] / [expr for var in [literal_list]]
    if isinstance(node, ast.ListComp):
        return _eval_simple_list_comp(node)
    # JoinedStr (f-string): 只支持常量部分，变量部分留空
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                inner = _literal_to_python(v.value)
                if inner is not None:
                    parts.append(str(inner))
                else:
                    return None  # 含未知变量的 f-string
            else:
                return None
        return "".join(parts)
    return None


def _eval_simple_list_comp(node: ast.ListComp) -> list | None:
    """对 [expr for var in iterable] 这种最简单的列表推导式求值。

    支持的 iterable：
        - range(n) / range(a, b) / range(a, b, step)
        - 字面量 List/Tuple

    支持的 expr：
        - Constant / Name (仅当引用循环变量)
        - JoinedStr (f-string) 使用循环变量
        - 二元运算（如 i+1）、UnaryOp
    不支持则返回 None。
    """
    # 只处理单层 generator、单变量、无 if 过滤
    if len(node.generators) != 1:
        return None
    gen = node.generators[0]
    if gen.ifs:
        return None
    if not isinstance(gen.target, ast.Name):
        return None
    loop_var = gen.target.id

    # 解析 iterable
    iterable_vals: list
    iter_node = gen.iter
    # 形式 range(...)
    if isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name) and iter_node.func.id == "range":
        try:
            arg_vals = [_literal_to_python(a) for a in iter_node.args]
        except Exception:
            return None
        if any(a is None or not isinstance(a, (int, float)) for a in arg_vals):
            return None
        arg_ints = [int(a) for a in arg_vals]
        if len(arg_ints) == 1:
            iterable_vals = list(range(arg_ints[0]))
        elif len(arg_ints) == 2:
            iterable_vals = list(range(arg_ints[0], arg_ints[1]))
        elif len(arg_ints) == 3:
            iterable_vals = list(range(arg_ints[0], arg_ints[1], arg_ints[2]))
        else:
            return None
    # 形式 字面量 list/tuple
    elif isinstance(iter_node, (ast.List, ast.Tuple)):
        resolved = _literal_to_python(iter_node)
        if not isinstance(resolved, (list, tuple)):
            return None
        iterable_vals = list(resolved)
    else:
        return None

    # 评估 element 表达式：用循环变量代入
    out: list = []
    for v in iterable_vals:
        val = _eval_expr_with_binding(node.elt, {loop_var: v})
        if val is None:
            return None
        out.append(val)
    return out


def _eval_expr_with_binding(expr: ast.AST, bindings: dict) -> Any:
    """在有限绑定下对表达式求值（仅 Constant / Name / JoinedStr / BinOp / UnaryOp）。"""
    if isinstance(expr, ast.Constant):
        return expr.value
    if isinstance(expr, ast.Name):
        return bindings.get(expr.id)
    if isinstance(expr, ast.UnaryOp):
        inner = _eval_expr_with_binding(expr.operand, bindings)
        if isinstance(inner, (int, float)):
            if isinstance(expr.op, ast.USub):
                return -inner
            if isinstance(expr.op, ast.UAdd):
                return inner
        return None
    if isinstance(expr, ast.BinOp):
        left = _eval_expr_with_binding(expr.left, bindings)
        right = _eval_expr_with_binding(expr.right, bindings)
        if left is None or right is None:
            return None
        try:
            if isinstance(expr.op, ast.Add):
                return left + right
            if isinstance(expr.op, ast.Sub):
                return left - right
            if isinstance(expr.op, ast.Mult):
                return left * right
            if isinstance(expr.op, ast.Mod):
                return left % right
        except Exception:
            return None
        return None
    if isinstance(expr, ast.JoinedStr):
        parts: list[str] = []
        for v in expr.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                inner = _eval_expr_with_binding(v.value, bindings)
                if inner is None:
                    return None
                parts.append(str(inner))
            else:
                return None
        return "".join(parts)
    return None


def _collect_code_literals(code: str) -> dict[str, Any]:
    """解析 Python 代码，收集所有顶层 / 函数内的赋值字面量。

    返回 {变量名: 字面量值}

    若 ast.parse 失败（例如 pred 被 max_tokens 截断），回退到 regex 兜底，
    抢救形如 ``name = [ ... ]`` / ``name = { ... }`` 的简单字面量赋值。
    """
    result: dict[str, Any] = {}
    if not code or not code.strip():
        return result
    try:
        tree = ast.parse(code)
    except Exception:
        return _regex_collect_literals(code)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            # 获取目标变量名
            targets = []
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        targets.append(t.id)
            else:
                if isinstance(node.target, ast.Name):
                    targets.append(node.target.id)
            if not targets:
                continue
            value = _literal_to_python(node.value) if node.value is not None else None
            if value is None:
                continue
            for name in targets:
                # 只收集列表/元组/字典/标量/字符串
                if isinstance(value, (list, tuple, dict, str, int, float, bool)):
                    # 后赋值覆盖前赋值（必要的简化）
                    result[name] = value
    return result


def _regex_collect_literals(code: str) -> dict[str, Any]:
    """AST 兜底：用正则抓取 ``name = [ ... ]`` / ``name = { ... }`` 这种
    顶层简单字面量赋值，用于 pred 被截断或存在少量语法错误的情况。

    仅处理：
      * 方括号包裹的列表（含数值 / 字符串 / 嵌套一层列表）
      * 大括号包裹的字典（键值均为字面量）
    对每个匹配片段，再用 ast.parse 解析该 **单表达式**，解析失败则整个跳过。
    为兼容截断末尾，会尝试在剩余内容末尾截断不完整的元素。
    """
    import ast as _ast

    out: dict[str, Any] = {}
    if not code:
        return out

    # 匹配 "name = [" 或 "name = {" 开头的位置；允许 name 形如 ab_cd / abc1
    pattern = re.compile(r"(?m)^([ \t]*)([A-Za-z_][A-Za-z_0-9]*)\s*=\s*([\[\{])")
    for m in pattern.finditer(code):
        var = m.group(2)
        opener = m.group(3)
        closer = "]" if opener == "[" else "}"
        # 从 opener 开始做括号匹配（考虑字符串中的括号）
        start = m.end() - 1  # opener 的位置
        i = start
        depth = 0
        in_str: str | None = None  # 记录当前字符串的引号字符
        esc = False
        end_idx = -1
        n = len(code)
        while i < n:
            ch = code[i]
            if in_str is not None:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == in_str:
                    in_str = None
            else:
                if ch in ("'", '"'):
                    in_str = ch
                elif ch in "[{":
                    depth += 1
                elif ch in "]}":
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            i += 1

        if end_idx < 0:
            # 未闭合 —— 尝试对已有内容做"末尾修剪"再解析
            snippet = code[start:]
            parsed = _try_parse_truncated_literal(snippet, opener, closer)
            if parsed is not None and var not in out:
                if isinstance(parsed, (list, tuple, dict, str, int, float, bool)):
                    out[var] = parsed
            continue

        snippet = code[start : end_idx + 1]
        try:
            expr = _ast.parse(snippet, mode="eval")
            val = _literal_to_python(expr.body)
        except Exception:
            val = None
        if val is not None and isinstance(val, (list, tuple, dict, str, int, float, bool)):
            # 保留首次赋值（兜底场景避免被后续脏值覆盖）
            out.setdefault(var, val)

    return out


def _try_parse_truncated_literal(snippet: str, opener: str, closer: str) -> Any:
    """对 ``[ ... `` 这种未闭合片段，尝试从后往前裁掉最后不完整的元素，
    补上闭合括号，再交给 ast 解析。最多回退 200 次仍失败则放弃。
    """
    import ast as _ast

    if not snippet or snippet[0] != opener:
        return None
    # 从右向左扫描，遇到逗号就尝试在此截断并闭合
    # 先排除尾部纯空白/破数字
    s = snippet.rstrip()
    # 去掉末尾可能存在的不完整片段（最后一个逗号之后）
    # 做最多 200 次回退
    tries = 0
    while tries < 200 and len(s) > 1:
        last_comma = s.rfind(",")
        if last_comma <= 0:
            break
        candidate = s[:last_comma] + closer
        try:
            expr = _ast.parse(candidate, mode="eval")
            val = _literal_to_python(expr.body)
            if val is not None:
                return val
        except Exception:
            pass
        s = s[:last_comma].rstrip()
        tries += 1
    return None


def _is_numeric_like(v: Any) -> bool:
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        s = v.strip().replace(",", "").replace("%", "")
        try:
            float(s)
            return True
        except Exception:
            return False
    return False


# 变量名黑名单：这些变量即使是数值列表，也不是真正的数据列，而是绘图配置
# 采用分级策略：
#   - EXACT: 精确匹配（或加 _ 前后缀安全匹配）
#   - PREFIX: 仅以该词开头的变量（如 "fig_"/"ax_"）
# 绝不对 "width/height/size" 做后缀匹配，避免误伤业务列（如 body_height / tree_width）
_NUMERIC_NAME_BLACKLIST_EXACT = {
    "explode",
    "explode_settings",
    "figsize",
    "alpha",
    "angle",
    "dpi",
    "padding",
    "pad",
    "margin",
    "linewidth",
    "linewidths",
    "markersize",
    "fontsize",
    "fontsizes",
    "rotation",
    "ticks",
    "xticks",
    "yticks",
    "xlim",
    "ylim",
    "zorder",
    "bar_width",
    "bar_widths",
    "bar_height",
    "gap",
    "spacing",
    "offset",
    "offsets",
    "pos",
    "position",
    "positions",
    "ind",
    "idx",
    "index",
    "indices",
    "seed",
    "bins",
    "xs",
    "ys",
    "x_pos",
    "y_pos",
    "x_positions",
    "y_positions",
}

# 仅以这些词为前缀时判为配置变量
_NUMERIC_NAME_BLACKLIST_PREFIXES = (
    "figsize",
    "linewidth",
    "fontsize",
    "markersize",
    "rotation",
    "explode",
    "xticks",
    "yticks",
    "xlim",
    "ylim",
    "padding",
    "margin",
    "color_",
    "cmap_",
    "hatch_",
    "style_",
    "edge_",
)

# 仅以这些词为后缀时判为配置变量（只限"强指向配置"的词，绝不含 width/height/size）
_NUMERIC_NAME_BLACKLIST_SUFFIXES = (
    "_pos",
    "_positions",
    "_offset",
    "_offsets",
    "_idx",
    "_index",
    "_indices",
    "_bins",
    "_ticks",
    "_lim",
    "_linewidth",
    "_fontsize",
    "_markersize",
    "_alpha",
    "_dpi",
    "_rotation",
    "_zorder",
    "_explode",
)

# 常见行头关键字（变量名包含这些时，该数值/字符串列表倾向于做行头）
_ROW_HEADER_NAME_HINTS = (
    "year",
    "years",
    "month",
    "months",
    "date",
    "dates",
    "day",
    "days",
    "time",
    "times",
    "category",
    "categories",
    "label",
    "labels",
    "name",
    "names",
    "row",
    "rows",
    "group",
    "groups",
    "item",
    "items",
    "type",
    "types",
    "age",
    "ages",
    "period",
    "periods",
    "quarter",
    "quarters",
    "week",
    "weeks",
    "hour",
    "hours",
    "id",
    "ids",
)


def _is_config_name(name: str) -> bool:
    """判断变量名是否属于绘图配置（而非数据列）"""
    n = name.lower()
    if n in _NUMERIC_NAME_BLACKLIST_EXACT:
        return True
    if any(n.startswith(p + "_") or n == p for p in _NUMERIC_NAME_BLACKLIST_PREFIXES):
        return True
    if any(n.endswith(s) for s in _NUMERIC_NAME_BLACKLIST_SUFFIXES):
        return True
    return False


def _is_row_header_name(name: str) -> bool:
    """判断变量名是否像是"行头"（year/month/date/x/label 之类）。"""
    n = name.lower()
    if n in _ROW_HEADER_NAME_HINTS:
        return True
    # x / xs / x_axis / x_vals 这类 x 轴相关
    if n in ("x", "xs", "x_axis", "xaxis", "x_vals", "x_values", "x_data"):
        return True
    # 以关键字开头/结尾
    return any(n.endswith("_" + kw) or n.startswith(kw + "_") for kw in _ROW_HEADER_NAME_HINTS)


def _is_strictly_monotonic(vals: list) -> bool:
    """判断一个数值列表是否严格单调（递增或递减）。"""
    try:
        nums = []
        for v in vals:
            if isinstance(v, (int, float)):
                nums.append(float(v))
            elif isinstance(v, str):
                nums.append(float(v.strip().replace(",", "").replace("%", "")))
            else:
                return False
        if len(nums) < 2:
            return False
        diffs = [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
        return all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
    except Exception:
        return False


def _collect_numeric_lists(literals: dict[str, Any]) -> list[tuple[str, list]]:
    """从收集到的字面量中，挑出所有"纯数值列表"（已排除配置项）。

    Returns:
        [(变量名, 数值列表), ...]
    """
    out: list[tuple[str, list]] = []
    for name, val in literals.items():
        if _is_config_name(name):
            continue
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            if all(_is_numeric_like(x) for x in val):
                # 过滤全零 / 全同值列（explode 未命中黑名单时的兜底）
                try:
                    floats = [float(str(x).replace("%", "")) for x in val]
                    if len(set(floats)) <= 1 and floats[0] in (0.0, 1.0):
                        continue
                except Exception:
                    pass
                out.append((name, list(val)))
    return out


def _collect_label_lists(literals: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """从收集到的字面量中，挑出所有"纯字符串列表"（类别标签）。

    顺便把字符串里的换行符替换为空格，避免破坏内部 CSV 的行分隔。
    """
    out: list[tuple[str, list[str]]] = []
    for name, val in literals.items():
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            if all(isinstance(x, str) and x.strip() for x in val):
                cleaned = [_clean_label(x) for x in val]
                out.append((name, cleaned))
    return out


def _clean_label(s: str) -> str:
    """清洗标签文本：去除换行/制表，折叠多空格，尝试去掉尾部的"数值/百分号/单位"后缀。

    处理形式：
      - "不明原因 50%" → "不明原因"
      - "买保险1700元" → "买保险"
      - "All of them: 55.1%" → "All of them"
      - "某类别 12.5" → "某类别"（当数字能明显判断为数据时）
    """
    if not isinstance(s, str):
        return str(s)
    # 换行/制表替换为空格
    t = s.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    # 折叠多空格
    t = re.sub(r"\s+", " ", t).strip()
    t = _strip_label_trailing_values(t)
    return t


def _strip_label_trailing_values(t: str) -> str:
    """去掉标签尾部的"数值+可选单位"后缀，保留类别主名部分。

    注意：仅在明确含数据后缀时剥离，避免误伤纯数字类别（如 "2020"）。
    使用循环迭代剥离，每轮去掉一个"尾部数值/单位"片段。
    """
    if not t:
        return t

    # ① 冒号分隔的数据部分："A: 55.1%" / "A：55.1"
    m = re.search(r"^(.+?)[：:]\s*[-+]?\d[\d,\.]*\s*%?\s*[^\s\d]*\s*$", t)
    if m:
        head = m.group(1).strip()
        if head and not head.replace(".", "").replace("-", "").isdigit():
            return head

    # 迭代剥离尾部的"数字+可选单位"片段
    cur = t
    # 允许的单位字符（中英文常见）
    unit_cls = r"(?:%|元|件|人|年|岁|月|日|次|吨|克|千克|米|厘米|美元|美分|°|℃|h|min|s|kg)"
    stripped = False
    for _ in range(6):  # 最多剥 6 轮，防止死循环
        new_cur = None

        # 模式 A：空格/逗号/斜杠前 → 数字（带 % 或单位才剥；避免误伤 "数据A-2" 这种延展名）
        # 分隔符不含 `-`（否则会误剥 "-2" "-V1"）
        m2 = re.search(
            r"^(.+?)\s*[\s,，/]\s*[-+]?\d+(?:\.\d+)?\s*" + unit_cls + r"\s*"
            r"(?:\s*[-—~～]\s*[-+]?\d+(?:\.\d+)?\s*" + unit_cls + r"?)?\s*$",
            cur,
        )
        if m2:
            new_cur = m2.group(1).strip()

        # 模式 B：中文字符紧接数字（无空格），如"买保险1700元"
        if new_cur is None:
            m3 = re.search(
                r"^(.*?[\u4e00-\u9fff])\s*[-+]?\d+(?:\.\d+)?\s*" + unit_cls + r"\s*$",
                cur,
            )
            if m3:
                new_cur = m3.group(1).strip()

        if new_cur is None or new_cur == cur:
            break
        # 保护：不要剥成空或纯数字
        if not new_cur or new_cur.replace(".", "").replace("-", "").isdigit():
            break
        cur = new_cur
        stripped = True

    if stripped:
        return cur.strip(" ,，/~～")
    return t


def _try_extract_csv_from_triple_quoted(code: str) -> str:
    """从 Python 代码里抢救"三引号 CSV 字面量"并转成内部 CSV。

    典型场景（TinyChart / ChartMoE 常见）：
        data = StringIO(\"\"\"
        Date,A,B
        2020,1,2
        2021,3,4
        \"\"\")
        df = pd.read_csv(StringIO('''...'''))
        data = \"\"\"col1,col2\\n...\"\"\"  # 直接三引号常量

    做法：扫描代码中所有 ``\"\"\"...\"\"\"`` 或 ``'''...'''`` 字面量，
    取"最像 CSV"的那一段（多行、至少一半行含逗号）转成内部 CSV。

    返回：内部 CSV（`\\t/\\n` 分隔），若没找到候选则返回 ``""``。
    """
    if not code:
        return ""

    # 抓所有 """...""" 和 '''...'''（非贪婪，跨行）
    candidates: list[str] = []
    for m in re.finditer(r'"""(.*?)"""', code, flags=re.DOTALL):
        candidates.append(m.group(1))
    for m in re.finditer(r"'''(.*?)'''", code, flags=re.DOTALL):
        candidates.append(m.group(1))

    if not candidates:
        return ""

    def _score(block: str) -> tuple[int, int]:
        """评分：多行、逗号行越多越好。返回 (逗号行数, 总行数)。"""
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if len(lines) < 2:
            return (0, 0)
        comma_lines = sum(1 for ln in lines if "," in ln)
        return (comma_lines, len(lines))

    # 选最像 CSV 的候选
    best_block = ""
    best_score = (0, 0)
    for block in candidates:
        s = _score(block)
        # 逗号行数必须 >= 2，且占一半以上
        if s[0] >= 2 and s[0] * 2 >= s[1] and s > best_score:
            best_block = block
            best_score = s

    if not best_block:
        return ""

    # 用 csv 模块严格解析，处理引号
    try:
        reader = csv.reader(io.StringIO(best_block.strip()))
        rows = [row for row in reader if any(cell.strip() for cell in row)]
    except Exception:
        return ""

    if len(rows) < 2:
        return ""

    # 裁齐：以首行列数为准，pad/truncate 其他行
    n_cols = len(rows[0])
    norm_rows: list[list[str]] = []
    for r in rows:
        if len(r) < n_cols:
            r = r + [""] * (n_cols - len(r))
        elif len(r) > n_cols:
            r = r[:n_cols]
        norm_rows.append([cell.strip() for cell in r])

    csv_rows = [" \\t ".join(cells) for cells in norm_rows]
    return " \\n ".join(csv_rows)


def python_code_to_internal_csv(text: str) -> str:
    """从 Python 代码中提取数值数据，转换为内部 CSV (\\t/\\n 分隔) 格式。

    核心策略（经实测优化）：
        代码中的变量名（如 ``content_cost``）几乎不可能与 anno 的中文列头
        （如 ``内容成本（亿元人民币）``）一致，因此**不使用变量名作为列头**。
        转而优先提取 matplotlib 绘图调用中 ``label=`` / ``labels=`` 参数里的
        中文字符串作为列头；若无则输出**无表头 CSV**（首行留空），让 csv_eval
        的 ``_is_empty_header`` 分支自动对齐，只比较行标签 + 数值。

    识别的数据形态（按优先级）：
        1) 字典 ``{列头: [数值, ...]}``（列头本身就是中文）；
        2) 二维数值列表 ``box_data = [[...], [...], ...]``（每行一个箱线图样本）；
        3) 多个一维数值列表（长度一致）+ 一个字符串标签列表作为行头；
        4) 仅有一个一维数值列表。

    Args:
        text: 模型输出的 Python 代码（可能含 ```python``` 围栏）

    Returns:
        内部 CSV 字符串；无法识别则返回空字符串。
    """
    code = _extract_python_code(text)
    if not code:
        return ""

    literals = _collect_code_literals(code)
    plot_labels = _collect_plot_labels(code)  # 从 plt.bar(..., label=...) 收集的列头文字

    # ---- 优先级 0.1：三引号内的标准 CSV 字面量（StringIO / read_csv 场景） ----
    # 模型常用写法：
    #   data = StringIO("""\n日期,A,B\n2020,1,2\n...""")
    #   df = pd.read_csv(StringIO('''...CSV 内容...'''))
    #   data = """col1,col2\nx,1\ny,2"""
    # 这些字面量包含现成的"逗号+换行"标准 CSV，直接转内部 CSV 而无需走 literal 收集。
    csv_from_triple = _try_extract_csv_from_triple_quoted(code)
    if csv_from_triple:
        return csv_from_triple

    if not literals:
        return ""

    # ---- 优先级 0.5：字典列表形式的"列定义" ----
    #   [{"data": [...], "label": "A"}, {"data": [...], "label": "B"}, ...]
    #   或 {"values": [...], "name": "X"} / {"y": [...], "label": "X"} 等
    col_def_result = _try_build_from_dict_list_of_cols(literals, code=code)
    if col_def_result:
        return col_def_result

    # ---- 优先级 1：字典 {中文列头: [数值, ...]} ----
    for name, val in literals.items():
        if isinstance(val, dict) and val:
            col_keys = list(val.keys())
            col_vals = [val[k] for k in col_keys]

            # 字典值是 dict：嵌套二层（行→列→值），与 JSON 任务一致
            if all(isinstance(v, dict) for v in col_vals):
                inner_keys: list[str] = []
                inner_seen: set = set()
                for cat, cat_vals in val.items():
                    if not isinstance(cat_vals, dict):
                        continue
                    for k in cat_vals.keys():
                        if k not in inner_seen:
                            inner_seen.add(k)
                            inner_keys.append(k)
                if col_keys and inner_keys:
                    # 外层为列，内层为行（与 SE_JSON 的约定一致）
                    header = [""] + [str(c).strip() for c in col_keys]
                    rows_text = [" \\t ".join(header)]
                    for ik in inner_keys:
                        cells = [str(ik).strip()]
                        for ck in col_keys:
                            cv = val.get(ck, {})
                            cells.append(str(cv.get(ik, "")).strip())
                        rows_text.append(" \\t ".join(cells))
                    return " \\n ".join(rows_text)

            # 字典值是等长 list：列式二维表；字典键直接作为列头（通常是中文）
            if all(isinstance(v, (list, tuple)) for v in col_vals):
                max_len = max((len(v) for v in col_vals), default=0)
                if max_len >= 1 and all(len(v) == max_len for v in col_vals):
                    row_labels = _find_matching_label_list(literals, max_len, exclude=name)
                    header = [""] + [str(c).strip() for c in col_keys]
                    rows_text = [" \\t ".join(header)]
                    for i in range(max_len):
                        label = row_labels[i] if row_labels else str(i + 1)
                        cells = [label]
                        for c in col_vals:
                            cells.append(_format_value(c[i]) if i < len(c) else "")
                        rows_text.append(" \\t ".join(cells))
                    return " \\n ".join(rows_text)

            # 字典值是标量：单列表格
            if all(_is_numeric_like(v) or isinstance(v, str) for v in col_vals):
                # 首行留空列头，让 csv_eval 走无表头对齐
                rows_text = [" \\t ".join(["", ""])]
                for k, v in val.items():
                    rows_text.append(" \\t ".join([str(k).strip(), _format_value(v)]))
                return " \\n ".join(rows_text)

    # ---- 优先级 2：二维数值列表（箱线图等）----
    matrix_entry = _find_numeric_matrix(literals)
    if matrix_entry is not None:
        matrix_name, matrix = matrix_entry
        n_rows = len(matrix)
        n_cols = len(matrix[0]) if matrix else 0

        # 行头：优先找长度为 n_rows 的字符串列表
        row_labels = _find_matching_label_list(literals, n_rows, exclude=matrix_name)
        # 列头：优先使用 plot_labels；仅当 chart_type=箱线图 时 5 列走默认列头；否则留空
        col_labels = _infer_col_labels(n_cols, plot_labels, chart_type=get_chart_type())

        header = [""] + col_labels
        rows_text = [" \\t ".join(header)]
        for i in range(n_rows):
            label = row_labels[i] if row_labels and i < len(row_labels) else str(i + 1)
            cells = [str(label).strip()]
            for j in range(n_cols):
                cells.append(_format_value(matrix[i][j]) if j < len(matrix[i]) else "")
            rows_text.append(" \\t ".join(cells))
        return " \\n ".join(rows_text)

    # ---- 优先级 3：多个一维数值列表 + 标签列表 ----
    numeric_lists = _collect_numeric_lists(literals)
    label_lists = _collect_label_lists(literals)

    if numeric_lists:
        from collections import Counter

        # --- Step A: 先识别"行头候选"，用它的长度来决定 target_len ---
        # 优先级：
        #   1) 字符串行头（label_lists 里变量名像行头、或最长）
        #   2) 数值行头（numeric_lists 里变量名含 year/month/date/age/x...）
        #   3) 数值行头（严格单调）
        header_candidate: tuple[str, list, bool] | None = None  # (name, values, is_string)
        # 1) 字符串行头：优先变量名像行头的
        str_hint_list = [(n, v) for n, v in label_lists if _is_row_header_name(n)]
        if str_hint_list:
            header_candidate = (str_hint_list[0][0], str_hint_list[0][1], True)
        # 2) 数值行头（变量名像行头）
        if header_candidate is None:
            num_hint_list = [(n, v) for n, v in numeric_lists if _is_row_header_name(n)]
            if num_hint_list:
                # 取"第一个行头命名"的变量
                header_candidate = (num_hint_list[0][0], num_hint_list[0][1], False)
        # 3) 严格单调的数值列
        if header_candidate is None:
            for n, v in numeric_lists:
                if _is_strictly_monotonic(v):
                    header_candidate = (n, v, False)
                    break

        # --- Step B: 决定 target_len ---
        if header_candidate is not None:
            target_len = len(header_candidate[1])
        else:
            # 没有行头候选：用长度众数
            len_counter = Counter(len(v) for _, v in numeric_lists)
            target_len = len_counter.most_common(1)[0][0]

        # --- Step C: 取长度 == target_len 的数值列作为 value_cols ---
        value_cols = [(n, v) for n, v in numeric_lists if len(v) == target_len]

        # Step C.5: 回退 —— 如果按 header_candidate 的长度取不到有效数据列，
        #   说明 header 的长度和数据长度不一致（LLM 写的代码常见 bug），
        #   改用众数长度，并放弃该 header_candidate。
        if header_candidate is not None:
            # header 本身也计入 value_cols，若 header 是唯一的 value 列，说明没有真正的数据
            non_header_count = sum(1 for n, _ in value_cols if n != header_candidate[0])
            if non_header_count == 0:
                len_counter = Counter(len(v) for _, v in numeric_lists)
                fallback_len = len_counter.most_common(1)[0][0]
                if fallback_len != target_len:
                    target_len = fallback_len
                    value_cols = [(n, v) for n, v in numeric_lists if len(v) == target_len]
                    header_candidate = None  # 弃用原 header

        # --- Step D: 确定 chosen_labels（从 header_candidate 或字符串列表） ---
        chosen_labels: list[str] | None = None
        chosen_label_name: str | None = None

        # D.1 先尝试字符串标签列表（若有长度匹配的）
        for lname, labels in label_lists:
            if len(labels) == target_len:
                chosen_labels = labels
                chosen_label_name = lname
                break

        # D.2 字符串标签列表中变量名又出现在 value_cols 里（数字串），需从 value_cols 剔除
        if chosen_label_name is not None:
            value_cols = [(n, v) for n, v in value_cols if n != chosen_label_name]

        # D.3 若还没有字符串行头，用 header_candidate 的数值列
        if chosen_labels is None and header_candidate is not None and not header_candidate[2]:
            if len(header_candidate[1]) == target_len:
                chosen_labels = [_format_value(x) for x in header_candidate[1]]
                # 从 value_cols 中剔除该行头列
                value_cols = [(n, v) for n, v in value_cols if n != header_candidate[0]]

        # 列头：优先使用 plot_labels；否则留空（无表头）
        col_count = len(value_cols)
        col_labels = _infer_col_labels(col_count, plot_labels, fallback_empty=True, chart_type=get_chart_type())

        header = [""] + col_labels
        rows_text = [" \\t ".join(str(h).strip() for h in header)]
        for i in range(target_len):
            label = chosen_labels[i] if chosen_labels and i < len(chosen_labels) else str(i + 1)
            cells = [_clean_label(str(label))]
            for _, v in value_cols:
                cells.append(_format_value(v[i]) if i < len(v) else "")
            rows_text.append(" \\t ".join(cells))
        return " \\n ".join(rows_text)

    return ""


def _format_value(v: Any) -> str:
    """格式化数值为字符串，小数末尾去零，整数不加 .0。"""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        # 去掉末尾多余 0
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        return s if s else str(v)
    return str(v).strip()


def _try_build_from_dict_list_of_cols(literals: dict[str, Any], code: str = "") -> str:
    """识别形如 ``[{"data": [...], "label": "X"}, ...]`` 的字典列表列定义。

    匹配规则：
      - 某变量是 list，元素全是 dict；
      - 每个 dict 都含（顺序尝试）：
          数据字段：``data / values / y / y_values / series``（值为列表或**变量名引用**）
          标签字段：``label / name / title / key / category``（值为字符串）
      - 所有数据列长度相同；
      - 至少 2 列（元素）。

    Args:
        literals: 已解析的字面量字典
        code: 原始代码（传入后可解析变量引用形式的 data 字段）

    生成的 CSV：
      - 列头：各 dict 的 label 值
      - 行头：优先用 literals 里长度匹配的字符串列表（label_lists）或"行头候选"变量
      - 数据：各 dict 的 data 值（可能来自 Name 引用查表）

    成功则返回内部 CSV；失败返回 ""。
    """
    data_keys = ("data", "values", "y", "y_values", "series", "points")
    label_keys = ("label", "name", "title", "key", "category")

    # 预解析：如果给了 code，抓取 {var_name -> list of dict (含 Name 引用)} 映射
    # 用来在 literals 里 data 为 None 时，回查变量名
    name_ref_map = _collect_dict_list_name_refs(code) if code else {}

    for var_name, val in literals.items():
        if not isinstance(val, list) or len(val) < 2:
            continue
        if not all(isinstance(x, dict) for x in val):
            continue

        # 对应的 Name 引用列表（与 val 等长）
        ref_list = name_ref_map.get(var_name, [])

        col_series: list[tuple[str, list]] = []  # [(label, data)]
        ok = True
        for idx, d in enumerate(val):
            # 找 label
            lbl: str | None = None
            for k in label_keys:
                v = d.get(k)
                if isinstance(v, str) and v.strip():
                    lbl = _clean_label(v)
                    break
            # 找 data：列表字面量 / 变量引用
            dat: list | None = None
            for k in data_keys:
                if k not in d:
                    continue
                v = d[k]
                if isinstance(v, list) and all(_is_numeric_like(x) for x in v):
                    dat = list(v)
                    break
                # v 为 None（原来是 Name 引用）：通过 ref_list 回查
                if v is None and idx < len(ref_list):
                    ref_name = ref_list[idx].get(k)
                    if ref_name and ref_name in literals:
                        ref_val = literals[ref_name]
                        if isinstance(ref_val, list) and all(_is_numeric_like(x) for x in ref_val):
                            dat = list(ref_val)
                            break
            if lbl is None or dat is None:
                ok = False
                break
            col_series.append((lbl, dat))

        if not ok or len(col_series) < 2:
            continue

        # 长度一致检查
        lens = {len(d) for _, d in col_series}
        if len(lens) != 1:
            continue
        target_len = next(iter(lens))
        if target_len < 1:
            continue

        # 行头：优先找长度匹配的字符串列表，然后找长度匹配的数值行头
        row_labels: list[str] | None = None
        for lname, lval in literals.items():
            if lname == var_name:
                continue
            if isinstance(lval, (list, tuple)) and len(lval) == target_len:
                if all(isinstance(x, str) and x.strip() for x in lval):
                    row_labels = [_clean_label(str(x)) for x in lval]
                    break
        if row_labels is None:
            for lname, lval in literals.items():
                if lname == var_name:
                    continue
                if isinstance(lval, (list, tuple)) and len(lval) == target_len:
                    if all(_is_numeric_like(x) for x in lval):
                        if _is_row_header_name(lname) or _is_strictly_monotonic(list(lval)):
                            row_labels = [_format_value(x) for x in lval]
                            break

        header = [""] + [lbl for lbl, _ in col_series]
        rows_text = [" \\t ".join(header)]
        for i in range(target_len):
            label = row_labels[i] if row_labels and i < len(row_labels) else str(i + 1)
            cells = [_clean_label(str(label))]
            for _, d in col_series:
                cells.append(_format_value(d[i]) if i < len(d) else "")
            rows_text.append(" \\t ".join(cells))
        return " \\n ".join(rows_text)

    return ""


def _collect_dict_list_name_refs(code: str) -> dict[str, list[dict[str, str]]]:
    """扫描代码，找到所有形如 ``var = [{"key": name_ref, ...}, ...]`` 的赋值，
    返回 ``{var_name: [ {key: referenced_name, ...}, ... ]}``。

    只记录 dict value 是 Name（变量引用）的字段，其他字段（字面量、属性访问等）跳过。
    """
    out: dict[str, list[dict[str, str]]] = {}
    if not code:
        return out
    try:
        tree = ast.parse(code)
    except Exception:
        return out

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        var_name = node.targets[0].id
        value = node.value
        if not isinstance(value, ast.List):
            continue
        ref_per_item: list[dict[str, str]] = []
        all_dict = True
        for elt in value.elts:
            if not isinstance(elt, ast.Dict):
                all_dict = False
                break
            item_refs: dict[str, str] = {}
            for kn, vn in zip(elt.keys, elt.values):
                if not (isinstance(kn, ast.Constant) and isinstance(kn.value, str)):
                    continue
                if isinstance(vn, ast.Name):
                    item_refs[kn.value] = vn.id
            ref_per_item.append(item_refs)
        if all_dict and ref_per_item:
            out[var_name] = ref_per_item
    return out


def _find_matching_label_list(literals: dict[str, Any], target_len: int, exclude: str = "") -> list[str] | None:
    """在字面量中找一个长度为 target_len 的纯字符串列表作为行头。

    排除似乎是颜色值 / 文件路径 / 单字符 等非类别标签的列表；
    以及常见不相关的变量名（color/colors/palette 等）。
    """
    blacklist_name_keywords = {
        "color",
        "colors",
        "palette",
        "cmap",
        "hatch",
        "marker",
        "linestyle",
        "linewidth",
        "alpha",
        "figsize",
        "path",
        "font",
    }

    def _looks_like_color(s: str) -> bool:
        s = s.strip()
        if not s:
            return False
        # #RRGGBB / #RGB
        if s.startswith("#") and len(s) in (4, 7, 9):
            return True
        # CSS 颜色单字
        css_colors = {
            "red",
            "green",
            "blue",
            "yellow",
            "orange",
            "purple",
            "pink",
            "cyan",
            "magenta",
            "black",
            "white",
            "gray",
            "grey",
            "brown",
            "lime",
            "teal",
            "navy",
            "olive",
            "maroon",
            "silver",
            "gold",
        }
        if s.lower() in css_colors:
            return True
        # rgba(...) / rgb(...)
        if s.startswith("rgb") or s.startswith("rgba"):
            return True
        return False

    def _is_valid_label_list(lvals: list[str]) -> bool:
        # 所有元素都是颜色值 -> 不作为标签
        if all(_looks_like_color(x) for x in lvals):
            return False
        return True

    # 优先选择变量名不在黑名单、且内容不像颜色值的列表
    good_candidates: list[list[str]] = []
    fallback_candidates: list[list[str]] = []
    for name, val in literals.items():
        if name == exclude:
            continue
        if not (isinstance(val, (list, tuple)) and len(val) == target_len):
            continue
        if not all(isinstance(x, str) and x.strip() for x in val):
            continue
        labels = [_clean_label(str(x)) for x in val]
        if not _is_valid_label_list(labels):
            continue
        lname_lower = name.lower()
        if any(k in lname_lower for k in blacklist_name_keywords):
            fallback_candidates.append(labels)
        else:
            good_candidates.append(labels)

    if good_candidates:
        return good_candidates[0]
    if fallback_candidates:
        return fallback_candidates[0]
    return None


def _find_numeric_matrix(literals: dict[str, Any]) -> tuple[str, list[list]] | None:
    """在字面量中查找形如 ``[[num, ...], [num, ...], ...]`` 的二维数值矩阵。

    要求：至少 2 行，每行长度相等（>=1），全部元素可数值化。
    """
    best: tuple[str, list[list]] | None = None
    for name, val in literals.items():
        if not isinstance(val, (list, tuple)) or len(val) < 2:
            continue
        if not all(isinstance(row, (list, tuple)) for row in val):
            continue
        row_lens = [len(row) for row in val]
        if len(set(row_lens)) != 1 or row_lens[0] == 0:
            continue
        if not all(_is_numeric_like(x) for row in val for x in row):
            continue
        candidate = (name, [list(row) for row in val])
        # 偏好行数 * 列数更大的矩阵（更可能是主数据）
        if best is None or len(candidate[1]) * len(candidate[1][0]) > len(best[1]) * len(best[1][0]):
            best = candidate
    return best


def _infer_col_labels(
    n_cols: int,
    plot_labels: list[str],
    fallback_empty: bool = True,
    chart_type: str = "",
) -> list[str]:
    """根据列数推断列头。

    优先使用从 matplotlib 绘图调用里抓到的 ``label=`` 值；若数量不匹配则：
      - 当 chart_type 明确为箱线图（boxplot）且 5 列时，使用箱线图默认列头
      - 否则返回 n_cols 个空字符串（无表头对齐模式）。
        放弃"5 列默认箱线图"的猜测：对柱状图的 5 年份数据是致命误判。
    """
    # plot_labels 精确匹配数量
    if plot_labels and len(plot_labels) == n_cols:
        return [str(x).strip() for x in plot_labels]
    # 仅当明确是箱线图时，5 列才默认为箱线图统计量
    if n_cols == 5 and chart_type and ("箱线" in chart_type or "box" in chart_type.lower()):
        return ["最小值", "Q1", "中位数", "Q3", "最大值"]
    # 无表头对齐
    if fallback_empty:
        return [""] * n_cols
    return [f"col_{i + 1}" for i in range(n_cols)]


def _collect_plot_labels(code: str) -> list[str]:
    """从 matplotlib 绘图代码中收集 ``label=`` / ``labels=`` 参数的字符串值。

    扫描常见 API：``plt.bar / plt.plot / plt.pie / plt.boxplot / plt.hist``、
    ``ax.bar / ax.plot / ax.pie / ax.boxplot / ax.hist``。
    优先级：
        1) labels=[...]（一次性传入多个）
        2) label="..."（单个，多次调用收集顺序）
    """
    if not code or not code.strip():
        return []
    try:
        tree = ast.parse(code)
    except Exception:
        return []

    plot_methods = {"bar", "barh", "plot", "pie", "boxplot", "hist", "scatter", "stackplot", "errorbar"}

    single_labels: list[str] = []  # 多次 label= 的顺序列表
    multi_labels: list[list[str]] = []  # labels=[...] 的直接列表

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # 取方法名
        method_name = ""
        f = node.func
        if isinstance(f, ast.Attribute):
            method_name = f.attr
        elif isinstance(f, ast.Name):
            method_name = f.id
        if method_name not in plot_methods:
            continue

        for kw in node.keywords:
            if kw.arg == "label":
                v = _literal_to_python(kw.value)
                if isinstance(v, str) and v.strip():
                    single_labels.append(v.strip())
            elif kw.arg == "labels":
                v = _literal_to_python(kw.value)
                if isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
                    multi_labels.append([str(x).strip() for x in v if str(x).strip()])

    # 优先返回最长的 labels=[...]（通常 pie 的 labels 会完整列出所有类别）
    if multi_labels:
        multi_labels.sort(key=len, reverse=True)
        return multi_labels[0]
    return single_labels


def python_code_to_mermaid(text: str) -> str:
    """从 Python 代码中提取流程图/图结构，转为 **mermaid flowchart** 文本。

    主要针对模型（如 ChartCoder）把流程图画成 ``matplotlib + networkx`` 代码的情况：
    ref 是 mermaid，pred 如果是 Python → 当作 mermaid 解析会直接失败 0 分。
    这里把 pred 归一化成与 ref 同语言的 mermaid，让下游 ``flowchart_eval_multi``
    按正常图对图比对。

    策略优先级：
      1) networkx：``G.add_node / add_edge``；
      2) edges 风格变量（list of [parent, child]）；
      3) 若只有孤立字符串节点，退化为 ``flowchart TD`` 下的一串无连边节点。

    找不到任何信号时返回空串，让调用方 0 分。
    """
    code = _extract_python_code(text)
    if not code:
        return ""

    # --- 1) networkx ---
    nx_nodes, nx_edges = _extract_networkx_graph(code)
    if nx_edges or nx_nodes:
        return _graph_to_mermaid(nx_nodes, nx_edges)

    # --- 2) edges 风格变量 ---
    literals = _collect_code_literals(code)
    for name, val in literals.items():
        lower_name = name.lower()
        if not any(k in lower_name for k in ["edge", "link", "relation", "parent_child"]):
            continue
        if not isinstance(val, (list, tuple)):
            continue
        pairs: list[tuple[str, str]] = []
        nodes_order: list[str] = []
        seen: set[str] = set()
        for item in val:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                a, b = item[0], item[1]
                if isinstance(a, (str, int, float)) and isinstance(b, (str, int, float)):
                    sa, sb = str(a).strip(), str(b).strip()
                    if sa and sb:
                        pairs.append((sa, sb))
                        for s in (sa, sb):
                            if s not in seen:
                                seen.add(s)
                                nodes_order.append(s)
        if pairs:
            return _graph_to_mermaid(nodes_order, pairs)

    # --- 3) 嵌套字典兜底（MSRL 等模型的常见形态）---
    #   data = { "父": { "子": ["孙1", "孙2"] }, ... }
    #   当 literals 里存在这种 dict-of-dict / dict-of-list 的层级结构时，
    #   把 key→child 关系抽出来形成 graph，保底给流程图一个有意义的结构匹配。
    nd_nodes, nd_edges = _extract_nested_dict_as_graph(literals)
    if nd_edges or len(nd_nodes) >= 2:
        return _graph_to_mermaid(nd_nodes, nd_edges)

    return ""


def _graph_to_mermaid(nodes: list[str], edges: list[tuple[str, str]]) -> str:
    """把 (nodes, edges) 渲染成最简 ``flowchart TD`` mermaid 文本。

    用 N0 / N1 / N2 作为节点 ID，避免原始 label 中的空格/引号/特殊字符
    导致 mermaid 解析失败；label 内容用双引号包裹，并对 ``"`` 做转义。
    """
    if not nodes and not edges:
        return ""
    # 收集全部节点
    order: list[str] = []
    seen: set[str] = set()
    for n in nodes:
        s = str(n).strip()
        if s and s not in seen:
            seen.add(s)
            order.append(s)
    for a, b in edges:
        for s in (str(a).strip(), str(b).strip()):
            if s and s not in seen:
                seen.add(s)
                order.append(s)
    if not order:
        return ""
    id_map = {lbl: f"N{i}" for i, lbl in enumerate(order)}

    def _q(lbl: str) -> str:
        return lbl.replace('"', '\\"')

    lines = ["flowchart TD"]
    for lbl in order:
        lines.append(f'    {id_map[lbl]}["{_q(lbl)}"]')
    for a, b in edges:
        sa, sb = str(a).strip(), str(b).strip()
        if sa in id_map and sb in id_map:
            lines.append(f"    {id_map[sa]} --> {id_map[sb]}")
    return "\n".join(lines)


def python_code_to_markdown_list(text: str) -> str:
    """从 Python 代码中提取逻辑结构图信息，转为 Markdown 无序列表。

    策略：
    1) 尝试找到 edges / edge_list / links 等变量（list of [parent, child] 或 tuple），
       可构造任意树形结构。
    1.5) networkx：``G.add_node(...)`` + ``G.add_edge(...)`` 构建 DAG/树。
    2) 位置感知：``ax.text(x, y, ...)`` 按 x 坐标聚类为层级（手绘思维导图）。
    3) 找不到则将所有字符串节点按顺序平铺为单层列表。
    """
    code = _extract_python_code(text)
    if not code:
        return ""

    literals = _collect_code_literals(code)
    # 注意：literals 可能为空，但代码里仍可能有 networkx 调用 / ax.text / 纯字符串，
    # 不在此处提前返回，交由后续策略决定。

    # 策略 1：edges 列表
    edge_candidates = []
    for name, val in literals.items():
        lower_name = name.lower()
        if any(k in lower_name for k in ["edge", "link", "relation", "parent_child"]):
            if isinstance(val, (list, tuple)):
                pairs = []
                for item in val:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        if isinstance(item[0], str) and isinstance(item[1], str):
                            pairs.append((item[0].strip(), item[1].strip()))
                if pairs:
                    edge_candidates.append(pairs)

    if edge_candidates:
        pairs = edge_candidates[0]
        children_map: dict[str, list[str]] = {}
        all_nodes: set[str] = set()
        children_set: set[str] = set()
        for parent, child in pairs:
            children_map.setdefault(parent, []).append(child)
            all_nodes.add(parent)
            all_nodes.add(child)
            children_set.add(child)

        roots = [n for n in all_nodes if n not in children_set]
        if not roots:
            roots = list(all_nodes)[:1]

        lines: list[str] = []
        visited: set[str] = set()

        def _dfs(node: str, level: int) -> None:
            if node in visited:
                return
            visited.add(node)
            lines.append(f"{'  ' * level}- {node}")
            for c in children_map.get(node, []):
                _dfs(c, level + 1)

        for r in roots:
            _dfs(r, 0)
        return "\n".join(lines)

    # 策略 1.5：networkx 图解析（ChartCoder 等模型生成 nx.Graph / nx.DiGraph）
    #   识别 G.add_node('X', ...) / G.add_edge('A', 'B', ...) 构建层级。
    nx_nodes, nx_edges = _extract_networkx_graph(code)
    if nx_edges:
        children_map: dict[str, list[str]] = {}
        all_nodes_set: set[str] = set(nx_nodes)
        children_set_nx: set[str] = set()
        for parent, child in nx_edges:
            children_map.setdefault(parent, []).append(child)
            all_nodes_set.add(parent)
            all_nodes_set.add(child)
            children_set_nx.add(child)

        roots = [n for n in nx_nodes if n in all_nodes_set and n not in children_set_nx]
        # 兜底：若所有节点都是某人的子节点（存在环），挑一个未在 children 里出现的
        if not roots:
            roots = [n for n in all_nodes_set if n not in children_set_nx]
        if not roots and all_nodes_set:
            roots = [next(iter(all_nodes_set))]

        lines_nx: list[str] = []
        visited_nx: set[str] = set()

        def _dfs_nx(node: str, level: int) -> None:
            if node in visited_nx:
                return
            visited_nx.add(node)
            lines_nx.append(f"{'  ' * level}- {node}")
            for c in children_map.get(node, []):
                _dfs_nx(c, level + 1)

        for r in roots:
            _dfs_nx(r, 0)
        # 遗漏孤立节点兜底
        for n in all_nodes_set:
            if n not in visited_nx:
                lines_nx.append(f"- {n}")
                visited_nx.add(n)
        if lines_nx:
            return "\n".join(lines_nx)
    elif len(nx_nodes) >= 2:
        # 只有 add_node 没有 add_edge：退化为扁平列表（第一个作根，其余作一级子）
        root = nx_nodes[0]
        lines_nx2 = [f"- {root}"] + [f"  - {n}" for n in nx_nodes[1:]]
        return "\n".join(lines_nx2)

    # 策略 2：位置感知的层级构造（针对手绘思维导图）
    #   从 ax.text(x, y, '文本') / draw_box(text, x, y, ...) 里提取带坐标的节点，
    #   按 x 坐标聚类到层级，相邻 x 层之间建立父子关系。
    positioned_nodes = _extract_positioned_text_nodes(code)
    if len(positioned_nodes) >= 3:
        md = _build_hierarchy_from_positions(positioned_nodes)
        if md:
            return md

    # 策略 2.5：嵌套字典 → 多级无序列表（MSRL 等模型的常见形态）
    #   data = {"A": {"B": ["C1", "C2"], "D": "E"}, ...}
    #   在扁平兜底之前，尝试把最"树状"的 nested dict 转成 markdown list。
    md_nd = _nested_dict_to_markdown_list(literals)
    if md_nd:
        return md_nd

    # 策略 3：扁平兜底——收集所有候选节点文字，合并去重后返回扁平列表
    node_texts = _extract_text_string_args(code)

    # 构造一个集合，便于去重
    seen = set(node_texts)

    # 从字面量里的纯字符串列表中补充
    for name, val in literals.items():
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            if all(isinstance(x, str) and x.strip() for x in val):
                for x in val:
                    t = _clean_label(str(x))
                    if not t or t in seen:
                        continue
                    if _is_plausible_node_text(t):
                        seen.add(t)
                        node_texts.append(t)
        elif isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, str):
                    t = _clean_label(v)
                    if not t or t in seen:
                        continue
                    if _is_plausible_node_text(t):
                        seen.add(t)
                        node_texts.append(t)

    if len(node_texts) >= 2:
        # 把第一个作为根，其余作为一级子节点（对 tree_eval 更友好）
        root = node_texts[0]
        others = node_texts[1:]
        lines = [f"- {root}"]
        for t in others:
            lines.append(f"  - {t}")
        return "\n".join(lines)

    return ""


def _extract_positioned_text_nodes(code: str) -> list[tuple[float, float, str]]:
    """从代码中抓取 ``ax.text(x, y, '文本')`` 类调用，返回 (x, y, text) 列表。

    识别的常见调用形式：
      * ``ax.text(x, y, '...')`` / ``plt.text(x, y, '...')``
      * ``ax.annotate('...', xy=(x, y))`` / ``plt.annotate('...', (x, y))``
      * 自定义绘制函数：``draw_box('...', x, y, ...)`` / ``draw_node('...', x, y, ...)``

    x/y 为字面量数字时才会被记录；变量引用不尝试求值。
    """
    if not code:
        return []

    try:
        tree = ast.parse(code)
    except Exception:
        return []

    nodes: list[tuple[float, float, str]] = []

    def _num(val) -> float | None:
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = ""
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
        lower = func_name.lower()

        # 1) ax.text(x, y, s)
        if lower == "text" and len(node.args) >= 3:
            vx = _num(_literal_to_python(node.args[0]))
            vy = _num(_literal_to_python(node.args[1]))
            vs = _literal_to_python(node.args[2])
            if vx is not None and vy is not None and isinstance(vs, str):
                t = _clean_label(vs)
                if _is_plausible_node_text(t):
                    nodes.append((vx, vy, t))
            continue

        # 2) ax.annotate(s, (x, y), ...) 或 ax.annotate(s, xy=(x, y))
        if lower == "annotate" and node.args:
            vs = _literal_to_python(node.args[0])
            if isinstance(vs, str):
                xy_pair = None
                if len(node.args) >= 2:
                    p = _literal_to_python(node.args[1])
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        xy_pair = (p[0], p[1])
                for kw in node.keywords:
                    if kw.arg == "xy":
                        p = _literal_to_python(kw.value)
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            xy_pair = (p[0], p[1])
                if xy_pair is not None:
                    vx = _num(xy_pair[0])
                    vy = _num(xy_pair[1])
                    if vx is not None and vy is not None:
                        t = _clean_label(vs)
                        if _is_plausible_node_text(t):
                            nodes.append((vx, vy, t))
            continue

        # 3) 自定义绘制：draw_box('txt', x, y, ...) / draw_node / add_node / draw_branch ...
        if any(lower.startswith(p) for p in ("draw_", "add_", "create_", "make_", "plot_")):
            # 尝试顺序：(s, x, y) 或 (x, y, s)
            if len(node.args) >= 3:
                a0 = _literal_to_python(node.args[0])
                a1 = _literal_to_python(node.args[1])
                a2 = _literal_to_python(node.args[2])
                # (s, x, y)
                if isinstance(a0, str) and _num(a1) is not None and _num(a2) is not None:
                    t = _clean_label(a0)
                    if _is_plausible_node_text(t):
                        nodes.append((_num(a1), _num(a2), t))
                    continue
                # (x, y, s)
                if _num(a0) is not None and _num(a1) is not None and isinstance(a2, str):
                    t = _clean_label(a2)
                    if _is_plausible_node_text(t):
                        nodes.append((_num(a0), _num(a1), t))
                    continue

    # 去重（保持首次出现顺序）
    seen: set[str] = set()
    out: list[tuple[float, float, str]] = []
    for x, y, t in nodes:
        if t in seen:
            continue
        seen.add(t)
        out.append((x, y, t))
    return out


def _build_hierarchy_from_positions(nodes: list[tuple[float, float, str]]) -> str:
    """根据节点的 (x, y) 坐标推断思维导图层级。

    算法：
      1) 按 x 坐标做聚类（容差 = x 跨度的 5%），从左到右排出层级 L0, L1, L2, ...；
      2) 同层节点按 y 降序排列（屏幕坐标常以 y 增大向上，或反之——都可；用绝对顺序即可）；
      3) 每个节点的父亲 = 前一层中 y 最接近的节点。
      4) 输出 Markdown 列表。

    若推断失败，返回空串；由调用方走扁平兜底。
    """
    if len(nodes) < 3:
        return ""

    xs = [x for x, _, _ in nodes]
    x_min, x_max = min(xs), max(xs)
    span = x_max - x_min
    if span <= 0:
        return ""  # 所有节点 x 相同，无法分层
    tol = max(span * 0.05, 0.5)

    # 按 x 排序，做合并聚类
    sorted_by_x = sorted(nodes, key=lambda n: n[0])
    layers: list[list[tuple[float, float, str]]] = []
    current_layer: list[tuple[float, float, str]] = []
    current_x: float | None = None
    for nd in sorted_by_x:
        if current_x is None or abs(nd[0] - current_x) <= tol:
            current_layer.append(nd)
            current_x = nd[0] if current_x is None else (current_x + nd[0]) / 2
        else:
            layers.append(current_layer)
            current_layer = [nd]
            current_x = nd[0]
    if current_layer:
        layers.append(current_layer)

    # 若只有 1 层，则退化为扁平列表（放弃层级推断）
    if len(layers) < 2:
        return ""

    # 每层按 y 降序（顶到底）
    for layer in layers:
        layer.sort(key=lambda n: -n[1])

    # 构造父子关系：每个子节点的父 = 上一层中 |y 差| 最小的节点
    # children_map: id(node_idx) -> list of (x, y, text)
    # 用 (x, y, text) 作为 node key（text 唯一已保证）
    children_map: dict[str, list[tuple[float, float, str]]] = {}
    roots = layers[0]  # 第一层即根
    for i in range(1, len(layers)):
        parent_layer = layers[i - 1]
        for child in layers[i]:
            # 找父节点
            parent = min(parent_layer, key=lambda p: abs(p[1] - child[1]))
            children_map.setdefault(parent[2], []).append(child)

    # DFS 输出
    lines: list[str] = []
    visited: set[str] = set()

    def _dfs(node: tuple[float, float, str], level: int) -> None:
        if node[2] in visited:
            return
        visited.add(node[2])
        lines.append(f"{'  ' * level}- {node[2]}")
        for child in children_map.get(node[2], []):
            _dfs(child, level + 1)

    for r in roots:
        _dfs(r, 0)
    return "\n".join(lines)


# ---- 节点文字过滤辅助 ----
_STYLE_TEXT_BLACKLIST = {
    "center",
    "left",
    "right",
    "top",
    "bottom",
    "middle",
    "bold",
    "normal",
    "italic",
    "regular",
    "simhei",
    "simsun",
    "microsoft yahei",
    "source han sans sc",
    "noto sans cjk sc",
    "arial",
    "times new roman",
    "sans-serif",
    "serif",
    "monospace",
    "black",
    "white",
    "gray",
    "grey",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "cyan",
    "magenta",
    "brown",
    "teal",
    "navy",
    "olive",
    "round",
    "square",
    "circle",
    "solid",
    "dashed",
    "dotted",
    "-",
    "--",
    "...",
    "…",
}


def _is_plausible_node_text(t: str) -> bool:
    """判断一个字符串是否可能是思维导图节点文字（而不是样式/配置）。"""
    if not t or len(t) < 1 or len(t) > 120:
        return False
    ls = t.strip().lower()
    if ls in _STYLE_TEXT_BLACKLIST:
        return False
    # 颜色 hex
    if ls.startswith("#") and len(ls) in (4, 7, 9):
        try:
            int(ls[1:], 16)
            return False
        except Exception:
            pass
    # boxstyle 配置串 "round,pad=0.3,rounding_size=0.2"
    if "=" in ls and ("pad" in ls or "rounding" in ls):
        return False
    # 带点的命名空间键 "axes.unicode_minus"
    if ls.count(".") >= 2 and " " not in ls:
        return False
    return True


def _extract_text_string_args(code: str) -> list[str]:
    """从 Python 代码中抓取所有绘文字 API 的字符串参数。

    涵盖的常见调用：
      * ``ax.text(x, y, 'xxx', ...)`` / ``plt.text(x, y, 'xxx', ...)``
      * ``ax.annotate('xxx', ...)`` / ``plt.annotate('xxx', ...)``
      * ``ax.set_title('xxx')`` / ``set_xlabel``/``set_ylabel``
      * 自定义 ``draw_box('xxx', ...)`` / ``draw_node('xxx', ...)``
      * 字典字面量中的 ``'text': 'xxx'`` / ``'name': 'xxx'`` / ``'label': 'xxx'``

    返回去重后的节点文本列表，顺序保持首次出现的顺序。
    """
    if not code:
        return []

    out: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        t = _clean_label(s)
        if not t or t in seen:
            return
        if not _is_plausible_node_text(t):
            return
        seen.add(t)
        out.append(t)

    # 1) 用 AST 精确抓取
    try:
        tree = ast.parse(code)
    except Exception:
        tree = None

    if tree is not None:
        text_func_names = {"text", "annotate", "set_title", "set_xlabel", "set_ylabel"}
        user_draw_prefixes = ("draw_", "add_", "make_", "create_", "plot_")

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id

                lower = func_name.lower()
                is_text_call = (
                    lower in text_func_names
                    or any(lower.startswith(p) for p in user_draw_prefixes)
                    or lower in {"t", "tt"}
                )
                if is_text_call:
                    # 抓该 Call 里所有字符串字面量参数
                    for a in list(node.args) + [kw.value for kw in node.keywords]:
                        v = _literal_to_python(a)
                        if isinstance(v, str):
                            _add(v)

            if isinstance(node, ast.Dict):
                # 字典中 'text'/'name'/'label'/'title'/'value' 键对应的字符串值
                key_hits = {"text", "name", "label", "title", "value", "content", "node"}
                for k_node, v_node in zip(node.keys, node.values):
                    if not isinstance(k_node, ast.Constant) or not isinstance(k_node.value, str):
                        continue
                    if k_node.value.lower() not in key_hits:
                        continue
                    v = _literal_to_python(v_node)
                    if isinstance(v, str):
                        _add(v)

    # 2) 正则兜底：捕获 ``ax.text(... , '字符串' , ...)`` 的第三个位置参数
    # 正则比 AST 弱，但可补充 AST 漏过的动态字符串
    pattern = re.compile(
        r"""(?:ax|plt)\.text\s*\(
            \s*[^,)]+?\s*,       # x
            \s*[^,)]+?\s*,       # y
            \s*(['"])(.+?)\1     # 文本
        """,
        re.VERBOSE | re.DOTALL,
    )
    for m in pattern.finditer(code):
        _add(m.group(2))

    return out


def _extract_networkx_graph(code: str) -> tuple[list[str], list[tuple[str, str]]]:
    """识别 networkx 图构建代码：``G.add_node('A', ...)`` / ``G.add_edge('A', 'B', ...)``。

    同样识别更通用的形式：
      * ``G.add_nodes_from(['A', 'B', ...])``
      * ``G.add_edges_from([('A','B'), ('B','C'), ...])``

    返回 (nodes, edges)；nodes 按首次出现顺序；edges 为 (父, 子) 对。
    对 AST 解析失败的情况，回退到 regex 兜底。
    """
    nodes: list[str] = []
    seen_nodes: set[str] = set()
    edges: list[tuple[str, str]] = []

    def _add_node(n: str) -> None:
        t = _clean_label(n)
        if not t or t in seen_nodes:
            return
        seen_nodes.add(t)
        nodes.append(t)

    def _add_edge(a: str, b: str) -> None:
        ta = _clean_label(a)
        tb = _clean_label(b)
        if not ta or not tb:
            return
        _add_node(ta)
        _add_node(tb)
        edges.append((ta, tb))

    if not code or not code.strip():
        return [], []

    parsed_ok = True
    try:
        tree = ast.parse(code)
    except Exception:
        parsed_ok = False
        tree = None

    if parsed_ok and tree is not None:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            method = ""
            if isinstance(node.func, ast.Attribute):
                method = node.func.attr
            elif isinstance(node.func, ast.Name):
                method = node.func.id
            m = method.lower()

            if m == "add_node" and node.args:
                v = _literal_to_python(node.args[0])
                if isinstance(v, str):
                    _add_node(v)
                elif isinstance(v, (int, float)):
                    _add_node(str(v))
            elif m == "add_nodes_from" and node.args:
                v = _literal_to_python(node.args[0])
                if isinstance(v, (list, tuple)):
                    for x in v:
                        if isinstance(x, str):
                            _add_node(x)
                        elif isinstance(x, (list, tuple)) and x and isinstance(x[0], str):
                            _add_node(x[0])  # (node, attrs) 元组
            elif m == "add_edge" and len(node.args) >= 2:
                a = _literal_to_python(node.args[0])
                b = _literal_to_python(node.args[1])
                if isinstance(a, (str, int, float)) and isinstance(b, (str, int, float)):
                    _add_edge(str(a), str(b))
            elif m == "add_edges_from" and node.args:
                v = _literal_to_python(node.args[0])
                if isinstance(v, (list, tuple)):
                    for pair in v:
                        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                            a, b = pair[0], pair[1]
                            if isinstance(a, (str, int, float)) and isinstance(b, (str, int, float)):
                                _add_edge(str(a), str(b))

    # regex 兜底（AST 失败或补漏）——匹配 add_node/add_edge 的单行调用
    if not parsed_ok or (not nodes and not edges):
        # 匹配 add_node('x', ...) 或 add_node("x", ...)
        for m in re.finditer(r"""\.add_node\s*\(\s*(['"])(.+?)\1""", code):
            _add_node(m.group(2))
        # add_edge('a', 'b', ...)
        for m in re.finditer(r"""\.add_edge\s*\(\s*(['"])(.+?)\1\s*,\s*(['"])(.+?)\3""", code):
            _add_edge(m.group(2), m.group(4))
        # add_edges_from([ ('a','b'), ... ])
        for m in re.finditer(r"""\(\s*(['"])(.+?)\1\s*,\s*(['"])(.+?)\3\s*\)""", code):
            # 此正则比较宽松，但仅在 add_edges_from 关键字附近生效
            pass  # 上面 AST 分支已覆盖，这里不再扩张

    return nodes, edges


# ============================================================
# 嵌套字典 → 树结构（流程图 / 思维导图 兜底）
# ============================================================


def _pick_best_nested_dict(literals: dict[str, Any]) -> dict | None:
    """从 literals 中挑一个"最像层级树"的 dict 变量。

    评分标准：
      * dict 且键数 >= 1；
      * 鼓励值是 dict / list 的嵌套（层级特征）；
      * 过滤纯数值叶子（避免把 ``{"A":[1,2,3]}`` 这种数据字典误认为树）。

    对每个 dict，计算：
      score = (含 dict/list 子节点的键数) * 2 + (字符串叶子数) - (纯数值叶子数)
    取 score 最大且 > 0 的那个。
    """

    def _score(d: dict) -> int:
        s = 0
        for v in d.values():
            if isinstance(v, dict):
                s += 2 + _score(v) // 2  # 嵌套 dict 强信号
            elif isinstance(v, list):
                # list 里全数值 → 低分；含字符串 → 高分
                if not v:
                    continue
                if all(isinstance(x, str) for x in v):
                    s += 2
                elif all(_is_numeric_like(x) for x in v):
                    s -= 1
                else:
                    s += 1
            elif isinstance(v, str) and v.strip():
                s += 1
            elif _is_numeric_like(v):
                s -= 1
        return s

    best: tuple[dict, int] | None = None
    for name, val in literals.items():
        if not isinstance(val, dict) or not val:
            continue
        # 变量名黑名单：常见的绘图配置 dict
        n = name.lower()
        if any(k in n for k in ("style", "color", "cmap", "font", "config", "setting", "option", "param", "kwarg")):
            continue
        sc = _score(val)
        if sc <= 0:
            continue
        if best is None or sc > best[1]:
            best = (val, sc)
    return best[0] if best else None


def _iter_nested_tree(obj: Any, visited: set | None = None) -> list:
    """将 nested dict / list / scalar 展平为层级 list 结构。

    返回形如 ``[(label, children), ...]`` 的树；children 可能为空 list 或更深层的 list。
    list 里的元素作为当前节点的子节点。dict 的 key 作为节点、value 递归展开。
    """
    if visited is None:
        visited = set()

    def _build(x: Any) -> list:
        # 防御深度/循环
        oid = id(x)
        if oid in visited:
            return []
        if isinstance(x, dict):
            visited.add(oid)
            out = []
            for k, v in x.items():
                lbl = str(k).strip()
                if not lbl:
                    continue
                children = _build(v)
                out.append((lbl, children))
            return out
        if isinstance(x, (list, tuple)):
            visited.add(oid)
            out = []
            for item in x:
                sub = _build(item)
                # _build 返回的可能是 [(label, children)] 列表，需展平一层
                if isinstance(sub, list):
                    out.extend(sub)
            return out
        # 标量叶子
        s = str(x).strip() if x is not None else ""
        if not s:
            return []
        return [(s, [])]

    return _build(obj)


def _nested_dict_to_markdown_list(literals: dict[str, Any]) -> str:
    """挑最像树的 nested dict，输出为 Markdown 多级无序列表。

    输出格式与 tree_eval 期望一致：每级缩进 2 空格。
    """
    picked = _pick_best_nested_dict(literals)
    if picked is None:
        return ""

    # 若顶层只有 1 个 key，则 root = 该 key；否则构造虚拟 root（不打印），直接输出子节点
    tree = _iter_nested_tree(picked)
    if not tree:
        return ""

    lines: list[str] = []

    def _dfs(items: list, level: int) -> None:
        for label, children in items:
            lbl = _clean_label(str(label))
            if not _is_plausible_node_text(lbl):
                continue
            lines.append("  " * level + "- " + lbl)
            if children:
                _dfs(children, level + 1)

    _dfs(tree, 0)
    # 至少要有 2 行才算有意义的树
    if len(lines) < 2:
        return ""
    return "\n".join(lines)


def _extract_nested_dict_as_graph(literals: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
    """挑最像树的 nested dict，输出为 (nodes, edges)，父→子有向边。

    节点按 BFS 顺序收集；边去重。
    """
    picked = _pick_best_nested_dict(literals)
    if picked is None:
        return [], []
    tree = _iter_nested_tree(picked)
    if not tree:
        return [], []

    nodes: list[str] = []
    seen: set[str] = set()
    edges: list[tuple[str, str]] = []
    edge_seen: set[tuple[str, str]] = set()

    def _add_node(n: str) -> None:
        t = _clean_label(n)
        if not t or t in seen or not _is_plausible_node_text(t):
            return
        seen.add(t)
        nodes.append(t)

    def _add_edge(a: str, b: str) -> None:
        ta = _clean_label(a)
        tb = _clean_label(b)
        if not ta or not tb:
            return
        if not (_is_plausible_node_text(ta) and _is_plausible_node_text(tb)):
            return
        _add_node(ta)
        _add_node(tb)
        key = (ta, tb)
        if key in edge_seen:
            return
        edge_seen.add(key)
        edges.append(key)

    def _dfs(items: list, parent: str | None) -> None:
        for label, children in items:
            lbl = str(label).strip()
            if not lbl:
                continue
            _add_node(lbl)
            if parent is not None:
                _add_edge(parent, lbl)
            if children:
                _dfs(children, lbl)

    _dfs(tree, None)
    return nodes, edges
