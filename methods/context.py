"""线程局部上下文 + 代码围栏剥离（被多个子模块复用）。"""

import re
import sys
import threading
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# ============================================================
# 线程局部上下文：用于把 chart_type 等 hint 从入口传递到 python_code_to_internal_csv
# 等深层函数，无需修改这些函数的签名。
# ============================================================
_judge_ctx = threading.local()


def set_judge_context(chart_type: str = "") -> None:
    _judge_ctx.chart_type = chart_type or ""


def get_chart_type() -> str:
    return getattr(_judge_ctx, "chart_type", "") or ""


def strip_code_fence(text: str, lang: str | None = None) -> str:
    """去掉 ```xxx ... ``` 代码块围栏，返回纯文本。

    Args:
        text: 原始文本
        lang: 指定期望的代码语言（json/csv/markdown 等），None 表示任意

    Returns:
        去掉围栏后的文本（若未命中则原样返回）
    """
    if not text:
        return ""
    t = text.strip()
    # 匹配 ```lang\n...\n```
    if lang:
        pattern = rf"^```(?:{lang}|{lang.upper()})?\s*\n?(.*?)\n?```\s*$"
    else:
        pattern = r"^```[a-zA-Z0-9_\-]*\s*\n?(.*?)\n?```\s*$"
    m = re.match(pattern, t, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return t
