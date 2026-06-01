"""dsl_parsers 包内共享的基础类型和工具函数。

不依赖包内任何其他模块，避免循环导入。
"""

from __future__ import annotations

import re
from typing import Literal

# 图 IR 类型
Graph = tuple[dict[str, str], list[tuple[str, str]], list[tuple[str, str, str]]]

# DSL 类型字面量
DSLType = Literal["mermaid", "dot", "plantuml", "diagrams", "d2", "cytoscape"]


def _empty_graph() -> Graph:
    return {}, [], []


def strip_code_fence(text: str, lang_hints: tuple[str, ...]) -> str:
    """去掉首尾 ```<lang> ... ``` 包裹。lang_hints 为该 DSL 常见的围栏语言标记。"""
    if not text:
        return ""
    t = text.strip()
    for lang in lang_hints:
        pat_open = re.compile(rf"^```\s*{re.escape(lang)}\s*\n?", flags=re.IGNORECASE)
        m = pat_open.match(t)
        if m:
            t = t[m.end() :]
            t = re.sub(r"\n?```\s*$", "", t)
            return t.strip()
    # 退化到任意 ```...``` 围栏
    m = re.match(r"^```[a-zA-Z0-9_\-]*\s*\n?(.*?)```\s*$", t, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return t
