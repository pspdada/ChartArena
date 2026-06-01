"""向后兼容入口：从 metrics.dsl_parsers 重新导出所有公开符号。

原始实现已拆分到 metrics/dsl_parsers/ 包中：
    _types.py     — 共享类型（Graph、DSLType、strip_code_fence）
    dot.py        — Graphviz DOT 解析器
    plantuml.py   — PlantUML 解析器
    diagrams.py   — mingrammer Diagrams 解析器
    d2.py         — D2 解析器
    cytoscape.py  — Cytoscape.js JSON 解析器
    __init__.py   — 路由（parse_flowchart）+ 评估入口（flowchart_eval_multi）

外部代码（如 methods/scoring.py）无需修改，继续从本模块导入即可。
"""

from .dsl_parsers import (  # noqa: F401
    DSLType,
    Graph,
    flowchart_eval_multi,
    flowchart_similarity,
    flowchart_similarity_graph,
    is_cytoscape,
    is_d2,
    is_diagrams,
    is_dot,
    is_mermaid,
    is_plantuml,
    parse_cytoscape,
    parse_d2,
    parse_diagrams,
    parse_dot,
    parse_flowchart,
    parse_mermaid,
    parse_plantuml,
    strip_code_fence,
)
from .dsl_parsers._types import _empty_graph  # noqa: F401

__all__ = [
    "Graph",
    "DSLType",
    "strip_code_fence",
    "is_dot",
    "is_plantuml",
    "is_diagrams",
    "is_d2",
    "is_cytoscape",
    "parse_dot",
    "parse_plantuml",
    "parse_diagrams",
    "parse_d2",
    "parse_cytoscape",
    "parse_flowchart",
    "flowchart_similarity_graph",
    "flowchart_eval_multi",
    "is_mermaid",
    "parse_mermaid",
    "flowchart_similarity",
]
