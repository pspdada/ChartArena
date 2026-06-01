"""parsers 包：各格式解析器（Markdown/JSON/Python代码/SVG/饼图）。"""

from .parse_json import (
    adjacency_csv_to_markdown_list,
    json_to_internal_csv,
    json_tree_to_markdown_list,
)
from .parse_markdown import (
    html_table_to_csv,
    is_csv_format,
    is_html_table,
    is_markdown_table,
    is_pipe_table,
    is_standard_csv,
    markdown_to_csv,
    pipe_table_to_csv,
    standard_csv_to_internal,
)
from .parse_pie import normalize_pie_prediction
from .parse_python_code import (
    python_code_to_internal_csv,
    python_code_to_markdown_list,
    python_code_to_mermaid,
)
from .parse_svg import svg_to_internal_csv, svg_to_markdown_list

__all__ = [
    # parse_json
    "adjacency_csv_to_markdown_list",
    "json_to_internal_csv",
    "json_tree_to_markdown_list",
    # parse_markdown
    "html_table_to_csv",
    "is_csv_format",
    "is_html_table",
    "is_markdown_table",
    "is_pipe_table",
    "is_standard_csv",
    "markdown_to_csv",
    "pipe_table_to_csv",
    "standard_csv_to_internal",
    # parse_pie
    "normalize_pie_prediction",
    # parse_python_code
    "python_code_to_internal_csv",
    "python_code_to_markdown_list",
    "python_code_to_mermaid",
    # parse_svg
    "svg_to_internal_csv",
    "svg_to_markdown_list",
]
