"""methods 包：评分流程各层模块。

结构：
    context.py          线程上下文 + 代码围栏剥离
    parsers/            各格式解析器（Markdown/JSON/Python代码/SVG/饼图）
    normalize.py        归一化入口（按 task 将 prediction 转为统一格式）
    scoring.py          评分路由 + JUDGE_FUNC 注册表
    prompts.py          推理 prompt 模板
"""
