"""
图表结构化提取推理 prompt 模板（0417 版本）

定义 SE_MD / SE_JSON / SE_CSV 三个任务的 prompt，每个任务根据图表类别
（数值类 data / 逻辑结构 logic / 流程图 flowchart）分别路由到不同 prompt。

可扩展性：
- 新增任务：在 `TASKS` 中追加任务名，并在 `INFER_PROMPT_TEMPLATES` 中补充各类别的 prompt；
- 新增图表类别：在 `CHART_CATEGORY_MAP` 中追加图表类型 → 类别的映射，
  并在 `INFER_PROMPT_TEMPLATES[task]` 中补充该类别的 prompt。
"""

from typing import Callable

SE_MD_PROMPT_DATA = """\
请解析图像中的图表内容，并将其中的数据提取为结构化的 Markdown **表格**格式。

具体要求如下：
1. **关注图表本身**：忽略页面中的其他元素（如装饰、背景、logo、水印等）；
2. **表头格式规范**：如果图表中同时存在类别标签和数值单位（如 Y 轴标注），请将两者以 “类别-单位” 的格式合并作为表头；
3. **标签保持原样**：对于图表中的类别标签，保持其原样，不要进行翻译或改写；
4. **数值保持精度**：对于图表中的数值，保持其原始语义与精度。"""

SE_MD_PROMPT_LOGIC = """\
请解析图像中的图表内容，并将其中的数据提取为结构化的 Markdown **多级无序列表**格式。

具体要求如下：
1. **输出格式**：使用 `-` 开头的无序列表格式，每个节点的文本内容作为列表项的文本；
2. **层级关系**：根据节点之间的连接关系确定列表项的层级，父节点对应一级列表项，子节点对应二级列表项，以此类推；
3. **内容完整**：完整提取每个框/节点内的文字，保持其原始语言和标点符号。"""


SE_JSON_PROMPT_DATA = """\请解析图像中的图表内容，并将其中的数据提取为结构化的 **JSON** 格式。

具体要求如下：
1. **输出结构**：返回一个 JSON 字典，必须包含 `title` 和 `values` 两个键。`values` 下的结构应以数据类别（含单位）为键，以具体的“维度-数值”映射字典为值；
2. **关注图表本身**：忽略页面中的其他非数据相关元素（如装饰、背景、logo、水印等）；
3. **标签保持原样**：对于图表中的各类标签（如标题、图例、轴标签），保持其原样提取，绝对不要进行翻译或自行改写；
4. **数值保持精度**：对于图表中的数值，保持其原始语义与精度。"""


SE_JSON_PROMPT_LOGIC = """\
请解析图像中的思维导图或逻辑结构图，并将其内容提取为树状结构的 **JSON** 格式。

具体要求如下：
1. **输出结构**：返回一个嵌套的 JSON 字典，每个节点必须包含 `name` 键（表示该节点的文本内容）。如果该节点有下级分支，必须包含 `children` 键，其值为包含下级节点字典的数组；如果该节点是叶子节点（无下级分支），则无需输出 `children` 键；
2. **层级关系**：严格根据图像中节点之间的连接线条或包含关系来确定 JSON 的嵌套层级。图的中心主题或根节点作为 JSON 的最外层对象；
3. **内容完整**：完整提取每个框/节点内的文字，保持其原始语言和标点符号，不要进行任何主观的精简、总结或补充。"""


SE_CSV_PROMPT_DATA = """\
请解析图像中的图表内容，并将其中的数据提取为结构化的 **CSV** 格式。

具体要求如下：
1. **输出格式**：返回标准的纯文本 CSV 格式数据，以英文逗号 `,` 作为分隔符，第一行为表头；
2. **表头格式规范**：如果图表中同时存在类别标签和数值单位（如 Y 轴标注），请将两者以 “类别-单位” 的格式合并作为该列的表头；
3. **关注图表本身**：忽略页面中的其他非数据相关元素（如装饰、背景、logo、水印等）；
4. **标签与精度**：对于图表中的类别标签保持原样（不翻译、不改写），对于数值保持其原始语义与精度。"""

SE_CSV_PROMPT_LOGIC = """\
请解析图像中的思维导图或逻辑结构图，并将其内容提取为基于父子节点关联（邻接表）的 **CSV** 格式。

具体要求如下：
1. **输出结构**：返回标准的纯文本 CSV 格式数据，必须且仅包含三列，表头为：`id,parent_id,name`；
2. **层级与 ID 分配**：
   - `id`：每个节点分配一个唯一的正整数 ID，从 1 开始递增；
   - `parent_id`：表示该节点的父节点 ID。最中心的主题节点（根节点）的 `id` 必须为 1，且其 `parent_id` 必须为 0；
   - `name`：代表节点内的文本内容；
3. **内容完整**：完整提取每个框/节点内的文字，保持其原始语言和标点符号，绝对不要进行主观精简或总结。"""

SE_CODE_PROMPT_DATA = """\
你是一名精通数据可视化的 Python 开发者，擅长基于给定图片编写可复现的 matplotlib 代码。我在一篇 STEM 论文中看到一张制作精良的图表，但没有对应的源代码，请你帮我生成可复现该图的 matplotlib Python 代码。

具体要求如下：
1. **图像尺寸**：必须使用 `figsize=(width, height)` 设置画布尺寸，使其与原图宽高比尽量一致；
2. **数据自提取**：我不会提供原始数据，你需要仔细观察图像并尽可能精确地估计/还原出图中的实际数据（包括坐标值、类别、数值、误差棒等），并在代码中显式定义为常量或 Python 数据结构（`list`、`dict`、`numpy.ndarray` 等）；
3. **可直接运行**：代码必须是**完整的、自包含的、可直接运行**的 Python 脚本，无需我额外提供任何变量或文件；仅允许使用常见库（如 `matplotlib`、`numpy`、`pandas`），并在开头写好 `import` 语句；
4. **还原视觉元素**：尽可能还原图中的标题、坐标轴标签、图例、刻度、网格、配色方案、数据标签、注释等关键视觉元素；
5. **标签保持原样**：对于图中的类别标签、标题、图例等文本，保持其原始语言与措辞，不要翻译或改写；
6. **输出格式**：只输出 Python 代码本身，用 ```python 代码块包裹，不要在代码块外添加任何解释或说明性文字。"""

SE_CODE_PROMPT_LOGIC = """\
你是一名精通 Python 绘图的开发者，擅长基于给定图片编写可复现的代码。请基于我提供的思维导图 / 逻辑结构图，生成可复现该图的 Python 代码。

具体要求如下：
1. **绘图库选择**：优先使用 `matplotlib` + `networkx` 来绘制节点与连接关系；若图像更适合树形布局，也可使用 `matplotlib` 的 `Rectangle` / `Annotation` 手工绘制；
2. **画布尺寸**：必须使用 `figsize=(width, height)` 设置画布尺寸，使其与原图宽高比尽量一致；
3. **结构完整**：完整提取每个框 / 节点内的文字，保持其原始语言和标点符号，并根据图中的连接线条或包含关系精确还原父子层级；
4. **显式输出结构数据（重要）**：在代码开头，必须先**显式定义两个 Python 列表字面量**，用于声明图中所有节点与父子关系，后续绘图必须基于这两个变量：
   - `nodes`：节点列表，元素为节点文本字符串，例如 `nodes = ["根节点", "子节点A", "子节点B", ...]`；
   - `edges`：父子关系列表，元素为 `(父节点文本, 子节点文本)` 的二元组，例如 `edges = [("根节点", "子节点A"), ("根节点", "子节点B"), ("子节点A", "孙节点A1"), ...]`；
   - 节点文本必须与图中文字**逐字一致**，父子关系必须覆盖图中**所有**连接线；
   - 绝对不要只在 `ax.text(x, y, "...")` 里散落书写节点文字而不定义 `edges`；
5. **可直接运行**：代码必须是**完整的、自包含的、可直接运行**的 Python 脚本，无需我额外提供任何变量或文件；仅允许使用常见库，并在开头写好 `import` 语句；
6. **还原视觉元素**：尽可能还原图中的配色、节点形状、字体大小、连接线样式等关键视觉元素；
7. **标签保持原样**：对于节点文本保持其原始语言与措辞，不要翻译或改写；
8. **输出格式**：只输出 Python 代码本身，用 ```python 代码块包裹，不要在代码块外添加任何解释或说明性文字。

**示例骨架**（仅用于说明 `nodes` / `edges` 的写法，请根据实际图像替换内容）：
```python
import matplotlib.pyplot as plt
import networkx as nx

# 1) 先声明节点与父子关系（必须）
nodes = ["主题", "分支A", "分支B", "分支A-1", "分支A-2"]
edges = [
    ("主题", "分支A"),
    ("主题", "分支B"),
    ("分支A", "分支A-1"),
    ("分支A", "分支A-2"),
]

# 2) 再基于 nodes / edges 绘图
G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
# ... 绘图细节 ...
```"""

SE_SVG_PROMPT_DATA = """\
你是一名精通 SVG 与数据可视化的前端开发者，擅长基于给定图片手写可渲染的 **SVG 代码**。请基于我提供的图表图片，生成可复现该图的 SVG 代码。

具体要求如下：
1. **根元素规范**：必须输出完整的 `<svg ...>...</svg>` 代码，根元素同时包含 `width`、`height` 与 `viewBox="0 0 {width} {height}"`，并保持宽高比与原图一致；
2. **数据自提取**：我不会提供原始数据，请仔细观察图像并尽可能精确地估计/还原出图中的实际数据（包括坐标轴刻度、类别、数值、误差棒等），直接以 SVG 图形元素（`<rect>`、`<circle>`、`<line>`、`<polyline>`、`<path>`、`<text>` 等）表达；
3. **完整的文本元素**：必须使用 `<text>` 元素完整保留图表中的**所有文字**，包括**标题、坐标轴标签（含单位）、坐标刻度值、类别名称、数据标签、图例文字**等，保持其原始语言与措辞，不要翻译或改写；**每条文字（类别名、数值、图例项等）应放在一个完整的 `<text>` 里**，不要把 "Teachers' salaries"、"Normal (n=18)" 这种同一条标签拆到多个 `<text>` 或多个没有 `dy` 偏移的 `<tspan>` 里；确需换行时请使用带 `dy` 偏移的 `<tspan>`；
4. **【关键】每个数据点必须有独立的数值 `<text>`**：无论原图是否显式标注数值，你都**必须**为图中的每一个数据点单独输出一个 `<text>` 元素，将你所估计/还原出的数值直接作为文字写出，并放置在对应数据点旁边（与图元几何位置对应）。具体要求如下：
   - **柱状图 / 条形图**：每根柱子的顶部（水平条形图则右端）必须有一个 `<text>` 元素写出该柱的数值；
   - **折线图**：每个折线数据点处（`<circle>` 或折点）必须有一个 `<text>` 元素写出该点数值；
   - **饼图**：每一片扇区内或旁边必须有一个 `<text>` 写出**百分比或数值**（与原图表一致）；
   - **箱线图**：每个箱的**最小值、Q1、中位数、Q3、最大值**共 5 个统计量，**每个都必须单独输出一个 `<text>` 元素**写在对应位置旁边（如最小值/最大值的 whisker 端点、Q1/Q3 的箱体上下沿、中位数的水平线旁）；
   - **雷达图**：每个坐标轴上数据点处必须有一个 `<text>` 写出该轴的数值；若多个系列共存，每个系列每个轴都要，并建议在数值 `<text>` 上加 `data-series="系列名"` 属性区分；
   - **组合图 / 双轴图**：柱和折线点两类都按上述要求写出数值 `<text>`；
   这些数值 `<text>` **不可省略**，且必须与几何元素（`<rect>`、`<polyline>` 等）并存，不得仅用几何位置隐含表达；
5. **还原视觉元素**：尽可能还原图中的配色（用 `fill`/`stroke` 属性）、坐标轴、网格线、图例等关键视觉元素；
6. **可直接渲染**：输出的 SVG 必须是**完整的、自包含的、可直接渲染**的代码，不依赖任何外部资源（不要使用 `<image href=...>` 等引用外部文件的元素）；
7. **输出格式**：只输出 SVG 代码本身，用 ```xml 代码块包裹，不要在代码块外添加任何解释或说明性文字。"""

SE_SVG_PROMPT_LOGIC = """\
你是一名精通 SVG 与信息图的前端开发者，擅长基于给定图片手写可渲染的 **SVG 代码**。请基于我提供的思维导图 / 逻辑结构图，生成可复现该图的 SVG 代码。

具体要求如下：
1. **根元素规范**：必须输出完整的 `<svg ...>...</svg>` 代码，根元素同时包含 `width`、`height` 与 `viewBox="0 0 {width} {height}"`，并保持宽高比与原图一致；
2. **节点绘制**：使用 `<rect>` / `<ellipse>` / `<g>` 绘制每个节点的边框，并用 `<text>` 元素完整写出节点内的文字，保持其原始语言和标点符号，不要进行任何主观的精简或改写；
3. **【关键】层级关系显式标注**：为便于机器解析父子层级，你**必须**把每一个节点（包括图形边框 + 文字）用一个单独的 `<g>` 容器包裹，并在 `<g>` 上添加 `data-parent="父节点文本"` 属性（父节点文本须与该父节点 `<text>` 内完整文字**完全一致**）。具体要求：
   - **根节点**：`data-parent=""`（空字符串）；
   - **非根节点**：`data-parent="其直接父节点的文本"`；
   - 每个 `<g data-parent="...">` 内部**有且仅有一个表征该节点自身的 `<text>`**（可以同时包含边框矩形/圆形等）；若节点文字被拆成多行，请合并到同一个 `<text>` 中或用 `<tspan>` 子节点书写；
   - 示例：
   ```xml
   <g data-parent=""><rect .../><text>Tips for your garage sale</text></g>
   <g data-parent="Tips for your garage sale"><rect .../><text>a garage sale</text></g>
   <g data-parent="a garage sale"><rect .../><text>a great way to reduce the amount of things</text></g>
   ```
4. **连接关系**：使用 `<line>` / `<path>` / `<polyline>` 绘制节点之间的连接线，必须覆盖图中**所有**父子/关联关系；
5. **还原视觉元素**：尽可能还原图中的配色（用 `fill`/`stroke` 属性）、节点形状、字体大小、连接线样式等关键视觉元素；
6. **可直接渲染**：输出的 SVG 必须是**完整的、自包含的、可直接渲染**的代码，不依赖任何外部资源；
7. **输出格式**：只输出 SVG 代码本身，用 ```xml 代码块包裹，不要在代码块外添加任何解释或说明性文字。"""


# ============================================================
# SE_MERMAID：流程图 → Mermaid flowchart 代码
# ============================================================

SE_MERMAID_PROMPT_FLOWCHART = """\
请仔细观察下面这张**流程图**（Flowchart）图片，将其完整转写为 **Mermaid** 流程图代码。

要求：
1. 使用 Mermaid 的 `flowchart` / `graph` 语法（推荐 `flowchart TD` 或 `flowchart LR`，按图中实际方向选择）；
2. 严格还原图中的**节点文字**，保持原始语言与标点符号，不要翻译、不要改写、不要简化；
3. 正确还原节点之间的**连接关系与方向**（箭头走向），如有条件分支请保留分支上的**标签文字**（例如 `是 / 否 / Yes / No`）；
4. 忽略图中的装饰、背景、logo、水印等与流程无关的内容；
5. **输出结构严格分两段**：
   - 先逐行**声明所有节点**，不要在声明行写箭头；
   - 再逐行**声明节点之间的关系**，不要在关系行里重复节点文字；
6. 只输出一个 Mermaid 代码块，不要添加额外说明文字。

输出示例：

```mermaid
flowchart TD
    A["客户询盘与报价"]
    B["销售合同谈判"]
    C["签订销售合同"]
    D{"合同条款有效?"}
    E["确认支付方式"]
    F["备货与生产"]
    G["商品检验与报关"]
    H{"报关单据完整?"}
    I["安排国际运输"]
    J["补正单据"]
    K["重新谈判/取消"]
    A --> B
    B --> C
    C --> D
    D -->|有效| E
    D -->|无效| K
    E --> F
    F --> G
    G --> H
    H -->|完整| I
    H -->|不完整| J
    J --> G
```"""


# ============================================================
# SE_GRAPHVIZ：流程图 → Graphviz DOT 代码
# ============================================================

SE_GRAPHVIZ_PROMPT_FLOWCHART = """\
请仔细观察下面这张**流程图**（Flowchart）图片，将其完整转写为 **Graphviz DOT** 代码。

要求：
1. 使用 `digraph` 声明（如 `digraph G { ... }`），按图中箭头方向用 `->` 表达有向边；
2. 每个节点必须显式定义 `label` 属性，如 `A [label="客户询盘与报价"]`；**节点 id 可以是任意唯一标识**，但 `label` 必须与图中文字**逐字一致**（保持原始语言与标点），不要翻译、不要改写、不要简化；
3. 条件分支上的**标签文字**（例如 `是 / 否 / Yes / No`）放在边的 `label` 属性里，如 `D -> E [label="有效"]`；
4. 忽略图中的装饰、背景、logo、水印等与流程无关的内容；
5. **输出结构严格分两段**：
   - 先逐行**声明所有节点**（`id [label="..."]`），不要在声明行里写箭头；
   - 再逐行**声明边**（`src -> dst [label="..."]`），不要在边行里再给节点加 label；
6. 只输出一个 DOT 代码块，不要添加额外说明文字。

输出示例：

```dot
digraph G {
    A [label="客户询盘与报价"];
    B [label="销售合同谈判"];
    C [label="签订销售合同"];
    D [label="合同条款有效?"];
    E [label="确认支付方式"];
    F [label="备货与生产"];
    G [label="商品检验与报关"];
    H [label="报关单据完整?"];
    I [label="安排国际运输"];
    J [label="补正单据"];
    K [label="重新谈判/取消"];
    A -> B;
    B -> C;
    C -> D;
    D -> E [label="有效"];
    D -> K [label="无效"];
    E -> F;
    F -> G;
    G -> H;
    H -> I [label="完整"];
    H -> J [label="不完整"];
    J -> G;
}
```"""


# ============================================================
# SE_PLANTUML：流程图 → PlantUML activity 代码
# ============================================================

SE_PLANTUML_PROMPT_FLOWCHART = """\
请仔细观察下面这张**流程图**（Flowchart）图片，将其完整转写为 **PlantUML** 活动图（activity diagram）代码。

要求：
1. 使用 `@startuml ... @enduml` 包裹；主体使用 **activity 新语法**：`:节点文字;` 表示活动节点，`if (...) then (...) / else (...) / endif` 表示条件分支；
2. **节点文字**必须与图中**逐字一致**（保持原始语言与标点），不要翻译、不要改写、不要简化；
3. 条件判断节点放在 `if (...)` 的小括号里，分支上的**标签文字**（例如 `是 / 否 / Yes / No`）放在 `then (...)` / `else (...)` 的小括号里；
4. 忽略图中的装饰、背景、logo、水印等与流程无关的内容；
5. 只输出一个 PlantUML 代码块（```plantuml ... ```），不要添加额外说明文字。

输出示例：

```plantuml
@startuml
start
:客户询盘与报价;
:销售合同谈判;
:签订销售合同;
if (合同条款有效?) then (有效)
  :确认支付方式;
  :备货与生产;
  :商品检验与报关;
  if (报关单据完整?) then (完整)
    :安排国际运输;
  else (不完整)
    :补正单据;
  endif
else (无效)
  :重新谈判/取消;
endif
stop
@enduml
```"""


# ============================================================
# SE_DIAGRAMS：流程图 → mingrammer diagrams (Python DSL)
# ============================================================

SE_DIAGRAMS_PROMPT_FLOWCHART = """\
请仔细观察下面这张**流程图**（Flowchart）图片，将其完整转写为 **mingrammer diagrams** 的 Python DSL 代码。

要求：
1. 使用 `from diagrams import Diagram, Edge` 引入，并在 `with Diagram(...)` 上下文中声明节点与连接关系；允许使用 `diagrams.programming.flowchart` 下的节点类（如 `Action / Decision / StartEnd`）；若不确定节点类别，**统一用 `Action("文字")` 即可**；
2. 每个节点必须使用变量绑定，如 `a = Action("客户询盘与报价")`；**括号内的字符串必须与图中文字逐字一致**（保持原始语言与标点），不要翻译、不要改写、不要简化；
3. 用 `>>` 表达有向连接；条件分支上的**标签文字**（例如 `是 / 否 / Yes / No`）写成 `a >> Edge(label="有效") >> b`；
4. 忽略图中的装饰、背景、logo、水印等与流程无关的内容；
5. **输出结构严格分两段**：
   - 先用 `变量 = Action("文字")` 逐行声明所有节点；
   - 再用 `a >> b` / `a >> Edge(label="xxx") >> b` 逐行声明边；
6. 只输出一个 Python 代码块（```python ... ```），不要添加额外说明文字。

输出示例：

```python
from diagrams import Diagram, Edge
from diagrams.programming.flowchart import Action, Decision

with Diagram("flowchart", show=False):
    a = Action("客户询盘与报价")
    b = Action("销售合同谈判")
    c = Action("签订销售合同")
    d = Decision("合同条款有效?")
    e = Action("确认支付方式")
    f = Action("备货与生产")
    g = Action("商品检验与报关")
    h = Decision("报关单据完整?")
    i = Action("安排国际运输")
    j = Action("补正单据")
    k = Action("重新谈判/取消")
    a >> b
    b >> c
    c >> d
    d >> Edge(label="有效") >> e
    d >> Edge(label="无效") >> k
    e >> f
    f >> g
    g >> h
    h >> Edge(label="完整") >> i
    h >> Edge(label="不完整") >> j
    j >> g
```"""


# ============================================================
# SE_D2：流程图 → D2 (terrastruct)
# ============================================================

SE_D2_PROMPT_FLOWCHART = """\
请仔细观察下面这张**流程图**（Flowchart）图片，将其完整转写为 **D2** 流程图代码（terrastruct/d2）。

要求：
1. 使用 D2 的基本语法：`id: "节点文字"` 声明节点，`src -> dst: "边标签"` 声明有向边；
2. 严格还原图中的**节点文字**，保持原始语言与标点符号，不要翻译、不要改写、不要简化；
3. 正确还原节点之间的**连接关系与方向**，条件分支上的标签文字（例如 `是 / 否 / Yes / No`）放在边的冒号后（如 `d -> e: "有效"`）；
4. 判断节点可以可选地加 `x.shape: diamond`，但这不是必须的；
5. 忽略图中的装饰、背景、logo、水印等与流程无关的内容；
6. **输出结构严格分两段**：
   - 先逐行**声明所有节点**（`id: "文字"`），不要在声明行写箭头；
   - 再逐行**声明边**（`src -> dst` 或 `src -> dst: "标签"`），不要在边行里再给节点加 label；
7. 只输出一个 D2 代码块（```d2 ... ```），不要添加额外说明文字。

输出示例：

```d2
a: "客户询盘与报价"
b: "销售合同谈判"
c: "签订销售合同"
d: "合同条款有效?"
e: "确认支付方式"
f: "备货与生产"
g: "商品检验与报关"
h: "报关单据完整?"
i: "安排国际运输"
j: "补正单据"
k: "重新谈判/取消"
d.shape: diamond
h.shape: diamond
a -> b
b -> c
c -> d
d -> e: "有效"
d -> k: "无效"
e -> f
f -> g
g -> h
h -> i: "完整"
h -> j: "不完整"
j -> g
```"""


# ============================================================
# SE_CYTOSCAPE：流程图 → Cytoscape.js JSON
# ============================================================

SE_CYTOSCAPE_PROMPT_FLOWCHART = """\
请仔细观察下面这张**流程图**（Flowchart）图片，将其完整转写为 **Cytoscape.js JSON** 格式的流程图数据。

要求：
1. 输出一个合法的 JSON 对象，顶层必须包含 `elements` 字段；`elements` 采用 **`{"nodes": [...], "edges": [...]}` 分块结构**；
2. 每个节点形如 `{"data": {"id": "a", "label": "客户询盘与报价"}}`；**`id` 可自由命名但需全局唯一**，`label` 必须与图中文字**逐字一致**（保持原始语言与标点），不要翻译、不要改写、不要简化；
3. 每条边形如 `{"data": {"source": "a", "target": "b"}}`，条件分支上的**标签文字**（例如 `是 / 否 / Yes / No`）放在 `label` 字段里，如 `{"data": {"source": "d", "target": "e", "label": "有效"}}`；
4. 忽略图中的装饰、背景、logo、水印等与流程无关的内容；
5. 只输出一个 JSON 代码块（```json ... ```），不要添加额外说明文字。

输出示例：

```json
{
  "elements": {
    "nodes": [
      {"data": {"id": "a", "label": "客户询盘与报价"}},
      {"data": {"id": "b", "label": "销售合同谈判"}},
      {"data": {"id": "c", "label": "签订销售合同"}},
      {"data": {"id": "d", "label": "合同条款有效?"}},
      {"data": {"id": "e", "label": "确认支付方式"}},
      {"data": {"id": "f", "label": "备货与生产"}},
      {"data": {"id": "g", "label": "商品检验与报关"}},
      {"data": {"id": "h", "label": "报关单据完整?"}},
      {"data": {"id": "i", "label": "安排国际运输"}},
      {"data": {"id": "j", "label": "补正单据"}},
      {"data": {"id": "k", "label": "重新谈判/取消"}}
    ],
    "edges": [
      {"data": {"source": "a", "target": "b"}},
      {"data": {"source": "b", "target": "c"}},
      {"data": {"source": "c", "target": "d"}},
      {"data": {"source": "d", "target": "e", "label": "有效"}},
      {"data": {"source": "d", "target": "k", "label": "无效"}},
      {"data": {"source": "e", "target": "f"}},
      {"data": {"source": "f", "target": "g"}},
      {"data": {"source": "g", "target": "h"}},
      {"data": {"source": "h", "target": "i", "label": "完整"}},
      {"data": {"source": "h", "target": "j", "label": "不完整"}},
      {"data": {"source": "j", "target": "g"}}
    ]
  }
}
```"""


# ============================================================
# 图表类别路由
# ============================================================

# 图表类别常量
CATEGORY_DATA = "data"  # 数值类图表
CATEGORY_LOGIC = "logic"  # 逻辑结构图（思维导图等）
CATEGORY_FLOWCHART = "flowchart"  # 流程图

# 默认类别（未知 chart_type 时使用）
DEFAULT_CHART_CATEGORY = CATEGORY_DATA

# chart_type（中文）→ 类别
CHART_CATEGORY_MAP: dict[str, str] = {
    # 数值类图表
    "柱状图": CATEGORY_DATA,
    "折线图": CATEGORY_DATA,
    "饼图": CATEGORY_DATA,
    "箱线图": CATEGORY_DATA,
    "雷达图": CATEGORY_DATA,
    "组合图": CATEGORY_DATA,
    # 逻辑结构图
    "思维导图": CATEGORY_LOGIC,
    # 流程图
    "流程图": CATEGORY_FLOWCHART,
}


def get_chart_category(chart_type: str) -> str:
    """根据图表类型（中文）获取类别。

    未命中时返回 DEFAULT_CHART_CATEGORY。
    """
    if not chart_type:
        return DEFAULT_CHART_CATEGORY
    return CHART_CATEGORY_MAP.get(chart_type, DEFAULT_CHART_CATEGORY)


# ============================================================
# prompt 模板映射：{任务名: {图表类别: prompt 模板}}
# ============================================================

INFER_PROMPT_TEMPLATES: dict[str, dict[str, str]] = {
    "SE_MD": {
        CATEGORY_DATA: SE_MD_PROMPT_DATA,
        CATEGORY_LOGIC: SE_MD_PROMPT_LOGIC,
    },
    "SE_JSON": {
        CATEGORY_DATA: SE_JSON_PROMPT_DATA,
        CATEGORY_LOGIC: SE_JSON_PROMPT_LOGIC,
    },
    "SE_CSV": {
        CATEGORY_DATA: SE_CSV_PROMPT_DATA,
        CATEGORY_LOGIC: SE_CSV_PROMPT_LOGIC,
    },
    "SE_CODE": {
        CATEGORY_DATA: SE_CODE_PROMPT_DATA,
        CATEGORY_LOGIC: SE_CODE_PROMPT_LOGIC,
    },
    "SE_SVG": {
        CATEGORY_DATA: SE_SVG_PROMPT_DATA,
        CATEGORY_LOGIC: SE_SVG_PROMPT_LOGIC,
    },
    # SE_MERMAID / SE_GRAPHVIZ / SE_PLANTUML / SE_DIAGRAMS 仅针对流程图类别生效；
    # 其他类别没有映射，get_prompts_for_api 会自然跳过
    "SE_MERMAID": {
        CATEGORY_FLOWCHART: SE_MERMAID_PROMPT_FLOWCHART,
    },
    "SE_GRAPHVIZ": {
        CATEGORY_FLOWCHART: SE_GRAPHVIZ_PROMPT_FLOWCHART,
    },
    "SE_PLANTUML": {
        CATEGORY_FLOWCHART: SE_PLANTUML_PROMPT_FLOWCHART,
    },
    "SE_DIAGRAMS": {
        CATEGORY_FLOWCHART: SE_DIAGRAMS_PROMPT_FLOWCHART,
    },
    "SE_D2": {
        CATEGORY_FLOWCHART: SE_D2_PROMPT_FLOWCHART,
    },
    "SE_CYTOSCAPE": {
        CATEGORY_FLOWCHART: SE_CYTOSCAPE_PROMPT_FLOWCHART,
    },
}

# ============================================================
# 流程图任务集合常量
# ============================================================
FLOWCHART_TASKS: set[str] = {
    "SE_MERMAID",
    "SE_GRAPHVIZ",
    "SE_PLANTUML",
    "SE_DIAGRAMS",
    "SE_D2",
    "SE_CYTOSCAPE",
}


def _resolve_prompt(task: str, category: str) -> str | None:
    """根据 (task, category) 解析出 prompt。"""
    task_map = INFER_PROMPT_TEMPLATES.get(task, {})
    if category in task_map:
        return task_map[category]
    return None


def get_prompts_for_api(
    tasks: list[str],
    chart_type: str = "",
) -> dict[str, str]:
    """根据任务列表和图表类型获取对应的 prompt 模板。

    基于 `chart_type` 路由到对应的图表类别（数值类 / 逻辑结构 / 流程图），
    再从 `(task, category)` 双维度映射中取出具体 prompt。

    路由规则（核心）：
    - 数值类 / 逻辑结构图：命中 SE_MD/JSON/CSV/CODE/SVG，**不**命中 SE_MERMAID；
    - 流程图（CATEGORY_FLOWCHART）：**仅**命中 SE_MERMAID，其他 5 个任务
      在 `INFER_PROMPT_TEMPLATES[task]` 中没有 flowchart key，会被自动跳过
      （此处静默跳过，不打印警告，避免流程图每条数据刷 5 行日志）。

    Args:
        tasks: 需要处理的任务列表
        chart_type: 图表类型（中文）；为空时按 DEFAULT_CHART_CATEGORY 处理

    Returns:
        dict[str, str]: {任务名: prompt 模板}
    """
    category = get_chart_category(chart_type)

    prompts: dict[str, str] = {}
    for task in tasks:
        if task not in INFER_PROMPT_TEMPLATES:
            print(f"警告: 未知任务 '{task}'，跳过")
            continue

        prompt = _resolve_prompt(task, category)
        if prompt is None:
            # 流程图 & 非流程图任务（SE_MD/JSON/CSV/CODE/SVG）
            # 或 非流程图 & 流程图类 task（SE_MERMAID/GRAPHVIZ/PLANTUML/DIAGRAMS）：静默跳过，不打印警告
            task_cats = set(INFER_PROMPT_TEMPLATES.get(task, {}).keys())
            is_expected_skip = (category == CATEGORY_FLOWCHART and CATEGORY_FLOWCHART not in task_cats) or (
                task in FLOWCHART_TASKS and category != CATEGORY_FLOWCHART
            )
            if not is_expected_skip:
                print(f"警告: 任务 '{task}' 在类别 '{category}' 下没有可用 prompt，跳过")
            continue

        prompts[task] = prompt

    return prompts


# ============================================================
# 提取函数
# ============================================================


def extract_structual_extraction(answer: str) -> tuple[bool, dict[str, str]]:
    """提取结构化提取结果（通用：直接返回原始答案）。

    Args:
        answer: 模型返回的答案

    Returns:
        tuple[bool, dict]: (是否提取成功, 提取的结果)
    """
    if answer and len(answer.strip()) > 0:
        return True, {"extracted_table": answer.strip()}
    return False, {}


def extract_python_code(answer: str) -> tuple[bool, dict[str, str]]:
    """从模型输出中提取 Python 代码块。

    优先匹配 ```python ... ``` 代码块；若无则匹配任意 ``` ... ``` 代码块；
    若仍无代码块但内容明显是 Python 代码（含 ``import`` 等关键字），则原样返回。

    Args:
        answer: 模型返回的答案

    Returns:
        tuple[bool, dict]: (是否提取成功, {"extracted_table": 代码字符串})
    """
    import re as _re

    if not answer or not answer.strip():
        return False, {}

    text = answer.strip()

    # 1) ```python ... ``` (大小写不敏感)
    m = _re.search(r"```[ \t]*(?:python|py|PYTHON)[ \t]*\n?(.*?)```", text, flags=_re.DOTALL)
    if m:
        code = m.group(1).strip()
        if code:
            return True, {"extracted_table": code}

    # 2) 任意 ```...``` 代码块
    m = _re.search(r"```[a-zA-Z0-9_\-]*[ \t]*\n?(.*?)```", text, flags=_re.DOTALL)
    if m:
        code = m.group(1).strip()
        if code and ("import" in code or "plt." in code or "matplotlib" in code):
            return True, {"extracted_table": code}

    # 3) 看起来就是纯 Python 代码
    if "import" in text and ("plt." in text or "matplotlib" in text or "pyplot" in text):
        return True, {"extracted_table": text}

    # 4) 兜底：只要非空就返回原文
    return True, {"extracted_table": text}


def extract_svg_code(answer: str) -> tuple[bool, dict[str, str]]:
    """从模型输出中提取 SVG 代码。

    优先匹配 ```xml ... ``` / ```svg ... ``` / ```html ... ``` 代码块；
    若无则直接从原文中抽取 ``<svg ...>...</svg>``；都失败则原样返回（只要含 ``<svg``）。

    Args:
        answer: 模型返回的答案

    Returns:
        tuple[bool, dict]: (是否提取成功, {"extracted_table": SVG 字符串})
    """
    import re as _re

    if not answer or not answer.strip():
        return False, {}

    text = answer.strip()

    # 1) ```xml / svg / html``` 代码块
    m = _re.search(r"```[ \t]*(?:xml|svg|html|XML|SVG|HTML)[ \t]*\n?(.*?)```", text, flags=_re.DOTALL)
    if m:
        code = m.group(1).strip()
        if code and "<svg" in code.lower():
            return True, {"extracted_table": code}

    # 2) 任意 ```...``` 代码块（含 <svg）
    for m2 in _re.finditer(r"```[a-zA-Z0-9_\-]*[ \t]*\n?(.*?)```", text, flags=_re.DOTALL):
        code = m2.group(1).strip()
        if code and "<svg" in code.lower():
            return True, {"extracted_table": code}

    # 3) 直接从原文中抽取 <svg ...>...</svg>
    m3 = _re.search(r"<svg\b[^>]*>.*?</svg>", text, flags=_re.DOTALL | _re.IGNORECASE)
    if m3:
        return True, {"extracted_table": m3.group(0).strip()}

    # 4) 看起来是 SVG（但未闭合）：兜底原样返回
    if "<svg" in text.lower():
        return True, {"extracted_table": text}

    return False, {}


def extract_mermaid_code(answer: str) -> tuple[bool, dict[str, str]]:
    """从模型输出中提取 Mermaid 流程图代码。

    优先匹配 ```mermaid ... ``` 代码块；若无则匹配任意 ``` ... ``` 代码块
    （其中内容以 `flowchart` / `graph` 开头）；再无则从原文中抽取首次出现
    `flowchart` / `graph` 关键词之后的内容。

    Args:
        answer: 模型返回的答案

    Returns:
        tuple[bool, dict]: (是否提取成功, {"extracted_table": Mermaid 代码字符串})
    """
    import re as _re

    if not answer or not answer.strip():
        return False, {}

    text = answer.strip()

    # 1) ```mermaid ... ```
    m = _re.search(r"```[ \t]*mermaid[ \t]*\n?(.*?)```", text, flags=_re.DOTALL | _re.IGNORECASE)
    if m:
        code = m.group(1).strip()
        if code:
            return True, {"extracted_table": code}

    # 2) 任意 ```...``` 代码块，且内容以 flowchart/graph 开头
    header_pat = _re.compile(r"^\s*(flowchart|graph)\s+", _re.IGNORECASE | _re.MULTILINE)
    for m2 in _re.finditer(r"```[a-zA-Z0-9_\-]*[ \t]*\n?(.*?)```", text, flags=_re.DOTALL):
        code = m2.group(1).strip()
        if code and header_pat.search(code):
            return True, {"extracted_table": code}

    # 3) 原文中首次出现的 flowchart/graph 行之后的内容
    m3 = header_pat.search(text)
    if m3:
        return True, {"extracted_table": text[m3.start() :].strip()}

    # 4) 兜底：非空就原样返回（至少保留上游日志可排查）
    return True, {"extracted_table": text}


def extract_dot_code(answer: str) -> tuple[bool, dict[str, str]]:
    """从模型输出中提取 Graphviz DOT 代码。

    优先匹配 ```dot ... ``` / ```graphviz ... ``` 代码块；若无则直接从原文中
    抽取 ``digraph ... { ... }`` / ``graph ... { ... }``；都失败则兄底原样返回。

    Args:
        answer: 模型返回的答案

    Returns:
        tuple[bool, dict]: (是否提取成功, {"extracted_table": DOT 代码字符串})
    """
    import re as _re

    if not answer or not answer.strip():
        return False, {}

    text = answer.strip()

    # 1) ```dot / graphviz``` 代码块
    m = _re.search(r"```[ \t]*(?:dot|graphviz|DOT|GRAPHVIZ)[ \t]*\n?(.*?)```", text, flags=_re.DOTALL)
    if m:
        code = m.group(1).strip()
        if code:
            return True, {"extracted_table": code}

    # 2) 任意 ```...``` 代码块，且含 digraph/graph 关键字
    header_pat = _re.compile(r"\b(?:strict\s+)?(?:di)?graph\b", _re.IGNORECASE)
    for m2 in _re.finditer(r"```[a-zA-Z0-9_\-]*[ \t]*\n?(.*?)```", text, flags=_re.DOTALL):
        code = m2.group(1).strip()
        if code and header_pat.search(code) and "{" in code:
            return True, {"extracted_table": code}

    # 3) 从原文中直接匹配 digraph/graph { ... }
    m3 = _re.search(
        r"((?:strict\s+)?(?:di)?graph\b[^{]*\{.*?\})",
        text,
        flags=_re.DOTALL | _re.IGNORECASE,
    )
    if m3:
        return True, {"extracted_table": m3.group(1).strip()}

    # 4) 兄底：非空就原样返回
    return True, {"extracted_table": text}


def extract_plantuml_code(answer: str) -> tuple[bool, dict[str, str]]:
    """从模型输出中提取 PlantUML 代码。

    优先匹配 ```plantuml / puml / uml ... ``` 代码块；若无则直接匹配
    ``@startuml ... @enduml`` 块；都失败则兄底原样返回。
    """
    import re as _re

    if not answer or not answer.strip():
        return False, {}

    text = answer.strip()

    # 1) ```plantuml / puml / uml```
    m = _re.search(r"```[ \t]*(?:plantuml|puml|uml|PLANTUML|PUML|UML)[ \t]*\n?(.*?)```", text, flags=_re.DOTALL)
    if m:
        code = m.group(1).strip()
        if code:
            return True, {"extracted_table": code}

    # 2) 任意 ```...``` 代码块，且含 @startuml
    for m2 in _re.finditer(r"```[a-zA-Z0-9_\-]*[ \t]*\n?(.*?)```", text, flags=_re.DOTALL):
        code = m2.group(1).strip()
        if code and "@startuml" in code.lower():
            return True, {"extracted_table": code}

    # 3) 从原文中匹配 @startuml ... @enduml
    m3 = _re.search(r"@startuml.*?@enduml", text, flags=_re.DOTALL | _re.IGNORECASE)
    if m3:
        return True, {"extracted_table": m3.group(0).strip()}

    # 4) 兄底
    return True, {"extracted_table": text}


def extract_diagrams_code(answer: str) -> tuple[bool, dict[str, str]]:
    """从模型输出中提取 mingrammer diagrams 的 Python DSL 代码。

    复用 extract_python_code 的马运逻辑（优先 ```python``` 块），对结果做一下验证：
    需包含 `from diagrams` / `import diagrams` 或 `with Diagram(` 任一即视为成功。
    """
    ok, extracted = extract_python_code(answer)
    if not ok:
        return False, {}
    code = extracted.get("extracted_table", "") or ""
    if "diagrams" in code or "with Diagram(" in code:
        return True, {"extracted_table": code}
    # 不含明显特征但又不能确认失败：兄底先用原文提交，让下游 AST 解析判定成败
    return True, {"extracted_table": code}


def extract_d2_code(answer: str) -> tuple[bool, dict[str, str]]:
    """从模型输出中提取 D2 流程图代码。

    优先匹配 ```d2 ... ``` 代码块；若无则匹配任意 ``` ... ``` 代码块（其中
    含 D2 特征：至少一行 `id: "xxx"` 且含 `->` 边）；再无则兜底原样返回。
    """
    import re as _re

    if not answer or not answer.strip():
        return False, {}

    text = answer.strip()

    # 1) ```d2``` 代码块（大小写不敏感）
    m = _re.search(r"```[ \t]*(?:d2|D2)[ \t]*\n?(.*?)```", text, flags=_re.DOTALL)
    if m:
        code = m.group(1).strip()
        if code:
            return True, {"extracted_table": code}

    # 2) 任意 ```...``` 代码块且含 `->` 边（排除 mermaid/dot/plantuml）
    for m2 in _re.finditer(r"```[a-zA-Z0-9_\-]*[ \t]*\n?(.*?)```", text, flags=_re.DOTALL):
        code = m2.group(1).strip()
        if not code or "->" not in code:
            continue
        low = code.lower()
        if "digraph" in low or "@startuml" in low or "flowchart" in low or "graph td" in low:
            continue
        # 至少一行 `x: y` 形式
        if _re.search(r"(?m)^\s*\w[\w.\-]*\s*:\s*\S", code):
            return True, {"extracted_table": code}

    # 3) 原文中含 `->` 且含 `id: "..."` 形式
    if "->" in text and _re.search(r"(?m)^\s*\w[\w.\-]*\s*:\s*\S", text):
        return True, {"extracted_table": text}

    # 4) 兜底
    return True, {"extracted_table": text}


def extract_cytoscape_code(answer: str) -> tuple[bool, dict[str, str]]:
    """从模型输出中提取 Cytoscape.js JSON。

    优先匹配 ```json / cytoscape / cyjs``` 代码块；若无则直接尝试从原文中
    抽取首个 `{ ... }` 顶层 JSON 块；都失败则兜底原样返回。
    """
    import re as _re

    if not answer or not answer.strip():
        return False, {}

    text = answer.strip()

    # 1) ```json / cytoscape / cyjs``` 代码块
    m = _re.search(
        r"```[ \t]*(?:json|cytoscape|cyjs|JSON|CYTOSCAPE|CYJS)[ \t]*\n?(.*?)```",
        text,
        flags=_re.DOTALL,
    )
    if m:
        code = m.group(1).strip()
        if code and "elements" in code:
            return True, {"extracted_table": code}
        if code:
            return True, {"extracted_table": code}

    # 2) 任意 ```...``` 代码块且含 "elements"
    for m2 in _re.finditer(r"```[a-zA-Z0-9_\-]*[ \t]*\n?(.*?)```", text, flags=_re.DOTALL):
        code = m2.group(1).strip()
        if code and "elements" in code and "{" in code:
            return True, {"extracted_table": code}

    # 3) 直接从原文中取第一个 `{` 到最后一个 `}`
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        block = text[first : last + 1].strip()
        if "elements" in block:
            return True, {"extracted_table": block}

    # 4) 兜底
    return True, {"extracted_table": text}


INFER_EXTRACT_FUNC: dict[str, Callable[[str], tuple[bool, dict[str, str]]]] = {
    "SE_MD": extract_structual_extraction,
    "SE_JSON": extract_structual_extraction,
    "SE_CSV": extract_structual_extraction,
    "SE_CODE": extract_python_code,
    "SE_SVG": extract_svg_code,
    "SE_MERMAID": extract_mermaid_code,
    "SE_GRAPHVIZ": extract_dot_code,
    "SE_PLANTUML": extract_plantuml_code,
    "SE_DIAGRAMS": extract_diagrams_code,
    "SE_D2": extract_d2_code,
    "SE_CYTOSCAPE": extract_cytoscape_code,
}
