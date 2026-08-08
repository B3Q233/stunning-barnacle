# Knowledge Point Visualization 技能优化设计

- 日期：2026-08-08
- 状态：已批准（brainstorming 完成，等待用户审阅本 spec）
- 涉及技能：`G:\Idea\.codex\skills\knowledge-point-visualization`

## 1. 背景与问题

用户反馈该技能在多次运行中表现不稳定，每次都需要重新说明并迭代才能得到可接受的结果。已确认的问题：

1. 生成的推理流程（`derivation.html`）省略很多步骤，出现“完整推导见扩展版”等跳步表述；
2. 数学公式显示为裸 LaTeX 源码（如 `\frac{...}`），说明 MathJax 未生效；
3. 复杂公式的步骤可视化效果差：长公式横向溢出/截断、步骤卡片臃肿、步骤间关系缺乏引导；
4. 缺少“公式含义”的直观可视化（例如梯度下降不清楚它在做什么），参与感不足。

## 2. 成功标准（用户确认）

1. 论文中每个编号公式都必须出现在推导页中，关键公式有从定义/前置到结论的逐步推导，不允许“完整推导见扩展版”类跳步；
2. 公式渲染必须可靠，不允许出现裸 LaTeX 文本；
3. 复杂公式的步骤可视化可读：不溢出、不臃肿、步骤间先后/因果关系清晰；
4. 关键公式提供可交互的含义演示（可拖动参数、播放动画、悬停解释），增加参与感；
5. 核心目标：每次运行一次达标，无需用户反复说明和迭代。

## 3. 采用方案

**方案 1：指令强化 + 自动化校验/修复 + 标准模板**（用户选定）

- SKILL.md 写入硬性生成规范；
- 新增标准 HTML 模板与交互组件库，AI 按模板填内容而非每次从零手写；
- 新增自动化校验脚本 `verify_outputs.py`，校验不通过则修复重跑，**不过不交付**。

## 4. 交付契约

每次运行（论文 Markdown 所在目录）必须产出以下全部文件，缺一即视为失败：

| 类别 | 文件 |
|---|---|
| 数据 | `meta.json`、`all.json`、`learning-cost.json`、`user-state.json`、`learning-path.json` |
| 图谱可视化 | `knowledge-graph.html`、`cost-editor.html`（从 assets 拷贝） |
| 推导页 | `derivation.html`（新增强制产物，使用标准模板生成） |
| 公式渲染 | `mathjax/tex-svg.js`（从 assets 拷贝） |

## 5. 自动化校验脚本 `scripts/verify_outputs.py`

新增脚本，按以下清单逐项校验；任一项失败 → 修复后重跑，直到全绿才交付：

1. **产物齐全**：第 4 节 9 个文件都存在；JSON 能解析且字段符合 `references/schemas.md`；
2. **MathJax 生效**：`derivation.html` 含 UTF-8 charset、`window.MathJax` 配置、相对路径的 `<script src="mathjax/tex-svg.js">`；全文无 `$...$` / `$$...$$` 错误定界符；`\(...\)` 与 `\[...\]` 配对平衡；
3. **公式全覆盖**：从论文 Markdown 提取全部 `\tag{N}` 编号（如 PGD 为 1–17），推导页必须包含相同编号集合，缺一个即报错（支持 `--formulas` 手工清单覆盖）；
4. **无跳步表述**：禁止“完整推导见扩展版 / 推导略 / 详见附录 / 同理可得”等无具体引用的省略词；若与某步骤同构，必须显式写出对应公式并引用步骤号；
5. **交互组件完整**：推导页每个 `data-demo="<id>"` 都必须在 `interactive-components.js` 注册表中存在；
6. **浏览器级渲染验证（`--browser-check`）**：用无头 Edge/Chrome 打开本地 `derivation.html`，读取页面写入的就绪标记（MathJax 排版完成、demo 初始化成功、无 JS 错误）；通过才代表公式真的渲染，而不是裸 LaTeX。

## 6. 推导页标准模板 `assets/derivation-template.html`

每次生成 `derivation.html` 的固定骨架（AI 只填内容，不重写结构与样式）：

- **头部**：UTF-8 charset、MathJax 配置 + 相对路径脚本、标准样式、交互组件库引用、就绪标记引导脚本；
- **导航栏**：链接到 `knowledge-graph.html`、`cost-editor.html`、公式索引、覆盖自检；
- **基础工具箱区**：沿用 `basic-concepts.md` 覆盖要求，每个概念 = 高中生水平通俗解释 + 生活例子 + 最小公式 + 论文哪里用到；
- **推导主线**：编号步骤卡，固定结构为 目标（goal）→ 前置知识点 chips → 微步骤列表（每步 = 一个等式 + 一行通俗理由）→ 结论 → 含义可视化区（嵌入交互 demo）→ 上一步/下一步导航；长推导默认折叠（展开全部/收起），单步公式容器 `overflow-x:auto` 防溢出；
- **公式索引**：论文全部编号公式目录，点击跳转到对应步骤；
- **覆盖自检表**：公式 → 所在步骤 → 工具箱解释，逐项核对后必须写“✓ 全部覆盖”。

## 7. 内容规范（面向高中生读者）

固化进 SKILL.md，LLM 生成时必须遵守：

1. **公式全覆盖**：论文每个 `\tag{N}` 公式都以相同编号出现在推导页的正确推导上下文中；
2. **逐步推导（面向高中生）**：
   - 每个微步骤 = 一个等式 + 通俗解释：先用一句话说明“这一步在做什么”，再用比喻/日常例子展开复杂概念（如梯度下降 = “蒙眼下山，每一步都往脚下最陡的方向迈一步”）；
   - 不默认读者有微积分/大学数学背景；涉及符号或概念时必须链到基础工具箱中对应的高中生水平定义；
   - 每个关键公式配含义可视化 demo；
   - 禁止“同理可得 / 推导略 / 见扩展版”等无引用省略；同构步骤必须显式写出式子并引用步骤号；
3. **含义可视化**：每个关键公式（梯度下降、隐函数求导、SGLD 更新等）配交互 demo，普通公式可只配图注；
4. **自检留痕**：页面底部覆盖自检表逐项打勾，作为“无跳步”的可见证据。

## 8. 交互组件库 `assets/interactive-components.js`

纯原生 JS、完全离线可用（不引 CDN、不用 fetch），提供通用画布/滑杆/播放暂停/悬停提示等基础工具，并内置可配置演示（AI 通过 `data-demo` 声明 + 参数配置复用）：

- `gradient-descent`：等高线图 + 下降路径动画，滑杆调学习率/起点，播放/暂停/重置；图注用高中生语言解释“为什么往最陡方向走”；
- `projection-box`：投影/clamp，二维示意点如何被截回可行域；
- `implicit-function`：沿曲线 F=0 拖动，显示切线和导数变化；
- `sgld-sampling`：先验 + 似然 → 后验分布动画，滑杆调噪声强度；
- `matrix-decomposition`：SVD/矩阵分解 UΣVᵀ 结构示意。

规则：

- 每个 demo 自带“这是什么 / 怎么玩 / 对应公式”三行说明卡；
- demo 初始化失败只在该卡片标红报错，不影响整页，且就绪标记记录失败项供校验脚本抓取；
- 内置 demo 覆盖不了时，允许 AI 用基础工具写自定义演示，但必须注册到组件表、走同一套就绪标记。

## 9. SKILL.md 流程改动

现有 JSON 管线不动（`build_graph.py`、`plan_path.py`、JSON schema 原样保留，降低风险）：

1. **输出契约**：`derivation.html` 列为强制产物；新增“第 7 步 校验与交付”——运行 `verify_outputs.py --browser-check`，全绿才算完成，失败则修复重跑；
2. **新增第 5.5 步 生成推导页**：用 `derivation-template.html` 骨架 + `interactive-components.js` 组件库 + 基础工具箱规则生成 `derivation.html`；
3. **强化基础概念覆盖章节**：每个条目 = 高中生水平通俗解释 + 生活例子 + 最小公式 + 论文位置；禁止默认读者有大学数学背景；
4. **明确离线约束**：全部页面只允许本地相对引用（MathJax 用内置 `tex-svg.js`），禁止 CDN/fetch，保证双击本地打开即可用；
5. **同步更新 `agents/openai.yaml`** 默认提示词：一开始就把“公式全覆盖 + 通俗解释 + 交互演示 + 校验通过”纳入目标。

## 10. 测试验收

用现有 `G:\Idea\MinerU-Skill\PGD_07b060\PGD.md` 作为验收样例：

1. 完整跑一遍新流程，`verify_outputs.py --browser-check` 必须全绿；
2. 无头 Edge 打开 `derivation.html`：断言 MathJax 真的排版出 SVG（不再是裸 LaTeX）、`__KPV_READY__` 标记为 true、所有 demo 初始化成功；
3. 断言推导页包含论文全部 17 个公式编号、无跳步词；
4. 抽查梯度下降等关键公式的交互 demo 能正常播放/拖动。

## 11. 范围外（明确不做）

- 不改 `build_graph.py`、`plan_path.py` 及 JSON schema；
- 不改 `knowledge-graph.html`、`cost-editor.html`（保持从 assets 拷贝使用）；
- 不引入任何外部运行时依赖（CDN、npm 包等）。
