---
name: knowledge-point-visualization
description: >
  Turn a complete paper Markdown into a Paper Tutor (论文深度教学器): a four-layer
  derivation.html (背景演化 → 直觉/例子 → 100% 逐步推导 → 实验验证) with formula
  source levels, why-cards, hand-computable numeric examples and four restricted
  visualization types, plus the knowledge-point graph (meta.json / all.json /
  learning-cost.json / learning-path.json / knowledge-graph.html). Use when the
  user asks to 可视化论文、论文推导、详细推导、知识点图谱、学习路径、论文教学，或给出论文
  Markdown 要求生成教学型 HTML。
---

# Knowledge Point Visualization（知识点可视化 / Paper Tutor）

## 定位（Paper Tutor 优先）

- `derivation.html` 是默认主交付：论文深度教学器（教材），不是论文笔记。
- `knowledge-graph.html` 是知识导航/高级视图，仅作为附录链接，不进入默认教学流程主页。
- `learning-path.json` 提供个性化学习顺序；`cost-editor.html` 是学习代价编辑器。
- 四层阅读体验：背景（我为什么学）→ 直觉（公式想解决什么）→ 推导（每一步为什么成立）→ 验证（实验如何对应理论）。

## Teaching Policy（最高优先级，约束 Step 1–7）

Paper Tutor 不是论文翻译，而是教材。任何知识点必须遵循：

**提出问题 → 建立直觉 → 给最小例子 → 数学推导 → 回到论文**

禁止：
- 先堆公式后解释；
- 先讲完整个例子再推导；
- 为了完整而重复同一内容（对称推导用镜像映射）；
- 把背景数学集中成一章讲完（工具箱是字典，不是先修课程）。

本策略优先级高于 Step 1–7 的所有输出规则；与后续步骤冲突时，以本策略为准。

## Output contract

- `meta.json`（Step 1/2）、`all.json`（Step 3）、`learning-cost.json`（Step 4）、
  `learning-path.json`（Step 6）、`knowledge-graph.html` / `cost-editor.html`
  （复制自 `assets/`）、`derivation.html`（Step 5.5）、`mathjax/tex-svg.js`、
  `interactive-components.js`。
- `derivation.html` 必须实现 Step 5.5 的 Paper Tutor 固定目录与教学规范；知识图谱
  页面定位为附录/高级视图。

## MathJax 离线渲染（禁止重复下载）

- LaTeX 渲染统一使用技能内置 `assets/mathjax/tex-svg.js`（MathJax 3.2.2 tex-svg-full，
  约 2.3 MB，离线可用）。
- 任何需要渲染公式的 HTML 输出：先把 `assets/mathjax/tex-svg.js` 复制到输出目录
  （如 `mathjax/tex-svg.js`），再在 `<head>` 用相对路径
  `<script src="mathjax/tex-svg.js"></script>` 引用，并在其前面配置
  `window.MathJax = { tex: { inlineMath: [['\\(','\\)']], displayMath: [['\\[','\\]']] }, svg: { fontCache: 'global' } };`
- 禁止 CDN、禁止引入其他版本或重复复制；输出目录已有同名文件则直接复用。
- 不渲染公式的页面（如图谱 HTML）无需引入。

## Basic concept coverage（Paper Tutor 升级）

1. 凡在公式或正文中出现过的概念，必须在 Chapter 1 基础工具箱有高中水平解释
   （一句话 + 生活例子 + 最小公式 + 论文位置），禁止“默认读者会”。
2. 背景公式也必须推导，不因“基础公式”跳过。概念推导示例：点积
   \(\hat p_{ui} = p_u^\top q_i\) 必须拆成四步——定义用户向量 → 定义物品向量 →
   为什么用点积 → 推广成矩阵形式。完整范例见 `references/teaching-examples.md`。
3. 每个符号必须链到 Chapter 2 符号总表。

## Relation model（知识关系建模，不变）

- 每个节点两个关系列表：`components`（组成关系，实线边）与 `prerequisites`
  （前置依赖，虚线边）；union 图必须为 DAG，各自列表不要求单独无环。
- 五种 kind（知识角色，不是复杂度）：`hs`（高中基础，双列表为空，depth 0）、
  `atom`（不可拆分，components 空）、`concept`（复合概念）、`method`（论文依赖的
  既有方法）、`contribution`（本文提出的创新知识）。禁止 `complex`/`background` 这类
  “复杂度/语境”kind。

## Step 1 — Phase A：显式抽取 → meta.json（不变）

读完整篇论文，按最小概念粒度抽取知识点；只记录论文明确要求或数学严格需要的依赖，
暂不扩充背景。所有节点 `provenance: "explicit"`，根节点 `"phase": "extraction"`。
字段：`id/name/short/kind/section/components/prerequisites/proofAnchor/description/
proof_type`（`definition` / `derivation` / `empirical`，默认 `derivation`）。重复概念合并。

## Step 2 — Phase B：背景扩充 → meta.json（不变）

递归补齐 `components`/`prerequisites` 直到 `kind: "hs"`；缺失的前置知识点自行添加并标
`provenance: "expanded"`；去重；校验闭包（非 hs 节点的依赖必须存在）。根节点
`"phase": "expanded"`，并声明扩张边界：

```json
"expansion": {
  "max_expand_depth": 5,
  "stop_domains": ["high_school"],
  "scope": ["math_direct", "algorithm_direct", "model_structure_direct"]
}
```

只扩一层直接依赖，不递归深挖基础链条（导数 → 极限 → 集合 → 逻辑 → 公理）；
超出 `max_expand_depth` 的 expanded 节点停止，保留为显式引用。

## Step 3 — 构建图谱 → all.json

```bash
python scripts/build_graph.py meta.json -o all.json [--alpha 0.5 --beta 0.5]
```

脚本校验 id、双关系边、union 环，并计算 `topo` / `dependency_depth` / `importance` /
`effective_cost` / `proof_type` / `proof`（`proof_status`、`proof_deps`、初始 proven 集
= hs 节点）。schema 见 `references/schemas.md`。

## Step 4 — 学习代价编辑器（不变）

复制 `assets/cost-editor.html` 到 `all.json` 旁；打开后设置 H（0–10）、重要性 I（0–10）、
mastery（known/weak/unknown），导出 `learning-cost.json`（可选 `user-state.json`）。

## Step 5 — 知识图谱可视化（定位：附录/高级视图）

复制 `assets/knowledge-graph.html` 到 `learning-cost.json` 旁；页面渲染 component 实线、
prerequisite 虚线、proof 域动画、Highlight（contribution / importance≥阈值）与
Expand（effective_cost>阈值 / mastery∈{weak,unknown}）。该页只作为附录/高级视图，
不进入默认教学流程主页。

## Step 5.5 — 论文深度教学器（Paper Tutor）→ derivation.html

以 `assets/derivation-template.html` 为骨架（AI 只填内容，不重写结构/样式）；复制
`assets/interactive-components.js` 与 `assets/mathjax/tex-svg.js` 到输出目录。固定目录：

- Chapter 0 研究背景（`#evolution`）
- Chapter 1 基础工具箱（`#toolbox`）
- Chapter 2 符号总表（`#symbols`）
- Chapter 3 核心公式（`#core`）
- Chapter 4 实验验证（`#experiments`）
- 附录（`#appendix`）
- 公式索引（`#formulas`）+ 覆盖自检（`#check`）

### 0. 研究背景（四层因果链 + 演化时间轴）

- 固定四层：① 以前的方法是什么（按时间顺序：UserCF → ItemCF → MF → WMF → LightGCN…）
  ② 为什么它们不够（每个方法一个核心缺陷 + 一个具体例子）③ 本文为什么这样设计
  （每个创新回答“解决上一代哪个问题”，形成因果链）④ 最终效果如何（创新 → 指标提升 →
  代价，说明为什么值得采用）。
- 演化时间轴：3–6 个阶段、禁止超过一页；表格列：阶段 / 内容，内容行依次为
  ① 方法（以前怎么做）② 缺陷（为什么不够）③ 创新（本文如何解决）④ 效果（提升了什么、
  代价是什么）。

### 1. 基础工具箱

所有概念高中水平解释（一句话 + 生活例子 + 最小公式 + 论文位置）；背景公式做概念推导；
清单缺失的概念先扩充 `references/basic-concepts.md` 再输出。工具箱是“字典”，不是
先修课程：只保留读者打开第一页就需要的基础概念（建议 ≤8 行，例如点积/矩阵/对角矩阵/
正则化/范数）；Gram 矩阵、半正定、Hessian、链式法则、加权最小二乘等复杂概念一律在
首次出现处就地解释一次，不进 Chapter 1，避免读者在学论文前先被迫学四页数学。

Chapter 1 只允许放“打开第一页就需要”的概念（默认白名单：点积、矩阵与转置、对角矩阵、
范数、正则化）；其余概念必须在第一次出现时插入。硬规则：若某概念首次出现在公式推导中，
必须先解释、再使用，不得要求读者返回 Chapter 1——“边学边补”。

### 2. 符号总表

每个符号：符号 / 形状 / 意义 / 为什么需要（至少 5 行）。每个公式前必须出现或链接到该表；
对易混符号（如 X 与 x_u、Y 与 y_i、C^u 与 C^i）要说明“是输入还是输出、来自哪一步”。

### 3. 核心公式（每个公式的固定流程）

### Formula Teaching Flow（每个公式的固定阅读顺序）

1. 提出问题（为什么需要这个公式）
2. 微型例子（2–5 行、可手算、只覆盖当前步）
3. 局部符号（仅当前公式涉及）
4. Formula Source Level（这是定义/优化/变形……）
5. Why Card（按来源自动裁剪）
6. 一步一解释推导
7. 回到论文（它影响了哪一个实验）

动机永远在数学前面；同一组数字贯穿本章。

**公式来源等级（Formula Source Level，每个公式必须标注，先说明类别再推导）**

| 类型 | 必须做什么 |
|---|---|
| 定义 Definition | 解释来源，不推导 |
| 假设 Assumption | 说明为什么这样假设 |
| 变形 Transformation | 每一步代数变换，不允许跳步 |
| 优化 Optimization | 说明为什么这样求导/最优化（梯度置零、闭式解、凸性） |
| 结论 Result | 说明由哪些公式得到 |

格式示例：`公式 (4) [Optimization]` —— 先说明“这是优化问题的解，不是定义”，再推导。

**Why Card（自适应：问题集合按公式来源自动选择，不固定四问；生活例子必须出现）**

| 来源等级 | Why Card 内容 |
|---|---|
| Definition | 为什么这样定义？它描述什么对象？ |
| Assumption | 为什么这样假设？放弃了什么？ |
| Transformation | 为什么能这样变形？用到了哪个数学性质？ |
| Optimization | 为什么这样优化？为什么不是 SGD？为什么梯度置零成立？代价是什么？ |
| Result | 它说明了什么？它由哪些公式得到？ |

不同来源的公式不得出现一模一样的问题集合。Why Card 与公式绑定：方法总览（如 ALS）
放完整卡片；推导子步骤（2.1–2.7 等）只解释“这一步为什么成立”，不再重复总览动机。

**推导步骤**

- 一个等式 + 一句解释 + 一个理由；禁止跳步（同理可得 / 推导略 / 详见附录 /
  完整推导见扩展版 / 推导从略）。
- 连续显示公式之间必须有解释文本（结构检测：A=B 后直接 A=C 判违规）。
- 背景公式同样推导，不因“基础”跳过。
- 对称/镜像推导用映射表：用户侧 100% 推导；物品侧等镜像结构给出映射表
  （Y→X、x_u→y_i、Cᵘ→Cⁱ、p(u)→p(i)）并论证“唯一变化是变量交换，正规方程保持同构”，
  不逐行重复全文。

**伴随式数值例子（Mandatory，认知脚手架）**

- 每个关键公式首次出现时，紧跟一个微型数值例子（≤2×2、≤3 维、小整数），只覆盖
  当前推导步骤，不提前引入后续变量；禁止“先讲完整个例子再推导”。
- 阅读顺序：动机 → 微型例子 → 推导一步 → 回到论文；例子不是独立模块，而是嵌入
  推导过程，出现在读者第一次遇到这个公式的时候。
- 同一章节优先复用同一组数字，让例子随推导演化（例：定义置信度用一个样本 →
  构造 C 矩阵用同一个用户 → 写损失沿用同一组物品 → 求导保留同一例子 → 闭式解最后
  才算逆矩阵）。
- 例子状态连续：同章例子的变量名、数值与语义默认继承上一节，除非明确声明“新的例子”；
  不得在下一节突然换一组数字，让读者重建上下文。
- 禁止集中设置“统一数值例子”章节。
- 对“为什么要这样建模”的核心转换（如观测 r → 偏好 p + 置信度 c），必须先给原始数据表，
  再逐步构造 p 与 c，说明职责分离：是否发生行为 vs 次数决定置信度；禁止直接丢出
  “p=1、c=4”而不解释 p 为什么不是 r。

**Example Continuity（强制）**

一个章节只能有一组主例子：
- 第一节建立数据；后续只允许增加变量，不允许重置数字。
- 若必须换例子，必须声明“新的例子”。
- 每一步例子只解释当前新增概念。例如 WMF：3.1 r → 3.2 p → 3.3 c → 3.4 C → 3.5 ALS，
  而不是每节重新生成用户。

### Matrix Example Representation（矩阵例子规范）

凡公式中出现矩阵、向量、向量乘法、矩阵求逆，禁止只给标量展开计算；必须同时给出：
1. 矩阵/向量的具体形式（必须用 LaTeX 渲染，如
   \(Y=\begin{bmatrix}1\\0.5\end{bmatrix}\)、\(C^u=\begin{bmatrix}4&0\\0&1\end{bmatrix}\)、
   \(p(u)=\begin{bmatrix}1\\0\end{bmatrix}\)）；
2. 维度标注（如 \(Y\in\mathbb{R}^{2\times1}\)）；
3. 矩阵乘法过程（至少展示一次，如 \(C^uY\) → \(Y^\top(C^uY)\) 分步）；
4. 最终数值结果。

统一格式（全部用 LaTeX）：矩阵 \(A=\begin{bmatrix}a_{11}&a_{12}\\a_{21}&a_{22}\end{bmatrix}\)；
向量 \(x=\begin{bmatrix}x_1\\x_2\end{bmatrix}\)；计算
\(Ax=\begin{bmatrix}a_{11}x_1+a_{12}x_2\\a_{21}x_1+a_{22}x_2\end{bmatrix}\)；
结果 \(=\begin{bmatrix}b_1\\b_2\end{bmatrix}\)。

举例中的公式同样用 LaTeX（inline \(...\)）渲染，禁止用纯文本方括号 [1; 0.5] 或文字
描述代替实际矩阵。

禁止：
- 只写 A=4.25、b=4 等中间变量；
- 只展开标量计算而隐藏矩阵来源；
- 用文字描述矩阵含义代替实际矩阵。

术语：一维闭式解中 (YᵀCᵘY+λI)⁻¹ 的标量化结果称为“正规方程左侧矩阵项的标量化结果 /
闭式解系数项”，不要叫“分母”，避免读者误以为 ALS 只是简单除法。

### 4. 实验验证（Theory Link：实验是论文证据）

每个实验必须引用对应公式，表格含“对应公式 / 验证假设 / 实验结果 / 结论 / 代价”字段，
格式示例：
- 对应公式：Eq.(3)
- 验证假设：置信度加权有效
- 实验结果：8.56% vs 10.49%
- 结论：支持 / 不支持

例如式 (3) 的偏好+置信度 → 8.56% vs 10.49% → 验证假设：点击次数不再直接作为目标、
高频行为只增加置信度；Gram 技巧（式 (4)）→ O(f²𝒩+f³m) → 验证假设：精确代数分解。
实验不是证明模型“好”，而是在验证前面哪条数学设计。

### 可视化（按需触发，不是章节配额）

图不是“每章配一张”，而是公式需要时才画；只有满足任一条件才画图：
- 空间关系（点积、梯度方向）→ 二维示意图；
- 结构关系（模型整体流程）→ 流程图；
- 前后变化（旧 vs 新、攻击前后）→ 前后对比 / 函数曲线；
- 矩阵模式（相似度、注意力、邻接）→ 热力图。

一张图只表达一个概念、只对应一个公式；图下必须写“这张图帮助理解公式 X”；同一章节
仍最多 1 张图；禁止知识图谱式高密度图用于教学页。没有满足条件的公式就不画图。

### 覆盖自检（#check）

论文每个编号公式都有对应章节；覆盖表逐项列出（公式 / 状态 / 所在章节），覆盖率必须
100%，输出“✅ 全部覆盖，无跳步”。

## Step 6 — 学习路径 → learning-path.json

```bash
python scripts/plan_path.py all.json --target <contribution_id> -o learning-path.json \
  [--state user-state.json] [--max-depth N]
```

取目标贡献节点的依赖闭包，按拓扑序输出，去掉 hs 与 mastery=known 节点。

## Step 7 — 校验与交付（不过不交付）

```bash
python scripts/verify_outputs.py <paper>.md . --browser-check
```

校验项：文件齐全 / MathJax 生效 / 公式全覆盖 / 无跳步 / demo 注册 / Paper Tutor 结构
（章节完整性、自适应 Why Card、数值例子、可视化每节≤1 且有公式说明、实验 Theory Link、
覆盖率表、连续等式跳步检测）/ 浏览器真实渲染。失败则修复后重跑，直到输出
`VERIFY PASSED` 才交付。

## Conventions

- 高中域：`kind: "hs"`，depth 0，不要求证明。
- Teaching Policy、Formula Teaching Flow、自适应 Why Card、Example Continuity、
  工具箱字典化（≤8 行）、对称推导用镜像映射、实验 Theory Link、按需可视化（每节≤1）
  是强制规范，不是建议。
- 默认入口永远是 derivation.html；知识图谱只出现在附录链接。
- 教学范例（点积四步、Softmax 例子→数学→回例子、Why Card、损失三问）见
  `references/teaching-examples.md`。
