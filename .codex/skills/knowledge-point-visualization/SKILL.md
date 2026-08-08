---
name: knowledge-point-visualization
description: >
  Turn a complete paper Markdown into a knowledge-point graph and HTML visualizations
  with an explicit relation model. The pipeline distinguishes two edge types
  (components = 组成关系, prerequisites = 前置依赖), five node kinds
  (hs / atom / concept / method / contribution), a two-phase extraction
  (explicit paper dependencies, then background expansion with boundaries),
  composite costs (dependency_depth plus normalized learning difficulty H),
  a machine-readable proof domain (proof_type + proof status), a learning-path
  planner, and separate highlight/expand visualization triggers. Use when the user
  asks to generate 知识点 meta.json/all.json/learning-path.json from a paper md,
  build a knowledge-point graph with component/prerequisite edges, set
  学习代价/重要性/掌握状态, or create an HTML knowledge-graph visualization driven
  by JSON.
---

# Knowledge Point Visualization（知识点可视化）

Input: one complete paper in Markdown (e.g. `PGD.md`). Work in the same directory as the
Markdown file. All outputs are written into that directory.

## Output contract

- `meta.json` — knowledge points with explicit relations (Step 1, then updated by Step 2).
- `all.json` — knowledge-point graph (Step 3, generated).
- `learning-cost.json` — graph plus per-node learning cost H, importance I, mastery
  (Step 4, user-edited).
- `user-state.json` — optional per-node mastery / importance overrides (Step 4).
- `learning-path.json` — ordered learning path for a target contribution (Step 6).
- `cost-editor.html`, `knowledge-graph.html` — copied from this skill's `assets/`.
- `derivation.html` — 推导页：公式全覆盖 + 逐步推导 + 通俗解释 + 交互演示（用 `assets/derivation-template.html` 骨架生成，内容规范见 Step 5.5）。
- `interactive-components.js` — 交互演示库，从 `assets/` 拷贝。

## MathJax 离线渲染（禁止重复下载）

LaTeX 公式渲染统一使用本技能内置的 MathJax 单文件包：
`assets/mathjax/tex-svg.js`（MathJax 3.2.2 tex-svg-full，约 2.3 MB，包含全部扩展，离线可用）。

- 任何需要渲染公式的 HTML 输出：先把 `assets/mathjax/tex-svg.js` 复制到输出目录（例如
  `mathjax/tex-svg.js`），再在 HTML 的 `<head>` 中用相对路径
  `<script src="mathjax/tex-svg.js"></script>` 引用，并在其前面配置：
  `window.MathJax = { tex: { inlineMath: [['\\(','\\)']], displayMath: [['\\[','\\]']] }, svg: { fontCache: 'global' } };`
- 不要从 CDN 下载 MathJax，也不要引入其他版本或重复拷贝。
- 若输出目录已经存在相同文件（例如 `PGD_07b060/mathjax/tex-svg.js`），直接复用，无需复制。
- 不渲染公式的页面（如本技能的图谱 HTML）不需要引入该文件。

## Basic concept coverage（基础概念覆盖，禁止省略）

生成任何“讲解/推导型 HTML”之前，先对照 `references/basic-concepts.md` 的强制清单做覆盖检查：
凡在公式或正文中出现过的概念，必须在页面中有高中水平的最小解释，禁止“默认读者会”。

- 把缺失概念集中写进一个“基础工具箱”小节（极限、Σ 性质、均值/方差、exp/log、
  正态密度、独立性、条件概率/全概率、线性方程组、矩阵基本运算、逆矩阵/可逆、
  半正定、迹、偏导/梯度、链式法则、凸、投影/clamp、范数、次微分、贝叶斯、采样等）。
- 每个概念给出直观定义 + 最小公式 + 在论文哪里用到；后续章节只引用不重复。
- 清单中没有但输出中出现的概念，先扩充 `references/basic-concepts.md` 再输出。
- 生成结束后在页面里放一个“覆盖自查”提示，确认没有跳步。
- 解释必须高中生可懂：每个概念给“一句话 + 生活例子 + 最小公式 + 论文位置”，禁止默认读者有大学数学背景。

## Relation model（知识关系建模）

Every node has two relation lists with distinct semantics:

- `components`（组成关系，solid edge）: the knowledge point is made of these parts.
  Example: 岭回归 `components = [最小二乘, 正则化]`.
- `prerequisites`（前置依赖，dashed edge）: these must be learned first, but are not
  parts of the node. Example: 岭回归 `prerequisites = [偏导数, ℓ2 范数]`.

The union graph of `components` ∪ `prerequisites` must be a DAG. Do NOT require each
list to be acyclic separately: the two edge types have different semantics (composition
vs. learning dependency), so a per-list cycle may be meaningful while the union stays
acyclic. Only the union matters for validation, topo order, and cost.

Example: `Transformer.components = [Attention, FFN]` and
`Attention.prerequisites = [Linear Algebra]` are different relation types; the union
`Linear Algebra → Attention → Transformer` must be a DAG, but `components` alone and
`prerequisites` alone are not required to be acyclic.

A node is understood only after its components and prerequisites are understood.

Node kinds:

`kind` expresses the knowledge ROLE, not structural complexity:

- `hs` — 基础知识：high-school knowledge; recursion stops; both lists empty; depth 0.
- `atom` — 不可拆分知识：smallest concept (e.g. 奇异值); `components` empty.
- `concept` — 一般知识概念：composite concepts (e.g. 岭回归, 正则化, KKT, SVD);
  `components` lists constituents. Structural complexity is visible through
  `components`, not through the kind.
- `method` — 论文/领域方法：existing methods the paper relies on but does not propose
  (e.g. 交替最小化, 核范数最小化, SVT, t 检验).
- `contribution` — 论文创新知识：new knowledge or core methods proposed by THIS paper
  (attack model, utility functions, PGA, implicit gradient, SGLD, experiments). Only
  these are the paper's target knowledge points.

Do not use a `complex` or `background` kind: those describe complexity/context, not
role. A node like SVD is `concept` (role) even when it also acts as background for the
paper.

## Step 1 — Phase A: explicit extraction → `meta.json`

Read the paper Markdown completely. Extract knowledge points at smallest-concept
granularity. Record only relations the paper text explicitly requires or that the
mathematics strictly needs — do not expand background knowledge yet. Set
`provenance: "explicit"` on every node, and `"phase": "extraction"` in the root.

For each node record `id`, `name`, `short`, `kind`, `section`, `components`,
`prerequisites`, `proofAnchor`, `description`, `proof_type`
(`definition` / `derivation` / `empirical`, default `derivation`). Merge duplicate
concepts into one node.

## Step 2 — Phase B: background expansion → update `meta.json`

Recursively resolve `components` and `prerequisites` for every node until `kind: "hs"`.
For any missing prerequisite, add the standard background knowledge point yourself,
mark it `provenance: "expanded"`, and continue. Deduplicate. Verify closure: every
non-hs node's components and prerequisites exist in the file. Set `"phase": "expanded"`.

Enforce an EXPANSION BOUNDARY to prevent graph explosion. Declare it in `meta.json`:

```json
"expansion": {
  "max_expand_depth": 5,
  "stop_domains": ["high_school"],
  "scope": ["math_direct", "algorithm_direct", "model_structure_direct"]
}
```

Rules:

- Only expand DIRECT dependencies: direct mathematics, direct algorithm structure,
  direct model-structure dependencies. Do NOT expand deeper foundational chains
  (导数 → 极限 → 集合 → 逻辑 → 公理). If a prerequisite is not already present,
  add it only when it is one step away; never recursively chain background expansion
  below the declared `stop_domains`.
- `max_expand_depth`: if an `expanded` node would end up deeper than this, stop and
  keep it as an explicit reference instead.
- `stop_domains`: expansion stops at these domains (default `high_school`).
- `build_graph.py` reads `expansion` and warns when an expanded node exceeds
  `max_expand_depth`.

This separation resolves the earlier conflict: Step 1 extracts only explicit paper
dependencies; Step 2 performs the recursive background expansion.

## Step 3 — Build the graph → `all.json`

Run the bundled builder (from the skill directory):

```bash
python scripts/build_graph.py meta.json -o all.json [--alpha 0.5 --beta 0.5]
```

The script validates ids, both edge types, and cycles on the UNION graph
(components ∪ prerequisites must be a DAG); computes:

- `topo` — Kahn order over components ∪ prerequisites;
- `dependency_depth` — = 1 + max(parent depth) over the union (hs = 0);
- `importance` — default by kind (contribution 8, method 6, concept 4, atom 3,
  hs 0), overridable per node;
- `effective_cost` — normalized to [0, 1]:
  `E = alpha * (dependency_depth / max_depth) + beta * (H / 10)` when H is set
  (alpha + beta = 1, defaults 0.5/0.5); otherwise `E = dependency_depth / max_depth`;
- `proof_type` — `definition` (introduced by definition/assumption, e.g. QKV,
  Attention block), `derivation` (proved from deps, e.g. complexity analysis),
  `empirical` (established by experiments). Default `derivation`.
- `proof` — machine-readable proof domain: per-node `proof_status`
  (`proven` / `provable` / `unproven`), `proof_deps` = components ∪ prerequisites,
  initial proven set = hs nodes, and the provability rule. `definition` nodes are
  introduced, not proved: they become provable when their referenced deps are proven
  (or immediately when they have no deps).

`edges` carry `type: "component" | "prerequisite"`. Schema: `references/schemas.md`.

## Step 4 — Learning-cost editor

Copy `assets/cost-editor.html` next to `all.json`. Open it, load `all.json`, and set:

- learning cost H (0–10): intrinsic difficulty of the knowledge point;
- importance I (0–10): how central the knowledge point is (contributions should be high);
- mastery: `known` / `weak` / `unknown` per node (user state).

Export `learning-cost.json` (all.json plus H/I/mastery/effective_cost) and optionally
`user-state.json` (mastery + importance overrides only).

## Step 5 — Visualization

Copy `assets/knowledge-graph.html` next to `learning-cost.json`. Open it and load the
JSON. The page:

- renders component edges solid and prerequisite edges dashed;
- computes normalized `effective_cost` in [0, 1];
- animates the proof domain from `proof.states` / `proof_deps` in topological order;
- separates TWO actions (importance ≠ difficulty):
  - **Highlight**（突出显示，金色描边）: `kind == contribution` or
    `importance >= k_imp` — prominent but not expanded;
  - **Expand**（需要学习，红色脉冲 + “需学习”标签）: non-hs node with
    `effective_cost > k` or `mastery ∈ {weak, unknown}`;
- always highlights `contribution` nodes as the paper's targets without forcing their
  prerequisites to expand.

## Step 5.5 — 生成推导页 → `derivation.html`

用 `assets/derivation-template.html` 作为骨架（AI 只填内容，不重写结构/样式）；把 `assets/interactive-components.js`、`assets/mathjax/tex-svg.js` 拷贝到输出目录。按以下规范生成：

1. **公式全覆盖**：论文中每个 `\tag{N}` 编号公式都必须以相同编号出现在推导页，且放在正确的推导上下文；
2. **逐步推导（面向高中生）**：每个微步骤 = 一个等式 + 通俗解释；先用一句话说明“这一步在做什么”，再用比喻/生活例子展开复杂概念；不默认读者有微积分背景；涉及符号必须链到基础工具箱；
3. **跳步禁令**：禁止“同理可得 / 推导略 / 完整推导见扩展版 / 详见附录”等无引用省略；同构步骤必须显式写出式子并引用步骤号；
4. **含义可视化**：每个关键公式（梯度下降、隐函数求导、SGLD 更新等）配 `data-demo` 交互演示；复杂公式步骤卡默认折叠，公式容器 `overflow-x:auto` 防溢出；
5. **覆盖自检**：页面底部“覆盖自检”表逐项打勾，标注“✓ 全部覆盖，无跳步”。

## Step 6 — Learning path → `learning-path.json`

Generate the actionable output: what to learn first.

```bash
python scripts/plan_path.py all.json --target p_sgld -o learning-path.json \
  [--state user-state.json] [--max-depth N]
```

The planner takes the dependency closure of the target contribution node, keeps the
topological order, drops hs nodes and `mastery == known` nodes, and writes an ordered
path (nodes with `dependency_depth`, `effective_cost`, `importance`). This answers
“我应该先学什么？”. Schema: `references/schemas.md`.

## Step 7 — 校验与交付（不过不交付）

运行：

```bash
python <skill>/scripts/verify_outputs.py <paper>.md . --browser-check
```

校验 6 项：产物齐全 / MathJax 生效 / 公式全覆盖 / 无跳步词 / demo 注册完整 / 浏览器真实渲染。失败则修复后重跑，直到输出 `VERIFY PASSED` 才能交付。

## Conventions

- High-school domain: `kind: "hs"`, depth 0, no proof required.
- Basic concept coverage: every symbol used in generated explanations must be covered
  by `references/basic-concepts.md` and explained at high-school level.
- `dependency_depth`: longest dependency chain over components ∪ prerequisites.
- Learning cost H: user-assigned difficulty in [0, 10], independent of depth.
- Importance I: user/kind-assigned centrality in [0, 10].
- Mastery: user state in {known, weak, unknown}; default unknown.
- Effective cost (normalized 0–1): `alpha*depth/maxDepth + beta*H/10`; alpha+beta=1.
- Proof rule: a node is provable iff every id in `proof_deps` is proven; `definition`
  nodes are introduced rather than proved; hs nodes are initially proven. The
  visualization must track this state, not just play topo order.
- 离线约束：页面只允许相对路径本地引用（`mathjax/tex-svg.js`、`interactive-components.js`），禁止 CDN、fetch、外部字体。
- 交付门槛：`scripts/verify_outputs.py --browser-check` 必须输出 `VERIFY PASSED` 才算完成。
