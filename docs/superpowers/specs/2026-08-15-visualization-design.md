# 数据显示功能（HTML 可视化）设计 spec

日期：2026-08-15
状态：待评审（评审通过后进入实施）

## 1. 背景与目标

仓库内模型训练与攻击实验会产出 `outputs/` 下的 `history.json`、`eval_log.csv`、
`config.yaml` 快照与 `{attack}_comparison.json`。当前仅有一个 matplotlib 脚本
（`models/lightgcn/outputs/plot_results.py`），无法做多实验对比、也依赖 Python
环境。目标是提供一个纯前端 HTML 工具：

- 导入本地实验数据（文件夹选择/拖拽），自动识别「模型 / 攻击方法 / 数据集 /
  某次实验（run_tag）」并生成可读名称，例如导入
  `attacks/pgd/outputs/ml100k/lightgcn/2026-08-09-21-54/` 显示为
  `attack-pgd-ml100k-lightgcn-2026年08月09日21时54分`。
- 支持多选实验，用不同颜色叠加折线（训练/验证 loss、recall@K、ndcg@K、
  target 指标等随 epoch 变化）。
- 提供常见图表：折线图、直方图（最终/最佳指标对比）、对比图
  （Clean vs Poisoned 分组柱状）。
- 支持编辑：修改实验标签、颜色、显示/隐藏、删除、图表标题等展示元数据。

## 2. 范围

### 包含

- 单页 HTML 应用，位于 `TPA/visualization/`。
- 本地导入：选中 run_tag 实验目录、或包含多个实验的父目录、或拖拽文件。
- 解析：`history.json`、`eval_log.csv`、`config.yaml`（快照子集）、
  `{attack}_comparison.json`。
- 识别与命名：根据路径段自动识别 kind / method / dataset / model / run_tag，
  run_tag 格式化为中文日期时间。
- 三类图表（折线/直方/对比）+ 多选 + 配色 + 图例。
- 导出当前图表为 PNG 图片（2x 分辨率、白底），文件名自动生成。
- 编辑展示元数据，localStorage 持久化，支持导出/导入快照 JSON。

### 不包含（YAGNI）

- 不改写原始数据文件（编辑只作用于展示元数据与图表配置）。
- 不新增后端服务：纯静态页面，浏览器本地运行（`file://` 或任意静态服务器均可）。
- 不做上传到服务器、多人协作、权限管理。
- 不解析论文对照表（`comparison_table.md` 属模型复现对照，不在本期范围）。

## 3. 数据源与识别规则

### 3.1 支持的实验产物目录

| 类型 | 路径模式 | 产物 |
|------|----------|------|
| 模型训练 | `models/{mf\|lightgcn}/outputs/{run_tag}/` | history.json、eval_log.csv、config.yaml |
| 攻击实验 | `attacks/{tpa\|pgd\|bandwagon\|random}/outputs/{dataset}/{model}/{run_tag}/` | history.json、config.yaml、{attack}_comparison.json、{attack}_comparison.md |

### 3.2 run_tag

- 目录名匹配 `^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}$`（如 `2026-08-09-21-54`）。
- 展示格式：`2026年08月09日21时54分`。

### 3.3 识别算法（parser.js，纯函数）

输入：导入路径的相对段列表（如 `["attacks","pgd","outputs","ml100k",
"lightgcn","2026-08-09-21-54"]`）与 run_tag 所在目录内的文件。

输出 `Experiment`：

```js
{
  id: "attack-pgd-ml100k-lightgcn-2026-08-09-21-54",
  kind: "attack" | "model" | "custom",
  method: "pgd",               // 攻击名或模型名
  dataset: "ml100k",           // 模型类实验从 config.yaml 补全
  model: "lightgcn",
  runTag: "2026-08-09-21-54",
  label: "attack-pgd-ml100k-lightgcn-2026年08月09日21时54分",
  color: "#5470c6",            // 导入时按调色板分配，可编辑
  history: [{ epoch, train_loss?, val_loss?, recall@10?, ndcg@10?, target_ndcg@10?, ... }],
  evalLog: [{ epoch, recall@10, ndcg@10, ... }],   // 模型实验
  best: { metric: { epoch, value, metrics, checkpoint } },  // history.json 的 best
  comparison: { model_utility, target_metrics },   // 攻击实验
  meta: { epochs, metrics: ["recall@10","ndcg@10"] }
}
```

识别要点：

- 出现段 `attacks` → kind=attack，其后第一段为 method，跳过 `outputs` 后依次为
  dataset、model、run_tag。
- 出现段 `models` → kind=model，其后第一段为 model，跳过 `outputs` 后为 run_tag；
  dataset 从同目录 `config.yaml` 的 `dataset:` 读取，缺省 `unknown`。
- 找不到上述模式（如只拖入单个 `history.json`）→ kind=custom，label 默认
  `自定义-<文件名>`，全部字段可编辑。
- 同一目录下文件解析：`history.json` → history；`eval_log.csv` → evalLog；
  `config.yaml` → meta（只需 `dataset` / `model.name` / `attack.name` /
  `training.epochs` / `evaluation.metrics`，用轻量行解析，不引入 YAML 库）；
  `*_comparison.json` → comparison。

### 3.4 指标

- 折线图可选指标：history 与 evalLog 中出现过的数值列，含 `train_loss`、
  `val_loss`、`recall@10`、`ndcg@10`、`target_hr@10`、`target_ndcg@10` 等，
  动态从数据中枚举。
- 直方图（最终/最佳）：优先取 `history.best` 中各指标 value；无 best 时取
  evalLog/history 最后一行的数值。
- 对比图（攻击）：`comparison.model_utility` 的 clean/poisoned 指标，
  与 `comparison.target_metrics` 各目标物品 `hr@k`/`ndcg@k` 的平均。

## 4. 架构与文件结构

纯静态单页应用，无构建步骤；逻辑拆成可单测的纯函数模块与薄 UI 层：

```
TPA/visualization/
  index.html            # 页面骨架：导入区、实验列表、图表容器
  styles.css            # 样式（深色/浅色中性主题）
  js/parser.js          # 纯函数：路径识别、文件解析、标签/时间格式化
  js/transforms.js      # 纯函数：折线/柱状/对比 series 构建、调色板
  js/app.js             # UI 装配：导入、多选、编辑、localStorage、快照导入导出
  lib/echarts.min.js    # 内置 ECharts（离线可用，约 1MB，提交入库）
  tests/test_parser.js      # Node 单测（assert，无第三方依赖）
  tests/test_transforms.js  # Node 单测
  README.md             # 用法 + 手动验证清单
```

职责边界：

- parser.js：输入文件内容/路径段 → 输出 Experiment 数据。不依赖 DOM/ECharts。
- transforms.js：输入 Experiments + 选择项 → ECharts option 的数据部分。不依赖 DOM。
- app.js：事件绑定、导入流程、选中状态、编辑、持久化、渲染调度。薄壳。
- index.html/styles.css：布局与外观。
- 每个 JS 模块采用 UMD 包装：Node 下 `module.exports` 供单测 require，浏览器下挂
  `window.TPAVisualizer.{parser,transforms,app}` 供页面脚本调用。

## 5. 用户流程

1. 打开 `index.html`（file:// 直接打开即可）。
2. 点击「导入数据」选择实验目录（或父目录批量导入，或拖拽）。
3. 左侧列表出现识别出的实验（label + kind/method/dataset/run_tag），默认勾选全部。
4. 选择图表类型与指标，右侧渲染图表；多选实验按颜色区分。
5. 编辑：改标签/颜色、隐藏/删除、改图表标题；状态自动存 localStorage。
6. 「导出快照」下载 JSON；「导入快照」恢复。
7. 点击「导出图片」下载当前图表 PNG（折线/直方/对比均可）。

## 6. 图表设计（ECharts）

1. **折线图**：x=epoch，y=选中指标；每个实验一条线（不同颜色），支持
   legend 切换、缩放（dataZoom）、悬浮提示。
2. **直方图**：x=实验 label（可旋转），每个指标一组柱（grouped bar）；
   指标多选；y=指标值。
3. **对比图**：仅攻击实验可用；x=实验，每组柱含 Clean / Poisoned 两个柱
   （可切换指标：模型效用指标或目标物品平均 HR/NDCG）。
- 每张图表提供「导出图片」：调用 ECharts `getDataURL({type:'png',
  pixelRatio:2, backgroundColor:'#fff'})` 下载 PNG，文件名规则
  `tpa-{chartType}-{metric}-{YYYYMMDD-HHmm}.png`（如
  `tpa-line-ndcg@10-20260815-1430.png`）。

## 7. 编辑功能（默认假设，待确认）

- 可编辑：实验 label、颜色、显示/隐藏、删除；图表标题；当前选择。
- 不编辑原始数据点；如需改数据，通过导出快照 JSON 手动编辑后重新导入。
- localStorage 键：`tpa.visualizer.v1`（容量超限时提示并降级为不持久化）。

## 8. 错误处理

- 文件读取失败/JSON 解析失败：跳过该文件并在列表标记「解析失败：原因」，
  不影响其他实验导入。
- 目录无 run_tag：若目录内含 history.json 或 *_comparison.json，按 custom 导入；
  否则提示「未发现可识别实验」。
- 指标缺失（某实验无某指标）：该实验在该图表中跳过，图例注明。
- 本地存储满：提示后仅本次会话保留。

## 9. 测试策略

- Node 20（本机已装）标准库 `assert`，无第三方依赖：
  - parser：路径识别（attack/model/custom）、run_tag 解析与格式化、各类文件解析、
    边界（缺文件、坏 JSON）。
  - transforms：多实验折线 series、分组柱、对比 series、调色板稳定性。
- 运行命令：`node --test TPA/visualization/tests/`（Node ≥18 支持）。
- 手动验证清单写入 README：真实导入 `attacks/pgd/outputs/ml100k/lightgcn/2026-08-09-21-54`
  与 `models/lightgcn/outputs/2026-08-09-23-01`，检查识别名、三张图、多选配色、编辑与刷新持久化。
- 仓库 Python 单测全量回归保持通过（本功能不改 Python 代码）。

## 10. 待确认问题（默认取第一项，可调整）

1. 编辑范围：A) 只编辑展示元数据（推荐）；B) 还要允许编辑数据点。
2. 图表库：A) 内置 ECharts 离线可用（推荐）；B) 仅 CDN 引用。
3. 存放位置：A) `TPA/visualization/`（推荐）；B) 仓库根 `visualization/`。
