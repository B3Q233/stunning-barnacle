# 攻击实验可视化 demo 设计（折线图 + 直方图对比）

日期：2026-08-15
状态：待评审（用户已口头确认方案 A 与指标显隐）

## 1. 背景与目标

仓库实验会产出 `history.json`（每轮训练/评估指标）与
`{attack}_comparison.json`（Clean vs Poisoned 最终对比）。旧版可视化（HTML +
ECharts，已提交于 git HEAD，工作区已删除）bug 过多，决定从零开始，按敏捷方式
交付一个最小 demo 后逐步扩展。

本 demo 的目标：纯前端单页，输入一次攻击实验的两个文件，输出两张图——折线图
（epoch × 每轮指标）与直方图（Clean vs Poisoned 对比），并支持指标勾选显隐。
核心是先把每轮记录提取成标准 JSON `{epoch: {指标}}`，为后续多实验对比打基础。

## 2. 范围

### 包含

- 单页 HTML（`file://` 打开即用，无后端、无构建步骤）。
- 输入入口：`history.json` 与 `attack_comparison.json` 两个文件选择（支持拖拽）。
- 数据处理：
  - 提取每轮数值标量指标，输出标准 JSON `{"1": {指标...}, "2": {...}, ...}`。
  - 动态枚举指标列，未来新增指标无需改代码。
  - 解析 `attack_comparison.json` 的 `model_utility` 与 `target_metrics`。
- 图表（ECharts）：
  - 折线图：x=epoch，每个指标一条线。
  - 直方图：Clean vs Poisoned 分组柱（模型效用指标 + 目标物品命中指标）。
  - 指标勾选显隐，即时重绘。
- 导出：下载标准 JSON 文件。
- 页面摘要：识别到的 epoch 数、指标列表。
- 错误提示：文件缺失/坏 JSON 不崩溃。

### 不包含（后续增量）

- 多实验同时对比、目录批量导入、run_tag 自动识别命名。
- `attack_comparison.json` 结果作为折线图参照线（后续增量再做）。
- `eval_log.csv` 解析、`targets` 每轮嵌套明细展示。
- 图表 PNG 导出、标签/颜色编辑、localStorage 持久化。
- 后端服务、多人协作。

## 3. 输入与数据模型

### 3.1 history.json（两种顶层形态都兼容）

- 形态 A（旧）：顶层为数组 `[{epoch, train_loss, val_loss, ...}, ...]`。
- 形态 B（新）：顶层为对象 `{"history": [...], "best": {...}}`。

统一提取规则：

- 取 `history` 数组（形态 A 直接取数组本身）。
- 每条记录取 `epoch` 为 key（JSON 序列化后为字符串），值为该轮所有**数值标量**
  字段（`train_loss` / `val_loss` / `recall@10` / `ndcg@10` / `target_hr@10` /
  `target_ndcg@10` 等）。
- 跳过非数值字段与嵌套对象（如 `targets` 明细）。
- 输出示例：

```json
{
  "1": {"train_loss": 0.0229, "val_loss": 0.0643, "recall@10": 0.1725, "ndcg@10": 0.2378},
  "2": {"train_loss": 0.0229, "val_loss": 0.0529, "recall@10": 0.1721, "ndcg@10": 0.2377}
}
```

### 3.2 attack_comparison.json

解析为：

```js
{
  modelUtility: { clean: { "recall@10": number, "ndcg@10": number }, poisoned: {...} },
  targetMetrics: { clean: { "251": { hr@k, ndcg@k, ... } }, poisoned: {...} }
}
```

直方图展示项：模型效用 `recall@10` / `ndcg@10`，目标物品命中 `hr@k` / `ndcg@k`
（多目标时取各目标平均值）。

### 3.3 指标枚举

从提取后的每轮记录动态收集数值字段（排除 `epoch`），作为折线图指标复选框列表；
直方图指标固定为上述展示项，同样支持勾选。

## 4. 页面布局与交互

```
┌────────────────────────────────────────────┐
│ 标题：TPA 攻击实验可视化                      │
│ [选择 history.json] [选择 attack_comparison.json] │
│ 摘要：epoch 数 / 指标列表 / 解析错误信息        │
│ [导出标准 JSON]                              │
├────────────────────────────────────────────┤
│ 指标显示控制（checkbox，默认全选）             │
├────────────────────────────────────────────┤
│ 折线图（epoch × 指标）                        │
├────────────────────────────────────────────┤
│ 直方图（Clean vs Poisoned）                   │
└────────────────────────────────────────────┘
```

交互规则：

- 勾选/取消指标复选框 → 立即重绘对应图。
- 未提供 `attack_comparison.json` → 直方图区域显示提示，不影响折线图。
- 文件解析失败 → 摘要区显示错误原因，不清空已成功的数据。
- ECharts 折线图启用 tooltip、legend、dataZoom（缩放）。

## 5. 架构与文件结构

纯静态单页，逻辑拆成可单测的纯函数模块与薄 UI 层：

```
TPA/visualization/
  index.html            # 页面骨架
  styles.css            # 样式
  js/parser.js          # 纯函数：history 提取、comparison 解析、指标枚举（UMD）
  js/transforms.js      # 纯函数：折线/直方 series 构建、显隐过滤（UMD）
  js/main.js            # UI 装配：文件读取、checkbox、ECharts 渲染
  lib/echarts.min.js    # 本地 vendored（约 1MB，离线可用）
  tests/test_parser.js      # Node 单测（assert）
  tests/test_transforms.js  # Node 单测
  README.md             # 用法 + 手动验证清单
```

职责边界：

- `parser.js`：输入文件文本 → 输出标准数据对象；不依赖 DOM/ECharts。
- `transforms.js`：输入数据对象 + 选中指标 → ECharts option 的数据部分。
- `main.js`：FileReader 读取、checkbox 状态、`echarts.init` 与 `setOption`。
- 模块用 UMD 包装：Node 下 `module.exports` 供单测，浏览器下挂
  `window.TPAVisualizer`。

## 6. 错误处理

- 文件读取失败 / JSON 解析失败：摘要区提示文件名与原因，跳过该文件。
- 缺字段（某轮无某指标）：折线该点置 `null`（断线），不报错。
- 两个文件都没选：提示先选择文件。
- 直方图数据缺失：显示"缺少对比数据"占位提示。

## 7. 测试策略

- Node（本机已装，标准库 `assert`，无第三方依赖）：
  - `parser.js`：两种 history 顶层形态、数值字段动态提取、`epoch` key 映射、
    坏 JSON 抛错、comparison 解析。
  - `transforms.js`：折线 series（含指标缺失断点）、显隐过滤、直方 series
    （clean/poisoned 分组、多目标均值）。
- 运行命令：`node --test TPA/visualization/tests/`。
- 仓库 Python unittest 全量保持通过（本功能不改 Python 代码）。

## 8. 实施顺序与 Git 约定

1. 先提交旧版删除：`git add -- TPA/visualization` 后提交
   `chore(visualization): 移除旧版可视化实现（工作区已删除）`（避免新旧文件混淆）。
2. 下载 ECharts 到 `lib/echarts.min.js`（需联网，实施时请求批准）。
3. 按 TDD 依次实现：parser.js（含测试）→ transforms.js（含测试）→
   index.html/styles.css/main.js → README.md。
4. 提交信息用 Conventional Commits，scope 为 `visualization`，中文描述；
   只 `git add` 明确路径。

## 9. 手动验证清单

1. `file://` 打开 `TPA/visualization/index.html`。
2. 选择
   `TPA/attacks/random/outputs/ml100k/lightgcn/2026-08-15-07-23/history.json`
   与同目录 `attack_comparison.json`。
3. 摘要显示 epoch 数（30）与指标列表（train_loss、val_loss、recall@10、
   ndcg@10、target_hr@10、target_ndcg@10）。
4. 折线图显示全部曲线；取消勾选 `train_loss` 后曲线消失，再勾选恢复。
5. 直方图显示 Clean/Poisoned 分组柱（模型效用与目标命中）。
6. 导出标准 JSON，检查为 `{"1": {...}, ..., "30": {...}}` 且无 `targets` 嵌套。
7. 选择坏 JSON 文件，页面提示错误且不崩溃。

## 12. 设计修订 v2（2026-08-15 用户确认，多实验选项卡）

### 12.1 修复

- 直方图 x 轴标签取消倾斜（去掉 `axisLabel.rotate: 20`），文字水平显示。

### 12.2 左侧导航（选项卡）

- 竖向列表，每项为横向选项卡条目：`[勾选框] 名称` + 操作按钮
  （改名 / 重置 history / 重置 comparison / 删除）。
- 「新建选项卡」按钮：弹窗输入名称（默认「实验 N」，可随时改名）。
- 导入方式（两种都支持，导入只能落在选项卡内）：
  - 选择整个实验目录（`webkitdirectory`）：从相对路径自动解析
    「攻击方法-实验时间」，如 `attacks/random/outputs/ml100k/lightgcn/
    2026-08-15-07-23/` → `random-2026-08-15-07-23`，同时读入目录下
    `history.json` 与 `*_comparison.json`；
  - 在选项卡内多选 `history.json` + `attack_comparison.json` 两个文件导入。
- 导入规则：每次导入都落到新建/空选项卡；当前选项卡已有数据时提示先新建。

### 12.3 顶部卡牌（指标控制）

- 只有左侧**勾选**的选项卡在卡牌中渲染，从上到下依次为各实验。
- 每个实验一段：名称 + 该实验全部指标 checkbox（默认全选）+ 每指标颜色选择器。
- 修改勾选/颜色即时重绘对应图。

### 12.4 图表

- 折线图：同一指标按实验拆线（`recall@10-实验A`、`recall@10-实验B`），
  颜色使用卡牌自定义颜色。
- 直方图：x 轴 = 实验，每个实验一组 Clean / Poisoned 柱；对比指标
  （模型效用 / 目标命中）在卡牌中勾选显隐，标签水平显示；柱色按
  实验的指标自定义色逐点着色。
- 未勾选含数据选项卡时显示占位提示。

### 12.5 结构变化

- `parser.js`：新增 `parseDirectoryPath(relativePath)` 与 `buildAutoName(info)`。
- `transforms.js`：新增 `buildMultiLineSeries(experiments)` 与
  `buildMultiBarSeries(experiments)`（保留 v1 单实验函数）。
- `main.js`：选项卡状态管理（新建/改名/重置/删除/勾选/激活）、目录导入、
  卡牌渲染；`index.html` / `styles.css` 增加侧栏与卡牌布局。
- Node 单测补充：目录路径解析、自动命名、多实验折线/直方 series、颜色覆盖。

## 13. 设计修订 v3（2026-08-15 用户确认，卡内导入 + 顶刊配色 + 组件化）

### 13.1 修复：ECharts 标题/图例重叠

- 两张图统一布局：`title.top=10`、`legend.type='scroll', top=44, left='center'`、
  `grid.top=96`，标题、图例、绘图区三层分离，系列多时不再互相挤压。

### 13.2 导入归属实验卡（移除全局导入）

- 删除顶部全局「导入实验目录 / 导入文件 / 拖拽」入口。
- 主区域为实验卡列表，每张卡自带：
  - 卡头：`[勾选框] 名称` + 数据状态（history ✓ / comparison ✓）+
    按钮（改名 / 重置 history / 重置 comparison / 删除）；
  - 卡体：无数据时显示导入区——「导入实验路径」（选目录自动填名并读入两个文件）
    ＋「添加数据」（卡内选择 `history.json` / `*_comparison.json`）；
  - 有数据后卡体显示该实验指标行（勾选 + 颜色）。
- 「＋ 添加实验卡」：打开 Modal，输入名称，或直接选择实验路径自动命名。

### 13.3 顶刊配色

- 系列色板替换为 Nature 色盲友好 10 色：
  `#3B4992 #EE0000 #008B45 #631879 #008280 #BB0021 #5F559B #A20056 #808180 #1B1919`。
- 布局改为期刊风格：白底、细灰边框、弱化头部、克制字号与留白。

### 13.4 组件化弹窗

- 新建/改名使用同一 Modal 组件（标题 + 输入框 + 确定/取消）；
- 删除使用确认 Modal；不再使用 `prompt()` / `confirm()`。

### 13.5 结构变化

- `transforms.js`：仅替换 `PALETTE` 为 Nature 色板。
- `index.html` / `styles.css` / `main.js`：实验卡列表布局、卡内导入、
  Modal 组件、图表布局修复、期刊配色。
- `parser.js` 与既有 Node 单测逻辑不变。
