# TPA 实验数据可视化

纯前端 HTML 工具：导入本地实验数据，自动识别「模型 / 攻击方法 / 数据集 / 某次
实验」，用折线图、直方图、对比图进行多实验对比，并支持导出 PNG 图片。

## 功能

- 导入：点击「导入数据」选择实验目录（可多选、可拖拽），自动识别实验名称，例如
  `attacks/pgd/outputs/ml100k/lightgcn/2026-08-09-21-54/` 显示为
  `attack-pgd-ml100k-lightgcn-2026年08月09日21时54分`。
- 图表：折线图（loss/指标随 epoch 变化）、直方图（最终/最佳指标对比）、
  对比图（攻击实验 Clean vs Poisoned）。
- 多选对比：勾选多个实验，自动分配不同颜色叠加显示。
- 编辑：修改实验名称、颜色，显示/隐藏，删除；修改图表标题；折线图支持
  「编辑数据点」——按 epoch 勾选要显示的值。
- 清空：一键清除全部导入数据与本地缓存（localStorage）。
- 导出：当前图表导出 PNG（2x 分辨率、白底）；实验数据导出/导入快照 JSON。
- 持久化：localStorage 保存导入与编辑状态，刷新后保留。

## 目录结构

```
visualization/
  index.html            # 页面
  styles.css            # 样式
  js/parser.js          # 路径识别 + 实验产物解析（纯函数）
  js/transforms.js      # 图表数据变换（纯函数）
  js/app.js             # 导入分组 / 状态 / 持久化
  js/main.js            # 页面装配与渲染
  lib/echarts.min.js    # 内置 ECharts（离线可用）
  tests/                # Node 标准库单测（无第三方依赖）
  README.md
```

## 使用

方式一：直接双击 `index.html`（file:// 打开）。

方式二：本地静态服务器：

```bash
cd TPA/visualization
python -m http.server 8080
```

然后浏览器访问 `http://localhost:8080/`。

### 导入技巧

- 导入只认形如 `2026-08-09-21-54` 的「子实验目录」，output 根目录下的其他文件
  （`xx.py`、`xx.json`、checkpoints、surrogate 等）一律忽略。
- 选择 `outputs` 根目录（如 `models/lightgcn/outputs/`）：批量导入其下所有子实验
  （推荐）。
- 直接选择某个子实验目录：按单个实验导入，并读取目录内 `config.yaml` 快照补全
  名称（攻击实验读 `attack.name`，模型实验读 `checkpoint_dir` 推断模型名）。
- 支持的文件：`history.json`、`eval_log.csv`、`config.yaml`、
  `*_comparison.json`。

## 图表说明

- 折线图：x 轴为 epoch，y 轴为所选指标（train_loss / val_loss / recall@K /
  ndcg@K / target_* 等），每条线一个实验；默认显示全部 epoch，可点实验列表中的
  「数据点」只显示选中的 epoch。
- 直方图：x 轴为实验，按指标分组柱；值取 `history.json` 的 best（无 best 时取
  最后一个有效值）。
- 对比图：仅攻击实验有意义；支持模型效用指标（recall@10 / ndcg@10）与目标物品
  平均 HR/NDCG（hr@k / ndcg@k），每组柱含 Clean / Poisoned。

## 图片导出

点击「导出图片」下载当前图表 PNG，文件名规则：
`tpa-{图表类型}-{指标}-{YYYYMMDD-HHmm}.png`，例如
`tpa-line-ndcg@10-20260815-1430.png`。

## 测试

Node 20+ 标准库单测（无第三方依赖）：

```bash
node --test TPA/visualization/tests/
```

## 手动验证清单

1. `file://` 打开 `index.html`，导入
   `TPA/attacks/pgd/outputs/ml100k/lightgcn/2026-08-09-21-54/`，
   列表显示 `attack-pgd-ml100k-lightgcn-2026年08月09日21时54分`。
2. 再导入 `models/lightgcn/outputs/2026-08-09-23-01/` 与
   `models/lightgcn/outputs/2026-08-09-23-39/`，勾选全部，折线图显示 3 条
   不同颜色曲线（指标切到 ndcg@10）。
3. 直方图显示多指标分组柱；对比图（攻击实验）显示 Clean/Poisoned 两组柱。
4. 点某实验的「数据点」，只勾选部分 epoch，折线图仅显示这些点。
5. 编辑标签/颜色、隐藏一条、改标题，刷新页面后状态保留。
6. 导出快照 → 清空 → 导入快照恢复。
7. 点击「导出图片」，下载 `tpa-line-ndcg@10-*.png`（或其他图表类型），
   图片可打开、内容为当前图表（2x 分辨率、白底）。

## 已知限制

- 纯前端只读：编辑仅作用于展示元数据与图表配置，不修改原始数据文件。
- 攻击实验若未生成 `*_comparison.json`，对比图会显示空值。
- localStorage 容量有限（约 5MB），数据过大时提示并仅本次会话保留。
