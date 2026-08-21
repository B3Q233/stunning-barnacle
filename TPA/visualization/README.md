# TPA 攻击实验可视化（demo v4）

纯前端单页：以「实验卡」管理多个攻击实验（`history.json` +
`*_comparison.json`），左侧简要列表 + 选中卡面板；history 与 comparison
选项分组、各管各的图；指标可勾选显隐、自定义颜色；Nature 顶刊数据色板 +
现代 UI 主题（设计令牌 / 靛蓝主色 / 毛玻璃吸顶 / 药丸徽章）。

## 打开方式

直接用浏览器打开 `index.html`（`file://` 即可，无需服务器）。

## 使用方法

### 添加实验卡

- 点击「＋ 添加实验卡」打开弹窗：
  - 输入名称后确定；
  - 或点「导入实验路径自动命名」选择实验目录：自动按「攻击方法-实验时间」
    命名（如 `random-2026-08-15-07-23`）并读入数据，弹窗自动关闭。

### 左侧实验列表

- 每项：`[勾选框] 名称 + H✓/C✓ 状态`（H=history，C=comparison）。
- 点击切换选中卡（高亮），主区域显示其完整面板；只有勾选的卡参与图表渲染。

### 选中卡面板

- 卡头：名称 + 数据状态 + 按钮（改名 / 重置 H / 重置 C / 删除）。
- 导入区（始终可见，重置后可直接重新导入）：
  - 「导入实验路径」：选实验目录，读入 `history.json` 与 `*_comparison.json`；
  - `history.json` / `comparison` 两个文件入口：卡内单独选择。
- 指标分组（有数据后显示）：
  - 「history 指标」：train_loss / val_loss / recall@10 / ndcg@10 等，
    只影响折线图；
  - 「comparison 对比项」：模型效用 / 目标命中，只影响直方图。
  - 每组每指标：checkbox（默认全选）+ 颜色选择器，修改即时重绘对应图。

### 图表

- 折线图：同一指标按实验拆线，x 轴为所有实验 epoch 并集；标题/图例/绘图区
  分层布局，不重叠。
- 直方图：x 轴为实验，每个实验一组 Clean / Poisoned 柱，标签水平显示。
- 无数据时显示占位提示；重置后重新导入可正常重建（ECharts 实例正确释放）。

### 导出

- 「导出当前实验卡标准 JSON」：下载当前选中卡的 `{epoch: {指标}}` 文件。

## 测试

```bash
node --test TPA/visualization/tests/
```

## Schema Registry（自定义 JSON 展示）

可视化引擎只认识统一的中间格式，每种 JSON 结构通过注册一个 Schema 声明
"展示哪些字段、如何展示"，新增格式无需改引擎代码。

### 目录

```
registry/
├── base.js            # VisualizationSchema 基类（name/title/type/x/series/match）
├── path.js            # 通用路径解析：history[].epoch / summary.best_hr / targets.908.ndcg
├── index.js           # register(schema) / getSchema(json) / schemas() / clear()
├── normalize.js       # normalize(json, schema) → {title,type,x,series:[{name,data}]}
└── schemas/
    ├── history.js     # line：history[] 折线
    ├── comparison.js  # metric：model_utility 对比
    ├── tier_stats.js  # bar：batch 各层均值
    └── meta.js        # metric：batch 元信息
```

### 使用

在实验卡"导入实验路径"之外，新增 **自定义 JSON** 入口：选择任意 JSON 文件，
引擎按 `match(json)` 匹配注册的 Schema 并渲染（line=折线 / bar=柱状 / metric=指标卡）。
内置 schema 覆盖 `history.json`、`*_comparison.json`、批量 `tier_stats.json` 与
`meta.json`。

### 新增一种 JSON 格式（约 15 行）

```js
// registry/schemas/my_format.js
const { VisualizationSchema } = require('../base.js');
const { register } = require('../index.js');

class MyFormatSchema extends VisualizationSchema {
  constructor() {
    super();
    this.name = 'my_format';
    this.title = 'My Format';
    this.type = 'line';
    this.x = 'result[].step';
    this.series = {
      'ASR': 'result[].attack_success',
      'HR': 'result[].hr',
      'NDCG': 'result[].ndcg',
    };
  }
  match(json) {
    return Array.isArray(json && json.result);
  }
}

register(new MyFormatSchema());
```

然后在 `index.html` 的 `<script>` 列表中加入该文件即可，引擎与其它 schema 无需改动。

## 手动验证清单

- 添加实验卡（输入名称 / 路径自动命名）后弹窗关闭，左侧列表出现该项。
- 重置 H 后卡内重新导入 history.json，折线图正常显示（不再空白）。
- history 组勾选只影响折线图；comparison 组勾选只影响直方图。
- 取消勾选左侧实验卡后其曲线/柱消失；点击切换选中卡显示对应面板。
- 删除弹窗确认后移除；导出 JSON 为 `{"1": {...}, ...}`。
