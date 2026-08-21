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

## Schema Registry + 可视化设计器

可视化引擎只认识统一的中间格式 `{title, type, x, series:[{name,data}]}`。
每种 JSON 结构通过一个 Schema（JSON 文件）声明"展示哪些字段、如何展示"，
按**结构指纹**匹配，与文件名无关。

### 导入流程（Runtime / Designer 双模式）

```
导入 JSON → 生成结构指纹 → 已有 Schema？→ 是：直接渲染
                                → 否：打开设计器
                                       ├─ 选择图类型（折线/柱状/饼图/指标卡）
                                       ├─ JSON 树状展示，勾选叶子字段（number 默认勾选）
                                       ├─ 修改字段别名（如 history[].target_ndcg@10 → Target NDCG）
                                       ├─ 折线图指定 X（默认第一个数值字段）
                                       └─ 保存并渲染 → Schema 写入 localStorage 自动注册
```

第二次导入**同结构**的 JSON（值不同、结构相同）会直接按指纹匹配渲染，无需再次配置。

### 目录

```
registry/
├── detector.js        # 结构指纹 fingerprint(json) + schemaId（fp_xxxxxxxx）
├── path.js            # 通用路径解析：history[].epoch / summary.best_hr / targets.908.ndcg
├── registry.js        # 注册中心：match(json) / saveCustom / all
├── normalize.js       # normalize(json, schema) → 统一格式
└── schemas/builtin.js # 浏览器内嵌的内置 schema 镜像

schema/                # 内置 schema 规范源（*.schema.json，指纹+series 声明）
├── history.schema.json
├── comparison.schema.json
├── tier_stats.schema.json
└── meta.schema.json

designer/
├── tree_builder.js    # JSON → 树（对象/数组分支、叶子含完整路径与类型）
└── designer.js        # 设计器弹窗：图类型 + 树状勾选 + 别名 + 保存/导出
```

### 自定义 schema 的持久化

设计器保存的 Schema 写入浏览器 `localStorage`（key `tpa_vis_custom_schemas`），
刷新页面后仍可复用；"导出 Schema"可下载 JSON（如 `fp_a81c92.schema.json`），
如需团队共享可放入 `schema/custom/` 并同步 `builtin.js` 镜像。

### 指纹匹配说明

`fingerprint(json)` 只记录键路径集合（数组记 `history[]`，递归首元素），
值不同但结构相同 → 指纹相同 → 同一 Schema。例如：

```json
{"history":[{"epoch":1,"loss":0.2}]}
```

指纹：`["history","history[]","history[].epoch","history[].loss"]`。
注意：键名参与指纹，因此 target id 类动态键（如 `targets.908`）会使指纹不同，
这类数据建议用设计器为具体结构保存一次自定义 Schema。

### 测试

```bash
node --test TPA/visualization/tests/
```

新增用例：detector（指纹/哈希）、path（路径解析）、registry（内置匹配/自定义保存）、
tree_builder（树与类型）、schemas（各内置结构归一化）、builtin（镜像与规范源一致）。

## 手动验证清单

- 添加实验卡（输入名称 / 路径自动命名）后弹窗关闭，左侧列表出现该项。
- 重置 H 后卡内重新导入 history.json，折线图正常显示（不再空白）。
- history 组勾选只影响折线图；comparison 组勾选只影响直方图。
- 取消勾选左侧实验卡后其曲线/柱消失；点击切换选中卡显示对应面板。
- 删除弹窗确认后移除；导出 JSON 为 `{"1": {...}, ...}`。
