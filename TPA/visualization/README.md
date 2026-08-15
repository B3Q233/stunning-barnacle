# TPA 攻击实验可视化（demo v1）

纯前端单页：输入 `history.json` 与 `attack_comparison.json`，输出每轮指标折线图
与 Clean vs Poisoned 直方图，支持指标勾选显隐与标准 JSON 导出。

## 打开方式

直接用浏览器打开 `index.html`（`file://` 即可，无需服务器）。

## 使用方法

1. 点击「选择」加载 `history.json` 与 `attack_comparison.json`（或拖拽到虚线框）。
2. 折线图按 epoch 展示每轮指标（train_loss / val_loss / recall@10 / ndcg@10 /
   未来新增指标自动出现）。
3. 直方图展示模型效用与目标命中指标的 Clean vs Poisoned 对比。
4. 取消勾选指标复选框即时隐藏对应曲线/柱组。
5. 「导出标准 JSON」下载 `{epoch: {指标}}` 文件。

## 测试

```bash
node --test TPA/visualization/tests/
```

## 手动验证清单

- 选择
  `TPA/attacks/random/outputs/ml100k/lightgcn/2026-08-15-07-23/history.json`
  与同目录 `attack_comparison.json`。
- 摘要显示 epoch 数 30 与指标列表；折线图 6 条曲线；直方图 Clean/Poisoned 分组柱。
- 勾选/取消即时重绘；导出 JSON 为 `{"1": {...}, ..., "30": {...}}` 且无 `targets`。
- 选择坏 JSON 文件时页面提示错误且不崩溃。
