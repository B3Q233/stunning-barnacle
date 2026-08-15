# 攻击实验可视化 demo 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付纯前端单页，输入 `history.json` 与 `attack_comparison.json`，输出每轮指标折线图与 Clean vs Poisoned 直方图，并支持指标勾选显隐与标准 JSON 导出。

**Architecture:** 纯静态单页（`file://` 打开即用）。`js/parser.js` 负责把 history 提取为标准 JSON `{epoch: {指标}}` 并解析 comparison；`js/transforms.js` 负责构建 ECharts series；`js/main.js` 是薄 UI 装配层。模块用 UMD 包装，Node 下可单测。

**Tech Stack:** 原生 HTML/CSS/JS（ES2020+）、ECharts 5（本地 vendored）、Node 标准库 `assert` 单测。

## Global Constraints

- 不新增第三方 Python 依赖；Python unittest 保持通过（本功能不改 Python 代码）。
- JS 测试只用 Node 标准库 `assert`；运行命令 `node --test TPA/visualization/tests/`。
- 文档与提交信息使用中文；Conventional Commits，type ∈ feat / fix / docs / chore，scope 为 `visualization`。
- 每个任务结束跑相关测试；只 `git add` 明确路径，禁止 `git add -f`。
- 原始数据文件只读，不修改。
- JS 模块用 UMD 包装：Node 下 `module.exports`，浏览器下挂 `window.TPAVisualizer.{parser,transforms,app}`。

---

### Task 1: 移除旧版可视化（git 清理）

**Files:**
- Delete（工作区已删除，需提交）：`TPA/visualization/*`

**Interfaces:**
- Consumes: 无
- Produces: git HEAD 中不再存在 `TPA/visualization/`，后续任务可在同路径新建文件且不受旧版干扰。

- [ ] **Step 1: 检查删除状态**

Run: `git -C G:\Idea status --short -- TPA/visualization`
Expected: 全部文件为 ` D`（工作区已删、未暂存）。

- [ ] **Step 2: 提交删除**

```bash
git -C G:\Idea add -- TPA/visualization
git -C G:\Idea commit -m "chore(visualization): 移除旧版可视化实现（工作区已删除，重新开发）"
```

- [ ] **Step 3: 验证**

Run: `git -C G:\Idea status --short -- TPA/visualization`
Expected: 无输出（已提交，工作区干净）。

---

### Task 2: parser.js（history 提取 + comparison 解析）

**Files:**
- Create: `TPA/visualization/js/parser.js`
- Test: `TPA/visualization/tests/parser.test.js`

**Interfaces:**
- Consumes: 无（纯函数，输入文件文本）。
- Produces（Task 3/4 依赖）:
  - `parseHistoryRows(text: string): Array<Record<string, number>>` — 每轮记录的数值标量字段（含 `epoch`），兼容顶层数组与 `{history:[...]}`；坏 JSON / 无记录 / 缺 epoch 抛 `Error`（中文消息）。
  - `extractEpochMetrics(text: string): Record<string, Record<string, number>>` — 标准 JSON `{"1": {指标}, "2": {...}}`，key 为 `String(epoch)`，值不含 `epoch`。
  - `listMetrics(epochMetrics: Record<string, Record<string, number>>): string[]` — 按出现顺序枚举数值指标名。
  - `parseComparison(text: string): {modelUtility, targetMetrics}` — `model_utility` / `target_metrics` 映射为 camelCase。

- [ ] **Step 1: 写失败测试**

```js
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const {
  parseHistoryRows, extractEpochMetrics, listMetrics, parseComparison,
} = require('../js/parser.js');

test('parseHistoryRows 兼容顶层数组形态并跳过嵌套 targets', () => {
  const rows = parseHistoryRows(JSON.stringify([
    { epoch: 1, train_loss: 0.5, val_loss: 0.4, targets: { '251': { 'hr@k': 0.1 } } },
  ]));
  assert.deepStrictEqual(rows, [{ epoch: 1, train_loss: 0.5, val_loss: 0.4 }]);
});

test('parseHistoryRows 兼容 {history:[...]} 形态并忽略 best', () => {
  const rows = parseHistoryRows(JSON.stringify({
    history: [{ epoch: 2, train_loss: 0.3 }],
    best: { 'recall@10': { value: 0.9 } },
  }));
  assert.deepStrictEqual(rows, [{ epoch: 2, train_loss: 0.3 }]);
});

test('坏 JSON 抛可读错误', () => {
  assert.throws(() => parseHistoryRows('{oops'), /解析失败/);
});

test('空 history 与缺 epoch 抛错', () => {
  assert.throws(() => parseHistoryRows('{"history":[]}'), /没有 history/);
  assert.throws(() => parseHistoryRows(JSON.stringify([{ train_loss: 0.1 }])), /epoch/);
});

test('extractEpochMetrics 输出 {epoch: {指标}}', () => {
  const out = extractEpochMetrics(JSON.stringify([
    { epoch: 1, train_loss: 0.5, 'recall@10': 0.1 },
    { epoch: 2, train_loss: 0.4, 'recall@10': 0.2 },
  ]));
  assert.deepStrictEqual(out, {
    '1': { train_loss: 0.5, 'recall@10': 0.1 },
    '2': { train_loss: 0.4, 'recall@10': 0.2 },
  });
});

test('listMetrics 按出现顺序枚举数值字段', () => {
  const out = extractEpochMetrics(JSON.stringify([
    { epoch: 1, train_loss: 0.5, val_loss: 0.4, 'recall@10': 0.1 },
    { epoch: 2, train_loss: 0.4, 'ndcg@10': 0.2 },
  ]));
  assert.deepStrictEqual(listMetrics(out), ['train_loss', 'val_loss', 'recall@10', 'ndcg@10']);
});

test('parseComparison 映射 model_utility 与 target_metrics', () => {
  const cmp = parseComparison(JSON.stringify({
    model_utility: { clean: { 'recall@10': 0.1 }, poisoned: { 'recall@10': 0.2 } },
    target_metrics: {
      clean: { '251': { 'hr@k': 0.01, 'ndcg@k': 0.005 } },
      poisoned: {},
    },
  }));
  assert.strictEqual(cmp.modelUtility.clean['recall@10'], 0.1);
  assert.strictEqual(cmp.modelUtility.poisoned['recall@10'], 0.2);
  assert.strictEqual(cmp.targetMetrics.clean['251']['hr@k'], 0.01);
  assert.deepStrictEqual(cmp.targetMetrics.poisoned, {});
});
```

- [ ] **Step 2: 运行确认失败**

Run: `node --test TPA/visualization/tests/parser.test.js`
Expected: FAIL，`Cannot find module '../js/parser.js'`。

- [ ] **Step 3: 实现 parser.js**

```js
'use strict';
(function (global, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    global.TPAVisualizer = global.TPAVisualizer || {};
    global.TPAVisualizer.parser = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {

  function parseHistoryRows(text) {
    let data;
    try {
      data = JSON.parse(text);
    } catch (e) {
      throw new Error(`history.json 解析失败: ${e.message}`);
    }
    const rows = Array.isArray(data)
      ? data
      : (data && Array.isArray(data.history) ? data.history : []);
    if (!rows.length) throw new Error('history.json 中没有 history 记录');
    return rows.map((row, idx) => {
      if (!row || typeof row !== 'object') {
        throw new Error(`history.json 第 ${idx + 1} 条记录不是对象`);
      }
      const out = {};
      for (const [k, v] of Object.entries(row)) {
        if (typeof v === 'number') out[k] = v;
      }
      if (!('epoch' in out)) {
        throw new Error(`history.json 第 ${idx + 1} 条记录缺少 epoch 数值字段`);
      }
      return out;
    });
  }

  function extractEpochMetrics(text) {
    const rows = parseHistoryRows(text);
    const out = {};
    for (const row of rows) {
      const metrics = {};
      for (const [k, v] of Object.entries(row)) {
        if (k !== 'epoch') metrics[k] = v;
      }
      out[String(row.epoch)] = metrics;
    }
    return out;
  }

  function listMetrics(epochMetrics) {
    const names = [];
    const seen = new Set();
    for (const metrics of Object.values(epochMetrics)) {
      for (const name of Object.keys(metrics)) {
        if (!seen.has(name)) {
          seen.add(name);
          names.push(name);
        }
      }
    }
    return names;
  }

  function parseComparison(text) {
    let data;
    try {
      data = JSON.parse(text);
    } catch (e) {
      throw new Error(`comparison 解析失败: ${e.message}`);
    }
    return {
      modelUtility: (data && data.model_utility) || { clean: {}, poisoned: {} },
      targetMetrics: (data && data.target_metrics) || { clean: {}, poisoned: {} },
    };
  }

  return { parseHistoryRows, extractEpochMetrics, listMetrics, parseComparison };
});
```

- [ ] **Step 4: 运行确认通过**

Run: `node --test TPA/visualization/tests/test_parser.js`
Expected: PASS，6 个用例。

- [ ] **Step 5: 提交**

```bash
git -C G:\Idea add -- TPA/visualization/js/parser.js TPA/visualization/tests/parser.test.js
git -C G:\Idea commit -m "feat(visualization): history 解析与标准 JSON 提取（含 Node 单测）"
```

---

### Task 3: transforms.js（series 构建 + 指标显隐）

**Files:**
- Create: `TPA/visualization/js/transforms.js`
- Test: `TPA/visualization/tests/transforms.test.js`

**Interfaces:**
- Consumes: Task 2 的 `extractEpochMetrics` / `parseComparison` 输出结构。
- Produces（Task 4 依赖）:
  - `PALETTE: string[]`
  - `buildLineSeries(epochMetrics, selectedMetrics): {xAxis: string[], series: [{name, type:'line', color, data: (number|null)[]}]}` — x 为按数值排序的 epoch key，缺指标为 `null`。
  - `buildComparisonItems(comparison): [{name, clean, poisoned}]` — 模型效用各指标 + 目标 `target_hr@k` / `target_ndcg@k`（多目标取平均）。
  - `buildComparisonSeries(items, selectedNames): {xAxis, series: [{name:'Clean'|'Poisoned', type:'bar', data}]}`。

- [ ] **Step 1: 写失败测试**

```js
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const {
  buildLineSeries, buildComparisonItems, buildComparisonSeries,
} = require('../js/transforms.js');

test('折线 series：x 按 epoch 排序，缺指标为 null', () => {
  const em = {
    '1': { train_loss: 0.5, 'recall@10': 0.1 },
    '3': { train_loss: 0.3 },
    '2': { train_loss: 0.4, 'recall@10': 0.2 },
  };
  const { xAxis, series } = buildLineSeries(em, ['train_loss', 'recall@10']);
  assert.deepStrictEqual(xAxis, ['1', '2', '3']);
  assert.strictEqual(series[0].type, 'line');
  assert.deepStrictEqual(series[0].data, [0.5, 0.4, 0.3]);
  assert.deepStrictEqual(series[1].data, [0.1, 0.2, null]);
});

test('直方对比项：模型效用 + 目标多目标取平均', () => {
  const cmp = {
    modelUtility: {
      clean: { 'recall@10': 0.1, 'ndcg@10': 0.2 },
      poisoned: { 'recall@10': 0.2, 'ndcg@10': 0.15 },
    },
    targetMetrics: {
      clean: {
        '251': { 'hr@k': 0.01, 'ndcg@k': 0.005 },
        '252': { 'hr@k': 0.03, 'ndcg@k': 0.015 },
      },
      poisoned: {
        '251': { 'hr@k': 0.05, 'ndcg@k': 0.02 },
        '252': { 'hr@k': 0.07, 'ndcg@k': 0.04 },
      },
    },
  };
  const items = buildComparisonItems(cmp);
  const names = items.map((it) => it.name);
  assert.ok(names.includes('recall@10'));
  assert.ok(names.includes('target_hr@k'));
  const hr = items.find((it) => it.name === 'target_hr@k');
  assert.strictEqual(hr.clean, 0.02);
  assert.strictEqual(hr.poisoned, 0.06);
});

test('直方 series：按选中项生成 Clean/Poisoned 两列', () => {
  const items = [
    { name: 'recall@10', clean: 0.1, poisoned: 0.2 },
    { name: 'ndcg@10', clean: 0.2, poisoned: 0.15 },
  ];
  const { xAxis, series } = buildComparisonSeries(items, ['ndcg@10']);
  assert.deepStrictEqual(xAxis, ['ndcg@10']);
  assert.strictEqual(series[0].name, 'Clean');
  assert.deepStrictEqual(series[0].data, [0.2]);
  assert.deepStrictEqual(series[1].data, [0.15]);
});
```

- [ ] **Step 2: 运行确认失败**

Run: `node --test TPA/visualization/tests/transforms.test.js`
Expected: FAIL，`Cannot find module '../js/transforms.js'`。

- [ ] **Step 3: 实现 transforms.js**

```js
'use strict';
(function (global, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    global.TPAVisualizer = global.TPAVisualizer || {};
    global.TPAVisualizer.transforms = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {

  const PALETTE = [
    '#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de',
    '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc',
  ];

  function colorFor(index) {
    return PALETTE[index % PALETTE.length];
  }

  function buildLineSeries(epochMetrics, selectedMetrics) {
    const xAxis = Object.keys(epochMetrics).sort((a, b) => Number(a) - Number(b));
    const series = selectedMetrics.map((name, i) => ({
      name,
      type: 'line',
      color: colorFor(i),
      data: xAxis.map((epoch) => {
        const v = epochMetrics[epoch][name];
        return typeof v === 'number' ? v : null;
      }),
    }));
    return { xAxis, series };
  }

  function avg(values) {
    const nums = values.filter((v) => typeof v === 'number');
    if (!nums.length) return null;
    return nums.reduce((a, b) => a + b, 0) / nums.length;
  }

  function buildComparisonItems(comparison) {
    const items = [];
    const mu = comparison.modelUtility || {};
    const muNames = new Set([
      ...Object.keys(mu.clean || {}),
      ...Object.keys(mu.poisoned || {}),
    ]);
    for (const name of muNames) {
      items.push({
        name,
        clean: mu.clean ? mu.clean[name] ?? null : null,
        poisoned: mu.poisoned ? mu.poisoned[name] ?? null : null,
      });
    }
    const tm = comparison.targetMetrics || {};
    const hr = { clean: [], poisoned: [] };
    const ndcg = { clean: [], poisoned: [] };
    for (const side of ['clean', 'poisoned']) {
      for (const t of Object.values(tm[side] || {})) {
        if (typeof t['hr@k'] === 'number') hr[side].push(t['hr@k']);
        if (typeof t['ndcg@k'] === 'number') ndcg[side].push(t['ndcg@k']);
      }
    }
    items.push({ name: 'target_hr@k', clean: avg(hr.clean), poisoned: avg(hr.poisoned) });
    items.push({ name: 'target_ndcg@k', clean: avg(ndcg.clean), poisoned: avg(ndcg.poisoned) });
    return items;
  }

  function buildComparisonSeries(items, selectedNames) {
    const picked = items.filter((it) => selectedNames.includes(it.name));
    return {
      xAxis: picked.map((it) => it.name),
      series: [
        { name: 'Clean', type: 'bar', data: picked.map((it) => it.clean) },
        { name: 'Poisoned', type: 'bar', data: picked.map((it) => it.poisoned) },
      ],
    };
  }

  return { PALETTE, colorFor, buildLineSeries, buildComparisonItems, buildComparisonSeries };
});
```

- [ ] **Step 4: 运行确认通过**

Run: `node --test TPA/visualization/tests/test_transforms.js`
Expected: PASS，3 个用例。

- [ ] **Step 5: 提交**

```bash
git -C G:\Idea add -- TPA/visualization/js/transforms.js TPA/visualization/tests/transforms.test.js
git -C G:\Idea commit -m "feat(visualization): 折线/直方 series 构建与指标显隐（含 Node 单测）"
```

---

### Task 4: 页面（index.html + styles.css + main.js + ECharts）

**Files:**
- Create: `TPA/visualization/index.html`
- Create: `TPA/visualization/styles.css`
- Create: `TPA/visualization/js/main.js`
- Create: `TPA/visualization/lib/echarts.min.js`（下载，需联网批准；拒绝时退回 CDN `<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js">`）

**Interfaces:**
- Consumes: Task 2 `extractEpochMetrics` / `listMetrics` / `parseComparison`；Task 3 `buildLineSeries` / `buildComparisonItems` / `buildComparisonSeries`；全局 `echarts`。
- Produces: `window.TPAVisualizer.app.initApp()`（页面底部调用），提供 `handleFiles(fileList)` 供拖拽/选择复用。

- [ ] **Step 1: 下载 ECharts（需网络，执行时请求批准）**

```bash
curl -L -o TPA/visualization/lib/echarts.min.js https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js
```
校验：文件约 1MB，开头含 ECharts 版权注释；否则视为失败，退回 CDN。

- [ ] **Step 2: 写 index.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>TPA 攻击实验可视化</title>
  <link rel="stylesheet" href="styles.css">
  <script src="lib/echarts.min.js"></script>
</head>
<body>
  <header>
    <h1>TPA 攻击实验可视化</h1>
    <div class="inputs">
      <label>history.json <input id="history-input" type="file" accept=".json"></label>
      <label>attack_comparison.json <input id="comparison-input" type="file" accept=".json"></label>
      <button id="export-json">导出标准 JSON</button>
    </div>
    <div id="drop-zone">拖拽文件到此处（history.json / *_comparison.json）</div>
    <div id="summary" class="summary"></div>
    <div id="message" class="message"></div>
  </header>
  <main>
    <section class="controls">
      <fieldset><legend>折线图指标</legend><div id="line-metrics" class="checks"></div></fieldset>
      <fieldset><legend>直方图对比项</legend><div id="bar-metrics" class="checks"></div></fieldset>
    </section>
    <section class="charts">
      <div id="chart-line" class="chart"></div>
      <div id="chart-bar" class="chart"></div>
    </section>
  </main>
  <script src="js/parser.js"></script>
  <script src="js/transforms.js"></script>
  <script src="js/main.js"></script>
  <script>window.TPAVisualizer.app.initApp();</script>
</body>
</html>
```

- [ ] **Step 3: 写 styles.css**

```css
body { margin: 0; font-family: "Microsoft YaHei", Arial, sans-serif; color: #333; }
header { padding: 16px 24px; background: #f7f8fa; border-bottom: 1px solid #e4e7ed; }
h1 { margin: 0 0 12px; font-size: 20px; }
.inputs { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
#drop-zone { margin-top: 10px; padding: 12px; border: 2px dashed #c0c4cc;
  border-radius: 6px; text-align: center; color: #909399; }
#drop-zone.active { border-color: #5470c6; color: #5470c6; }
.summary { margin-top: 8px; font-size: 13px; color: #606266; }
.message { margin-top: 6px; font-size: 13px; color: #e6a23c; }
.message.error { color: #f56c6c; }
main { padding: 16px 24px; }
.controls { display: flex; gap: 24px; margin-bottom: 16px; }
.checks { display: flex; flex-wrap: wrap; gap: 4px 14px; }
.checks label { font-size: 13px; cursor: pointer; }
.charts { display: flex; flex-direction: column; gap: 16px; }
.chart { width: 100%; height: 420px; border: 1px solid #e4e7ed; border-radius: 6px; }
.placeholder { padding: 80px; text-align: center; color: #909399; }
```

- [ ] **Step 4: 写 main.js**

```js
'use strict';
(function (global) {
  const parser = global.TPAVisualizer.parser;
  const transforms = global.TPAVisualizer.transforms;
  const state = { epochMetrics: null, comparison: null, lineSelected: [], barSelected: [] };
  let lineChart = null;
  let barChart = null;
  const $ = (id) => document.getElementById(id);

  function showMessage(text, isError) {
    $('message').textContent = text;
    $('message').className = isError ? 'message error' : 'message';
  }

  function readFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(new Error(`读取 ${file.name} 失败`));
      reader.readAsText(file, 'utf-8');
    });
  }

  async function handleFiles(fileList) {
    const errors = [];
    for (const file of fileList) {
      try {
        const text = await readFile(file);
        if (file.name === 'history.json') {
          state.epochMetrics = parser.extractEpochMetrics(text);
        } else if (file.name.endsWith('_comparison.json')) {
          state.comparison = parser.parseComparison(text);
        } else {
          errors.push(`跳过未知文件: ${file.name}`);
        }
      } catch (e) {
        errors.push(`${file.name}: ${e.message}`);
      }
    }
    renderAll();
    if (errors.length) showMessage(errors.join('；'), true);
  }

  function renderSummary() {
    if (!state.epochMetrics) {
      $('summary').textContent = '未加载数据';
      return;
    }
    const epochs = Object.keys(state.epochMetrics).length;
    $('summary').textContent =
      `epoch 数: ${epochs}；指标: ${parser.listMetrics(state.epochMetrics).join(', ')}`;
  }

  function renderLineCheckboxes() {
    const box = $('line-metrics');
    box.innerHTML = '';
    const metrics = state.epochMetrics ? parser.listMetrics(state.epochMetrics) : [];
    state.lineSelected = metrics.slice();
    for (const m of metrics) {
      const label = document.createElement('label');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = true;
      cb.addEventListener('change', () => {
        state.lineSelected = cb.checked
          ? [...new Set([...state.lineSelected, m])]
          : state.lineSelected.filter((x) => x !== m);
        renderLineChart();
      });
      label.appendChild(cb);
      label.appendChild(document.createTextNode(m));
      box.appendChild(label);
    }
  }

  function renderBarCheckboxes() {
    const box = $('bar-metrics');
    box.innerHTML = '';
    const items = state.comparison ? transforms.buildComparisonItems(state.comparison) : [];
    state.barSelected = items.map((it) => it.name);
    for (const it of items) {
      const label = document.createElement('label');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = true;
      cb.addEventListener('change', () => {
        state.barSelected = cb.checked
          ? [...new Set([...state.barSelected, it.name])]
          : state.barSelected.filter((x) => x !== it.name);
        renderBarChart();
      });
      label.appendChild(cb);
      label.appendChild(document.createTextNode(it.name));
      box.appendChild(label);
    }
  }

  function renderLineChart() {
    if (!state.epochMetrics) return;
    lineChart = lineChart || echarts.init($('chart-line'));
    const { xAxis, series } = transforms.buildLineSeries(state.epochMetrics, state.lineSelected);
    lineChart.setOption({
      title: { text: '每轮指标折线图' },
      tooltip: { trigger: 'axis' },
      legend: { data: series.map((s) => s.name) },
      grid: { left: 60, right: 30, bottom: 60 },
      xAxis: { type: 'category', data: xAxis, name: 'epoch' },
      yAxis: { type: 'value' },
      dataZoom: [{ type: 'inside' }, { type: 'slider' }],
      series,
    }, true);
  }

  function renderBarChart() {
    if (!state.comparison) {
      $('chart-bar').innerHTML = '<div class="placeholder">未提供 attack_comparison.json，无对比数据</div>';
      return;
    }
    barChart = barChart || echarts.init($('chart-bar'));
    const items = transforms.buildComparisonItems(state.comparison);
    const { xAxis, series } = transforms.buildComparisonSeries(items, state.barSelected);
    barChart.setOption({
      title: { text: 'Clean vs Poisoned 对比' },
      tooltip: { trigger: 'axis' },
      legend: { data: ['Clean', 'Poisoned'] },
      grid: { left: 60, right: 30, bottom: 60 },
      xAxis: { type: 'category', data: xAxis, axisLabel: { rotate: 20 } },
      yAxis: { type: 'value' },
      series,
    }, true);
  }

  function renderAll() {
    renderSummary();
    renderLineCheckboxes();
    renderBarCheckboxes();
    renderLineChart();
    renderBarChart();
  }

  function exportJson() {
    if (!state.epochMetrics) {
      showMessage('没有可导出的数据', true);
      return;
    }
    const blob = new Blob([JSON.stringify(state.epochMetrics, null, 2)],
      { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'epoch_metrics.json';
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function initApp() {
    $('history-input').addEventListener('change', (e) => handleFiles(e.target.files));
    $('comparison-input').addEventListener('change', (e) => handleFiles(e.target.files));
    $('export-json').addEventListener('click', exportJson);
    const drop = $('drop-zone');
    ['dragenter', 'dragover'].forEach((ev) =>
      drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('active'); }));
    ['dragleave', 'drop'].forEach((ev) =>
      drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('active'); }));
    drop.addEventListener('drop', (e) => handleFiles(e.dataTransfer.files));
    showMessage('请选择 history.json 与 attack_comparison.json');
  }

  global.TPAVisualizer = global.TPAVisualizer || {};
  global.TPAVisualizer.app = { initApp, handleFiles, exportJson };
})(window);
```

- [ ] **Step 5: 提交**

```bash
git -C G:\Idea add -- TPA/visualization/index.html TPA/visualization/styles.css TPA/visualization/js/main.js TPA/visualization/lib/echarts.min.js
git -C G:\Idea commit -m "feat(visualization): 单页折线+直方图渲染与指标显隐（ECharts）"
```

---

### Task 5: README + 全量回归

**Files:**
- Create: `TPA/visualization/README.md`

**Interfaces:**
- Consumes: 全部前序任务产物。
- Produces: 使用文档与手动验证清单。

- [ ] **Step 1: 写 README.md**

```markdown
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
```

- [ ] **Step 2: 全量回归**

Run:
```bash
node --test TPA/visualization/tests/
cd TPA && G:\Idea\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```
Expected: JS 全绿；Python 全量通过（含 `test_history_completeness` 等 45 项）。

- [ ] **Step 3: 提交**

```bash
git -C G:\Idea add -- TPA/visualization/README.md
git -C G:\Idea commit -m "docs(visualization): 使用文档与手动验证清单"
```

---

## 计划自检

- Spec 覆盖：标准 JSON 提取（Task 2）、comparison 解析（Task 2）、折线/直方图
  （Task 3/4）、指标显隐（Task 3/4）、导出 JSON（Task 4）、错误提示（Task 2/4）、
  Node 单测（Task 2/3）、README 与全量回归（Task 5）、旧版 git 清理（Task 1）。
- 无占位符：每个代码步骤含完整代码；ECharts 下载失败有明确 CDN 回退。
- 类型一致：`epochMetrics` / `comparison` / `items` / `selectedNames` 命名在各任务间一致。

---

# 迭代 2（多实验选项卡 + 指标自定义）任务

对应 spec 第 12 节。基线：v1 已提交于 `feat/attack-viz-demo`。

### Task 6: parser.js 目录路径解析与自动命名

**Files:**
- Modify: `TPA/visualization/js/parser.js`（追加函数并导出）
- Test: `TPA/visualization/tests/parser.test.js`（追加用例）

**Interfaces:**
- Consumes: 无。
- Produces（Task 8 依赖）:
  - `parseDirectoryPath(relativePath: string): {method, dataset, model, runTag} | null`
  - `buildAutoName(info: object): string | null`（`${method}-${runTag}`）

- [ ] **Step 1: 追加失败测试**

在 `parser.test.js` 末尾追加：

```js
const { parseDirectoryPath, buildAutoName } = require('../js/parser.js');

test('parseDirectoryPath 解析攻击实验目录相对路径', () => {
  const info = parseDirectoryPath(
    'attacks/random/outputs/ml100k/lightgcn/2026-08-15-07-23/history.json');
  assert.deepStrictEqual(info, {
    method: 'random', dataset: 'ml100k', model: 'lightgcn',
    runTag: '2026-08-15-07-23',
  });
});

test('parseDirectoryPath 非实验路径返回 null', () => {
  assert.strictEqual(parseDirectoryPath('tmp/foo.txt'), null);
});

test('buildAutoName 组合 攻击方法-实验时间', () => {
  assert.strictEqual(
    buildAutoName({ method: 'random', runTag: '2026-08-15-07-23' }),
    'random-2026-08-15-07-23');
  assert.strictEqual(buildAutoName(null), null);
});
```

- [ ] **Step 2: 运行确认失败**

Run: `node --test TPA/visualization/tests/parser.test.js`
Expected: FAIL，`parseDirectoryPath is not a function`。

- [ ] **Step 3: 实现**

在 `parser.js` 的 factory 内、`return` 之前追加：

```js
function parseDirectoryPath(relativePath) {
  const parts = String(relativePath || '').split('/').filter(Boolean);
  const attacksIdx = parts.indexOf('attacks');
  const outIdx = parts.indexOf('outputs');
  const runIdx = parts.findIndex((s) => /^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}$/.test(s));
  if (attacksIdx === -1 || outIdx === -1 || runIdx === -1
      || !(outIdx > attacksIdx && runIdx > outIdx)) {
    return null;
  }
  return {
    method: parts[attacksIdx + 1] || null,
    dataset: parts[outIdx + 1] || null,
    model: parts[outIdx + 2] || null,
    runTag: parts[runIdx],
  };
}

function buildAutoName(info) {
  if (!info || !info.method || !info.runTag) return null;
  return `${info.method}-${info.runTag}`;
}
```

`return` 语句改为同时导出两个新函数。

- [ ] **Step 4: 运行确认通过**

Run: `node --test TPA/visualization/tests/parser.test.js`
Expected: PASS，10 个用例。

- [ ] **Step 5: 提交**

```bash
git -C G:\Idea add -- TPA/visualization/js/parser.js TPA/visualization/tests/parser.test.js
git -C G:\Idea commit -m "feat(visualization): 实验目录路径解析与自动命名（含 Node 单测）"
```

---

### Task 7: transforms.js 多实验 series

**Files:**
- Modify: `TPA/visualization/js/transforms.js`（追加函数并导出，保留 v1 函数）
- Test: `TPA/visualization/tests/transforms.test.js`（追加用例）

**Interfaces:**
- Consumes: Task 6 产物无关；输入结构为实验列表，每项：
  `{name, epochMetrics, comparison, metricOptions: {metric: {selected, color}}}`。
- Produces（Task 8 依赖）:
  - `buildMultiLineSeries(experiments): {xAxis: string[], series: [{name: `${metric}-${exp.name}`, type:'line', color, data}]}`（x 为所有实验 epoch 并集，缺失为 null）
  - `buildMultiBarSeries(experiments): {xAxis: 实验名[], series: [{name: `${metric}-Clean|Poisoned`, type:'bar', data}]}`（柱按实验指标色逐点着色）

- [ ] **Step 1: 追加失败测试**

在 `transforms.test.js` 末尾追加：

```js
const { buildMultiLineSeries, buildMultiBarSeries } = require('../js/transforms.js');

test('多实验折线：同一指标按实验拆线且颜色可自定义', () => {
  const exps = [
    { name: 'A', epochMetrics: { '1': { 'recall@10': 0.1 }, '2': { 'recall@10': 0.2 } },
      metricOptions: { 'recall@10': { selected: true, color: '#111111' } } },
    { name: 'B', epochMetrics: { '1': { 'recall@10': 0.3 } },
      metricOptions: { 'recall@10': { selected: true, color: '#333333' } } },
  ];
  const { xAxis, series } = buildMultiLineSeries(exps);
  assert.deepStrictEqual(xAxis, ['1', '2']);
  assert.deepStrictEqual(series.map((s) => s.name), ['recall@10-A', 'recall@10-B']);
  assert.strictEqual(series[0].color, '#111111');
  assert.deepStrictEqual(series[0].data, [0.1, 0.2]);
  assert.deepStrictEqual(series[1].data, [0.3, null]);
});

test('多实验直方：x=实验，每指标 Clean/Poisoned 两列并逐点着色', () => {
  const exps = [
    { name: 'A', comparison: { modelUtility: { clean: { 'recall@10': 0.1 }, poisoned: { 'recall@10': 0.2 } }, targetMetrics: {} },
      metricOptions: { 'recall@10': { selected: true, color: '#111111' } } },
    { name: 'B', comparison: { modelUtility: { clean: { 'recall@10': 0.3 }, poisoned: { 'recall@10': 0.4 } }, targetMetrics: {} },
      metricOptions: { 'recall@10': { selected: true, color: '#333333' } } },
  ];
  const { xAxis, series } = buildMultiBarSeries(exps);
  assert.deepStrictEqual(xAxis, ['A', 'B']);
  assert.deepStrictEqual(series.map((s) => s.name),
    ['recall@10-Clean', 'recall@10-Poisoned']);
  assert.strictEqual(series[0].data[0].value, 0.1);
  assert.strictEqual(series[0].data[0].itemStyle.color, '#111111');
  assert.strictEqual(series[1].data[1].value, 0.4);
});
```

- [ ] **Step 2: 运行确认失败**

Run: `node --test TPA/visualization/tests/transforms.test.js`
Expected: FAIL，`buildMultiLineSeries is not a function`。

- [ ] **Step 3: 实现**

在 `transforms.js` 的 factory 内、`return` 之前追加：

```js
function buildMultiLineSeries(experiments) {
  const xSet = new Set();
  const metricNames = [];
  for (const exp of experiments) {
    for (const e of Object.keys(exp.epochMetrics || {})) xSet.add(Number(e));
    for (const [name, opt] of Object.entries(exp.metricOptions || {})) {
      if (opt && opt.selected && !metricNames.includes(name)) metricNames.push(name);
    }
  }
  const xAxis = [...xSet].sort((a, b) => a - b).map(String);
  const series = [];
  for (const metric of metricNames) {
    for (const exp of experiments) {
      const opt = (exp.metricOptions || {})[metric];
      if (!opt || !opt.selected) continue;
      series.push({
        name: `${metric}-${exp.name}`,
        type: 'line',
        color: opt.color || colorFor(series.length),
        data: xAxis.map((e) => {
          const v = exp.epochMetrics[e] ? exp.epochMetrics[e][metric] : undefined;
          return typeof v === 'number' ? v : null;
        }),
      });
    }
  }
  return { xAxis, series };
}

function buildMultiBarSeries(experiments) {
  const xAxis = experiments.map((e) => e.name);
  const barNames = new Set();
  const metricNames = [];
  for (const exp of experiments) {
    for (const it of buildComparisonItems(exp.comparison || {})) barNames.add(it.name);
    for (const [name, opt] of Object.entries(exp.metricOptions || {})) {
      if (opt && opt.selected && barNames.has(name) && !metricNames.includes(name)) {
        metricNames.push(name);
      }
    }
  }
  const series = [];
  for (const metric of metricNames) {
    const cleanData = [];
    const poisonedData = [];
    for (const exp of experiments) {
      const opt = (exp.metricOptions || {})[metric];
      const item = buildComparisonItems(exp.comparison || {})
        .find((it) => it.name === metric);
      const color = opt ? opt.color : undefined;
      const mk = (v) => (v === null || v === undefined
        ? null : { value: v, itemStyle: color ? { color } : undefined });
      cleanData.push(mk(item ? item.clean : null));
      poisonedData.push(mk(item ? item.poisoned : null));
    }
    series.push({ name: `${metric}-Clean`, type: 'bar', data: cleanData });
    series.push({ name: `${metric}-Poisoned`, type: 'bar', data: poisonedData });
  }
  return { xAxis, series };
}
```

`return` 语句追加导出两个新函数。

- [ ] **Step 4: 运行确认通过**

Run: `node --test TPA/visualization/tests/transforms.test.js`
Expected: PASS，5 个用例。

- [ ] **Step 5: 提交**

```bash
git -C G:\Idea add -- TPA/visualization/js/transforms.js TPA/visualization/tests/transforms.test.js
git -C G:\Idea commit -m "feat(visualization): 多实验折线/直方 series 与逐点着色（含 Node 单测）"
```

---

### Task 8: 页面重构（侧栏选项卡 + 卡牌 + 双图）

**Files:**
- Modify: `TPA/visualization/index.html`
- Modify: `TPA/visualization/styles.css`
- Modify: `TPA/visualization/js/main.js`

**Interfaces:**
- Consumes: Task 6 `parseDirectoryPath` / `buildAutoName`；Task 7
  `buildMultiLineSeries` / `buildMultiBarSeries` / `buildComparisonItems` /
  `colorFor`；Task 2 `extractEpochMetrics` / `listMetrics` / `parseComparison`。
- Produces: `window.TPAVisualizer.app.initApp()`。

- [ ] **Step 1: 重写 index.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>TPA 攻击实验可视化</title>
  <link rel="stylesheet" href="styles.css">
  <script src="lib/echarts.min.js"></script>
</head>
<body>
  <header>
    <h1>TPA 攻击实验可视化</h1>
    <div class="inputs">
      <button id="btn-dir">导入实验目录</button>
      <input id="dir-input" type="file" webkitdirectory multiple hidden>
      <button id="btn-import">导入文件（多选）</button>
      <input id="import-input" type="file" multiple accept=".json" hidden>
      <button id="export-json">导出当前选项卡标准 JSON</button>
    </div>
    <div id="drop-zone">拖拽 history.json / *_comparison.json 到此处（自动新建选项卡导入）</div>
    <div id="message" class="message"></div>
  </header>
  <main class="layout">
    <aside class="sidebar">
      <h2>实验选项卡</h2>
      <button id="new-tab">+ 新建选项卡</button>
      <ul id="tab-list"></ul>
    </aside>
    <section class="content">
      <div id="metric-card" class="card"></div>
      <div id="chart-line" class="chart"></div>
      <div id="chart-bar" class="chart"></div>
    </section>
  </main>
  <script src="js/parser.js"></script>
  <script src="js/transforms.js"></script>
  <script src="js/main.js"></script>
  <script>window.TPAVisualizer.app.initApp();</script>
</body>
</html>
```

- [ ] **Step 2: 重写 styles.css**

```css
body { margin: 0; font-family: "Microsoft YaHei", Arial, sans-serif; color: #333; }
header { padding: 16px 24px; background: #f7f8fa; border-bottom: 1px solid #e4e7ed; }
h1 { margin: 0 0 12px; font-size: 20px; }
.inputs { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
button { cursor: pointer; }
#drop-zone { margin-top: 10px; padding: 10px; border: 2px dashed #c0c4cc;
  border-radius: 6px; text-align: center; color: #909399; font-size: 13px; }
#drop-zone.active { border-color: #5470c6; color: #5470c6; }
.message { margin-top: 6px; font-size: 13px; color: #e6a23c; }
.message.error { color: #f56c6c; }
.layout { display: grid; grid-template-columns: 280px 1fr; gap: 16px; padding: 16px 24px; }
.sidebar { border: 1px solid #e4e7ed; border-radius: 6px; padding: 12px; height: fit-content; }
.sidebar h2 { margin: 0 0 8px; font-size: 15px; }
#new-tab { width: 100%; margin-bottom: 8px; }
#tab-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; }
.tab-item { display: flex; align-items: center; gap: 6px; padding: 6px 8px;
  border: 1px solid #e4e7ed; border-radius: 4px; }
.tab-item.active { border-color: #5470c6; background: #ecf5ff; }
.tab-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  cursor: pointer; font-size: 13px; }
.tab-btns { display: flex; gap: 2px; }
.tab-btns button { font-size: 12px; padding: 0 4px; }
.content { display: flex; flex-direction: column; gap: 16px; min-width: 0; }
.card { border: 1px solid #e4e7ed; border-radius: 6px; padding: 12px; }
.card h3 { margin: 0 0 6px; font-size: 14px; }
.exp-section { margin-bottom: 10px; }
.metric-row { display: inline-flex; align-items: center; gap: 4px;
  margin-right: 14px; font-size: 13px; }
.metric-row input[type="color"] { width: 26px; height: 22px; padding: 0; border: none; }
.chart { width: 100%; height: 420px; border: 1px solid #e4e7ed; border-radius: 6px; }
.placeholder { padding: 60px; text-align: center; color: #909399; }
```

- [ ] **Step 3: 重写 main.js**

```js
'use strict';
(function (global) {
  const parser = global.TPAVisualizer.parser;
  const transforms = global.TPAVisualizer.transforms;
  const state = { tabs: [], activeTabId: null };
  let nextTabSeq = 1;
  let lineChart = null;
  let barChart = null;
  const $ = (id) => document.getElementById(id);

  function showMessage(text, isError) {
    $('message').textContent = text;
    $('message').className = isError ? 'message error' : 'message';
  }

  function readFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(new Error(`读取 ${file.name} 失败`));
      reader.readAsText(file, 'utf-8');
    });
  }

  function addTab(name) {
    const tab = {
      id: `tab-${Date.now()}-${nextTabSeq}`,
      name: name || `实验 ${nextTabSeq}`,
      checked: true,
      epochMetrics: null,
      comparison: null,
      metricOptions: {},
    };
    nextTabSeq += 1;
    state.tabs.push(tab);
    state.activeTabId = tab.id;
    return tab;
  }

  function activeTab() {
    return state.tabs.find((t) => t.id === state.activeTabId) || null;
  }

  function tabHasData(tab) {
    return !!(tab.epochMetrics || tab.comparison);
  }

  function buildMetricOptions(tab) {
    const names = [];
    if (tab.epochMetrics) {
      for (const m of parser.listMetrics(tab.epochMetrics)) {
        if (!names.includes(m)) names.push(m);
      }
    }
    if (tab.comparison) {
      for (const it of transforms.buildComparisonItems(tab.comparison)) {
        if (!names.includes(it.name)) names.push(it.name);
      }
    }
    const options = {};
    names.forEach((name, i) => {
      options[name] = { selected: true, color: transforms.colorFor(i) };
    });
    return options;
  }

  async function importFilesInto(tab, files) {
    const errors = [];
    for (const file of files) {
      try {
        const text = await readFile(file);
        if (file.name === 'history.json') {
          tab.epochMetrics = parser.extractEpochMetrics(text);
        } else if (file.name.endsWith('_comparison.json')) {
          tab.comparison = parser.parseComparison(text);
        } else {
          errors.push(`跳过未知文件: ${file.name}`);
        }
      } catch (e) {
        errors.push(`${file.name}: ${e.message}`);
      }
    }
    tab.metricOptions = buildMetricOptions(tab);
    if (errors.length) showMessage(errors.join('；'), true);
  }

  async function handleDirectoryImport(fileList) {
    const files = [...fileList];
    if (!files.length) return;
    const info = parser.parseDirectoryPath(files[0].webkitRelativePath || files[0].name);
    let tab = activeTab();
    if (!tab || tabHasData(tab)) tab = addTab(`实验 ${nextTabSeq}`);
    const autoName = parser.buildAutoName(info);
    if (autoName && !tabHasData(tab)) tab.name = autoName;
    await importFilesInto(tab, files);
    renderAll();
  }

  async function handleManualImport(fileList) {
    const files = [...fileList];
    if (!files.length) return;
    let tab = activeTab();
    if (!tab || tabHasData(tab)) {
      tab = addTab(prompt('选项卡名称', `实验 ${nextTabSeq}`) || `实验 ${nextTabSeq}`);
    }
    await importFilesInto(tab, files);
    renderAll();
  }

  function makeBtn(text, title, onClick) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = text;
    b.title = title;
    b.addEventListener('click', onClick);
    return b;
  }

  function renderTabs() {
    const ul = $('tab-list');
    ul.innerHTML = '';
    for (const tab of state.tabs) {
      const li = document.createElement('li');
      li.className = 'tab-item' + (tab.id === state.activeTabId ? ' active' : '');
      const check = document.createElement('input');
      check.type = 'checkbox';
      check.checked = tab.checked;
      check.title = '是否参与渲染';
      check.addEventListener('change', () => {
        tab.checked = check.checked;
        renderAll();
      });
      const nameSpan = document.createElement('span');
      nameSpan.className = 'tab-name';
      nameSpan.textContent = tab.name;
      nameSpan.title = '点击设为当前选项卡';
      nameSpan.addEventListener('click', () => {
        state.activeTabId = tab.id;
        renderAll();
      });
      const btns = document.createElement('span');
      btns.className = 'tab-btns';
      btns.appendChild(makeBtn('✎', '改名', () => {
        const n = prompt('修改名称', tab.name);
        if (n && n.trim()) {
          tab.name = n.trim();
          renderAll();
        }
      }));
      btns.appendChild(makeBtn('↺H', '重置 history', () => {
        tab.epochMetrics = null;
        tab.metricOptions = buildMetricOptions(tab);
        renderAll();
      }));
      btns.appendChild(makeBtn('↺C', '重置 comparison', () => {
        tab.comparison = null;
        tab.metricOptions = buildMetricOptions(tab);
        renderAll();
      }));
      btns.appendChild(makeBtn('×', '删除选项卡', () => {
        if (confirm(`删除选项卡「${tab.name}」？`)) {
          state.tabs = state.tabs.filter((t) => t.id !== tab.id);
          if (state.activeTabId === tab.id) {
            state.activeTabId = state.tabs.length ? state.tabs[0].id : null;
          }
          renderAll();
        }
      }));
      li.append(check, nameSpan, btns);
      ul.appendChild(li);
    }
  }

  function renderCard() {
    const card = $('metric-card');
    card.innerHTML = '';
    const tabs = state.tabs.filter((t) => t.checked && tabHasData(t));
    if (!tabs.length) {
      card.innerHTML = '<div class="placeholder">勾选左侧选项卡以显示指标选项</div>';
      return;
    }
    for (const tab of tabs) {
      const section = document.createElement('div');
      section.className = 'exp-section';
      const h = document.createElement('h3');
      h.textContent = tab.name;
      section.appendChild(h);
      for (const [name, opt] of Object.entries(tab.metricOptions)) {
        const row = document.createElement('label');
        row.className = 'metric-row';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = opt.selected;
        cb.addEventListener('change', () => {
          opt.selected = cb.checked;
          renderCharts();
        });
        const color = document.createElement('input');
        color.type = 'color';
        color.value = opt.color;
        color.addEventListener('input', () => {
          opt.color = color.value;
          renderCharts();
        });
        const span = document.createElement('span');
        span.textContent = name;
        row.append(cb, color, span);
        section.appendChild(row);
      }
      card.appendChild(section);
    }
  }

  function renderLineChart(tabs) {
    const withHistory = tabs.filter((t) => t.epochMetrics);
    if (!withHistory.length) {
      $('chart-line').innerHTML = '<div class="placeholder">勾选含 history 数据的选项卡</div>';
      return;
    }
    lineChart = lineChart || echarts.init($('chart-line'));
    const { xAxis, series } = transforms.buildMultiLineSeries(withHistory);
    lineChart.setOption({
      title: { text: '每轮指标折线图' },
      tooltip: { trigger: 'axis' },
      legend: { type: 'scroll', data: series.map((s) => s.name) },
      grid: { left: 60, right: 30, bottom: 60 },
      xAxis: { type: 'category', data: xAxis, name: 'epoch' },
      yAxis: { type: 'value' },
      dataZoom: [{ type: 'inside' }, { type: 'slider' }],
      series,
    }, true);
  }

  function renderBarChart(tabs) {
    const withCmp = tabs.filter((t) => t.comparison);
    if (!withCmp.length) {
      $('chart-bar').innerHTML = '<div class="placeholder">勾选含 comparison 数据的选项卡</div>';
      return;
    }
    barChart = barChart || echarts.init($('chart-bar'));
    const { xAxis, series } = transforms.buildMultiBarSeries(withCmp);
    barChart.setOption({
      title: { text: 'Clean vs Poisoned 对比' },
      tooltip: { trigger: 'axis' },
      legend: { type: 'scroll', data: series.map((s) => s.name) },
      grid: { left: 60, right: 30, bottom: 60 },
      xAxis: { type: 'category', data: xAxis, axisLabel: { interval: 0 } },
      yAxis: { type: 'value' },
      series,
    }, true);
  }

  function renderCharts() {
    const tabs = state.tabs.filter((t) => t.checked && tabHasData(t));
    renderLineChart(tabs);
    renderBarChart(tabs);
  }

  function renderAll() {
    renderTabs();
    renderCard();
    renderCharts();
  }

  function exportJson() {
    const tab = activeTab();
    if (!tab || !tab.epochMetrics) {
      showMessage('当前选项卡没有 history 数据', true);
      return;
    }
    const blob = new Blob([JSON.stringify(tab.epochMetrics, null, 2)],
      { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${tab.name}-epoch_metrics.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function initApp() {
    $('new-tab').addEventListener('click', () => {
      addTab(prompt('选项卡名称', `实验 ${nextTabSeq}`) || `实验 ${nextTabSeq}`);
      renderAll();
    });
    $('btn-dir').addEventListener('click', () => $('dir-input').click());
    $('dir-input').addEventListener('change', (e) => handleDirectoryImport(e.target.files));
    $('btn-import').addEventListener('click', () => $('import-input').click());
    $('import-input').addEventListener('change', (e) => handleManualImport(e.target.files));
    $('export-json').addEventListener('click', exportJson);
    const drop = $('drop-zone');
    ['dragenter', 'dragover'].forEach((ev) =>
      drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('active'); }));
    ['dragleave', 'drop'].forEach((ev) =>
      drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('active'); }));
    drop.addEventListener('drop', (e) => handleManualImport(e.dataTransfer.files));
    showMessage('请先新建选项卡，或直接导入实验目录/文件');
  }

  global.TPAVisualizer = global.TPAVisualizer || {};
  global.TPAVisualizer.app = { initApp, handleDirectoryImport, handleManualImport, exportJson };
})(window);
```

- [ ] **Step 4: 语法与单测校验**

Run: `node --check TPA/visualization/js/main.js`；`node --test TPA/visualization/tests/`
Expected: 语法 OK；JS 全绿。

- [ ] **Step 5: 提交**

```bash
git -C G:\Idea add -- TPA/visualization/index.html TPA/visualization/styles.css TPA/visualization/js/main.js
git -C G:\Idea commit -m "feat(visualization): 多实验选项卡、卡牌指标显隐与颜色自定义"
```

---

### Task 9: README v2 + 全量回归

**Files:**
- Modify: `TPA/visualization/README.md`

- [ ] **Step 1: 更新 README**

补充：多实验选项卡用法（新建/导入目录自动命名/改名/重置/删除/勾选）、卡牌
指标显隐与颜色自定义、直方图 x 轴为实验、导入需新建/空选项卡的规则。

- [ ] **Step 2: 全量回归**

Run:
```bash
node --test TPA/visualization/tests/
cd TPA && G:\Idea\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```
Expected: JS 全绿；Python 全量通过。

- [ ] **Step 3: 提交**

```bash
git -C G:\Idea add -- TPA/visualization/README.md
git -C G:\Idea commit -m "docs(visualization): 迭代 2 使用文档与验证清单更新"
```

---

# 迭代 3（卡内导入 + 顶刊配色 + 组件化）任务

对应 spec 第 13 节。基线：迭代 2 已提交于 `feat/attack-viz-demo`。

### Task 10: transforms.js 顶刊色板

**Files:**
- Modify: `TPA/visualization/js/transforms.js`（替换 `PALETTE`）
- Test: `TPA/visualization/tests/transforms.test.js`（追加用例）

- [ ] **Step 1: 追加失败测试**

```js
test('PALETTE 为顶刊 10 色且无重复', () => {
  const { PALETTE } = require('../js/transforms.js');
  assert.strictEqual(PALETTE.length, 10);
  assert.strictEqual(new Set(PALETTE).size, 10);
});
```

- [ ] **Step 2: 运行确认失败**

Run: `node --test TPA/visualization/tests/transforms.test.js`
Expected: FAIL（当前 9 色）。

- [ ] **Step 3: 替换 PALETTE**

```js
const PALETTE = [
  '#3B4992', '#EE0000', '#008B45', '#631879', '#008280',
  '#BB0021', '#5F559B', '#A20056', '#808180', '#1B1919',
];
```

- [ ] **Step 4: 运行确认通过**

Run: `node --test TPA/visualization/tests/transforms.test.js`
Expected: PASS，6 个用例。

- [ ] **Step 5: 提交**

```bash
git -C G:\Idea add -- TPA/visualization/js/transforms.js TPA/visualization/tests/transforms.test.js
git -C G:\Idea commit -m "feat(visualization): 系列色板替换为 Nature 顶刊 10 色（含 Node 单测）"
```

---

### Task 11: 页面重构（实验卡 + 卡内导入 + Modal + 图表布局）

**Files:**
- Modify: `TPA/visualization/index.html`
- Modify: `TPA/visualization/styles.css`
- Modify: `TPA/visualization/js/main.js`

**Interfaces:**
- Consumes: Task 2/6 parser、Task 7 transforms（含新色板）。
- Produces: `window.TPAVisualizer.app.initApp()`。

- [ ] **Step 1: 重写 index.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>TPA 攻击实验可视化</title>
  <link rel="stylesheet" href="styles.css">
  <script src="lib/echarts.min.js"></script>
</head>
<body>
  <header>
    <div class="brand">
      <h1>TPA 攻击实验可视化</h1>
      <span class="hint">实验数据对比 · 顶刊配色</span>
    </div>
    <div class="toolbar">
      <button id="add-card">＋ 添加实验卡</button>
      <button id="export-json">导出当前实验卡标准 JSON</button>
    </div>
    <div id="message" class="message"></div>
  </header>
  <main>
    <section id="card-list" class="card-list"></section>
    <section class="charts">
      <div class="chart-panel"><div id="chart-line" class="chart"></div></div>
      <div class="chart-panel"><div id="chart-bar" class="chart"></div></div>
    </section>
  </main>
  <div id="modal-overlay" class="modal-overlay hidden">
    <div class="modal">
      <h3 id="modal-title"></h3>
      <div id="modal-body"></div>
      <div class="modal-actions">
        <button id="modal-cancel" class="btn-ghost">取消</button>
        <button id="modal-ok" class="btn-primary">确定</button>
      </div>
    </div>
  </div>
  <script src="js/parser.js"></script>
  <script src="js/transforms.js"></script>
  <script src="js/main.js"></script>
  <script>window.TPAVisualizer.app.initApp();</script>
</body>
</html>
```

- [ ] **Step 2: 重写 styles.css**

```css
* { box-sizing: border-box; }
body { margin: 0; background: #fff; color: #222;
  font-family: -apple-system, "Segoe UI", "Microsoft YaHei", Arial, sans-serif; }
header { padding: 18px 28px; border-bottom: 1px solid #e5e7eb;
  display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; }
.brand h1 { margin: 0; font-size: 18px; font-weight: 600; letter-spacing: .5px; }
.brand .hint { color: #9ca3af; font-size: 12px; margin-left: 10px; }
.toolbar { display: flex; gap: 10px; }
button { font-size: 13px; padding: 7px 14px; border-radius: 6px; cursor: pointer;
  background: #fff; border: 1px solid #d1d5db; color: #374151; }
button:hover { border-color: #3B4992; color: #3B4992; }
.btn-primary { background: #3B4992; border-color: #3B4992; color: #fff; }
.btn-primary:hover { background: #2f3b78; color: #fff; }
.btn-danger:hover { border-color: #EE0000; color: #EE0000; }
.message { margin-top: 8px; font-size: 13px; color: #b45309; }
.message.error { color: #dc2626; }
main { max-width: 1280px; margin: 0 auto; padding: 20px 28px; }
.card-list { display: flex; flex-direction: column; gap: 12px; margin-bottom: 20px; }
.exp-card { border: 1px solid #e5e7eb; border-radius: 8px;
  box-shadow: 0 1px 2px rgba(0,0,0,.04); }
.exp-card.checked { border-left: 3px solid #3B4992; }
.exp-header { display: flex; align-items: center; gap: 12px;
  padding: 10px 14px; border-bottom: 1px solid #f3f4f6; flex-wrap: wrap; }
.exp-name { font-weight: 600; font-size: 14px; }
.exp-status { font-size: 12px; color: #6b7280; }
.exp-actions { margin-left: auto; display: flex; gap: 6px; }
.exp-actions button { padding: 3px 8px; font-size: 12px; }
.exp-body { padding: 12px 14px; }
.import-area { border: 1px dashed #d1d5db; border-radius: 6px;
  padding: 12px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
  background: #fafafa; font-size: 13px; }
.import-area label { display: inline-flex; align-items: center; gap: 6px; }
.metric-rows { display: flex; flex-wrap: wrap; gap: 4px 18px; }
.metric-row { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; }
.metric-row input[type="color"] { width: 26px; height: 22px; padding: 0; border: none; }
.charts { display: flex; flex-direction: column; gap: 20px; }
.chart-panel { border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px; }
.chart { width: 100%; height: 440px; }
.placeholder { padding: 60px; text-align: center; color: #9ca3af; }
.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,.35);
  display: flex; align-items: center; justify-content: center; z-index: 100; }
.modal-overlay.hidden { display: none; }
.modal { background: #fff; border-radius: 10px; width: 380px; max-width: 90vw;
  padding: 20px; box-shadow: 0 10px 30px rgba(0,0,0,.18); }
.modal h3 { margin: 0 0 14px; font-size: 16px; }
.modal input[type="text"] { width: 100%; padding: 8px 10px; font-size: 14px;
  border: 1px solid #d1d5db; border-radius: 6px; margin-bottom: 12px; }
.modal .sep { text-align: center; color: #9ca3af; font-size: 12px; margin: 6px 0 10px; }
.modal-actions { display: flex; justify-content: flex-end; gap: 10px; }
```

- [ ] **Step 3: 重写 main.js**

```js
'use strict';
(function (global) {
  const parser = global.TPAVisualizer.parser;
  const transforms = global.TPAVisualizer.transforms;
  const state = { cards: [], activeCardId: null };
  let nextCardSeq = 1;
  let lineChart = null;
  let barChart = null;
  const $ = (id) => document.getElementById(id);

  function showMessage(text, isError) {
    $('message').textContent = text;
    $('message').className = isError ? 'message error' : 'message';
  }

  function readFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(new Error(`读取 ${file.name} 失败`));
      reader.readAsText(file, 'utf-8');
    });
  }

  function openModal(title, opts) {
    const o = opts || {};
    const overlay = $('modal-overlay');
    $('modal-title').textContent = title;
    const body = $('modal-body');
    body.innerHTML = '';
    let inputEl = null;
    if (o.input) {
      inputEl = document.createElement('input');
      inputEl.type = 'text';
      inputEl.value = o.inputValue || '';
      body.appendChild(inputEl);
    }
    if (o.extra) {
      const sep = document.createElement('div');
      sep.className = 'sep';
      sep.textContent = '或';
      body.appendChild(sep);
      const extraBtn = document.createElement('button');
      extraBtn.textContent = o.extraLabel || '导入实验路径';
      extraBtn.addEventListener('click', () => o.onExtra && o.onExtra());
      body.appendChild(extraBtn);
    }
    const ok = $('modal-ok');
    ok.textContent = o.okText || '确定';
    const cancel = $('modal-cancel');
    overlay.classList.remove('hidden');

    function close() {
      overlay.classList.add('hidden');
      ok.removeEventListener('click', confirmModal);
      cancel.removeEventListener('click', close);
      if (inputEl) inputEl.removeEventListener('keydown', onKey);
    }
    function onKey(e) {
      if (e.key === 'Enter') confirmModal();
    }
    function confirmModal() {
      const value = inputEl ? inputEl.value.trim() : null;
      close();
      o.onOk && o.onOk(value);
    }
    ok.addEventListener('click', confirmModal);
    cancel.addEventListener('click', close);
    if (inputEl) {
      inputEl.addEventListener('keydown', onKey);
      inputEl.focus();
    }
  }

  function addCard(name) {
    const card = {
      id: `card-${Date.now()}-${nextCardSeq}`,
      name: name || `实验 ${nextCardSeq}`,
      checked: true,
      epochMetrics: null,
      comparison: null,
      metricOptions: {},
    };
    nextCardSeq += 1;
    state.cards.push(card);
    state.activeCardId = card.id;
    return card;
  }

  function activeCard() {
    return state.cards.find((c) => c.id === state.activeCardId) || null;
  }

  function cardHasData(card) {
    return !!(card.epochMetrics || card.comparison);
  }

  function isDefaultName(card) {
    return /^实验 \d+$/.test(card.name);
  }

  function buildMetricOptions(card) {
    const names = [];
    if (card.epochMetrics) {
      for (const m of parser.listMetrics(card.epochMetrics)) {
        if (!names.includes(m)) names.push(m);
      }
    }
    if (card.comparison) {
      for (const it of transforms.buildComparisonItems(card.comparison)) {
        if (!names.includes(it.name)) names.push(it.name);
      }
    }
    const options = {};
    names.forEach((name, i) => {
      options[name] = { selected: true, color: transforms.colorFor(i) };
    });
    return options;
  }

  function applyFileToCard(card, file, text) {
    if (file.name === 'history.json') {
      card.epochMetrics = parser.extractEpochMetrics(text);
    } else if (file.name.endsWith('_comparison.json')) {
      card.comparison = parser.parseComparison(text);
    } else {
      throw new Error(`跳过未知文件: ${file.name}`);
    }
  }

  async function importDirIntoCard(card, fileList) {
    const files = [...fileList];
    if (!files.length) return;
    const info = parser.parseDirectoryPath(files[0].webkitRelativePath || files[0].name);
    const autoName = parser.buildAutoName(info);
    if (autoName && isDefaultName(card)) card.name = autoName;
    const errors = [];
    for (const file of files) {
      try {
        applyFileToCard(card, file, await readFile(file));
      } catch (e) {
        errors.push(`${file.name}: ${e.message}`);
      }
    }
    card.metricOptions = buildMetricOptions(card);
    if (errors.length) showMessage(errors.join('；'), true);
    renderAll();
  }

  async function importFilesIntoCard(card, fileList) {
    const files = [...fileList];
    if (!files.length) return;
    const errors = [];
    for (const file of files) {
      try {
        applyFileToCard(card, file, await readFile(file));
      } catch (e) {
        errors.push(`${file.name}: ${e.message}`);
      }
    }
    card.metricOptions = buildMetricOptions(card);
    if (errors.length) showMessage(errors.join('；'), true);
    renderAll();
  }

  function renderCards() {
    const list = $('card-list');
    list.innerHTML = '';
    for (const card of state.cards) {
      list.appendChild(buildCardEl(card));
    }
  }

  function buildCardEl(card) {
    const el = document.createElement('div');
    el.className = 'exp-card' + (card.checked ? ' checked' : '');
    el.dataset.cardId = card.id;

    const header = document.createElement('div');
    header.className = 'exp-header';
    const check = document.createElement('input');
    check.type = 'checkbox';
    check.checked = card.checked;
    check.title = '是否参与渲染';
    check.addEventListener('change', () => {
      card.checked = check.checked;
      renderAll();
    });
    const name = document.createElement('span');
    name.className = 'exp-name';
    name.textContent = card.name;
    name.title = '设为当前实验卡（导出目标）';
    name.addEventListener('click', () => {
      state.activeCardId = card.id;
      renderAll();
    });
    const status = document.createElement('span');
    status.className = 'exp-status';
    status.textContent = `history ${card.epochMetrics ? '✓' : '—'} · comparison ${card.comparison ? '✓' : '—'}`;
    const actions = document.createElement('span');
    actions.className = 'exp-actions';
    actions.appendChild(btn('改名', () => {
      openModal('修改实验卡名称', {
        input: true, inputValue: card.name, onOk: (v) => {
          if (v) { card.name = v; renderAll(); }
        },
      });
    }));
    actions.appendChild(btn('重置 H', () => {
      card.epochMetrics = null;
      card.metricOptions = buildMetricOptions(card);
      renderAll();
    }));
    actions.appendChild(btn('重置 C', () => {
      card.comparison = null;
      card.metricOptions = buildMetricOptions(card);
      renderAll();
    }));
    actions.appendChild(btn('删除', () => {
      openModal(`删除实验卡「${card.name}」？`, {
        okText: '删除', onOk: () => {
          state.cards = state.cards.filter((c) => c.id !== card.id);
          if (state.activeCardId === card.id) {
            state.activeCardId = state.cards.length ? state.cards[0].id : null;
          }
          renderAll();
        },
      });
    }, true));
    header.append(check, name, status, actions);
    el.appendChild(header);

    const body = document.createElement('div');
    body.className = 'exp-body';
    if (!cardHasData(card)) {
      body.appendChild(buildImportArea(card));
    } else {
      body.appendChild(buildMetricRows(card));
    }
    el.appendChild(body);
    return el;
  }

  function buildImportArea(card) {
    const area = document.createElement('div');
    area.className = 'import-area';
    const dirBtn = btn('导入实验路径', () => {
      const input = document.createElement('input');
      input.type = 'file';
      input.webkitdirectory = true;
      input.multiple = true;
      input.addEventListener('change', (e) => importDirIntoCard(card, e.target.files));
      input.click();
    });
    const histLabel = document.createElement('label');
    histLabel.append('history.json ');
    const histInput = document.createElement('input');
    histInput.type = 'file';
    histInput.accept = '.json';
    histInput.addEventListener('change', (e) => importFilesIntoCard(card, e.target.files));
    histLabel.appendChild(histInput);
    const cmpLabel = document.createElement('label');
    cmpLabel.append('comparison ');
    const cmpInput = document.createElement('input');
    cmpInput.type = 'file';
    cmpInput.accept = '.json';
    cmpInput.addEventListener('change', (e) => importFilesIntoCard(card, e.target.files));
    cmpLabel.appendChild(cmpInput);
    area.append(dirBtn, histLabel, cmpLabel);
    return area;
  }

  function buildMetricRows(card) {
    const wrap = document.createElement('div');
    wrap.className = 'metric-rows';
    for (const [name, opt] of Object.entries(card.metricOptions)) {
      const row = document.createElement('label');
      row.className = 'metric-row';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = opt.selected;
      cb.addEventListener('change', () => {
        opt.selected = cb.checked;
        renderCharts();
      });
      const color = document.createElement('input');
      color.type = 'color';
      color.value = opt.color;
      color.addEventListener('input', () => {
        opt.color = color.value;
        renderCharts();
      });
      const span = document.createElement('span');
      span.textContent = name;
      row.append(cb, color, span);
      wrap.appendChild(row);
    }
    return wrap;
  }

  function btn(text, onClick, danger) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = text;
    if (danger) b.className = 'btn-danger';
    b.addEventListener('click', onClick);
    return b;
  }

  function chartBase(title) {
    return {
      title: { text: title, top: 10, left: 'center', textStyle: { fontSize: 14 } },
      tooltip: { trigger: 'axis' },
      legend: { type: 'scroll', top: 44, left: 'center' },
      grid: { top: 96, left: 70, right: 30, bottom: 60 },
    };
  }

  function renderLineChart(cards) {
    const withHistory = cards.filter((c) => c.epochMetrics);
    if (!withHistory.length) {
      $('chart-line').innerHTML = '<div class="placeholder">勾选含 history 数据的实验卡</div>';
      return;
    }
    lineChart = lineChart || echarts.init($('chart-line'));
    const { xAxis, series } = transforms.buildMultiLineSeries(withHistory);
    const option = chartBase('每轮指标折线图');
    option.xAxis = { type: 'category', data: xAxis, name: 'epoch' };
    option.yAxis = { type: 'value' };
    option.dataZoom = [{ type: 'inside' }, { type: 'slider' }];
    option.series = series;
    option.legend.data = series.map((s) => s.name);
    lineChart.setOption(option, true);
  }

  function renderBarChart(cards) {
    const withCmp = cards.filter((c) => c.comparison);
    if (!withCmp.length) {
      $('chart-bar').innerHTML = '<div class="placeholder">勾选含 comparison 数据的实验卡</div>';
      return;
    }
    barChart = barChart || echarts.init($('chart-bar'));
    const { xAxis, series } = transforms.buildMultiBarSeries(withCmp);
    const option = chartBase('Clean vs Poisoned 对比');
    option.xAxis = { type: 'category', data: xAxis, axisLabel: { interval: 0 } };
    option.yAxis = { type: 'value' };
    option.series = series;
    option.legend.data = series.map((s) => s.name);
    barChart.setOption(option, true);
  }

  function renderCharts() {
    const cards = state.cards.filter((c) => c.checked && cardHasData(c));
    renderLineChart(cards);
    renderBarChart(cards);
  }

  function renderAll() {
    renderCards();
    renderCharts();
  }

  function exportJson() {
    const card = activeCard();
    if (!card || !card.epochMetrics) {
      showMessage('当前实验卡没有 history 数据', true);
      return;
    }
    const blob = new Blob([JSON.stringify(card.epochMetrics, null, 2)],
      { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${card.name}-epoch_metrics.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function initApp() {
    $('add-card').addEventListener('click', () => {
      openModal('添加实验卡', {
        input: true,
        inputValue: `实验 ${nextCardSeq}`,
        extra: true,
        extraLabel: '导入实验路径自动命名',
        onExtra: () => {
          const input = document.createElement('input');
          input.type = 'file';
          input.webkitdirectory = true;
          input.multiple = true;
          input.addEventListener('change', (e) => {
            const files = [...e.target.files];
            if (!files.length) return;
            const info = parser.parseDirectoryPath(files[0].webkitRelativePath || files[0].name);
            const card = addCard(parser.buildAutoName(info) || `实验 ${nextCardSeq}`);
            importDirIntoCard(card, files);
          });
          input.click();
        },
        onOk: (v) => {
          addCard(v || `实验 ${nextCardSeq}`);
          renderAll();
        },
      });
    });
    $('export-json').addEventListener('click', exportJson);
    showMessage('点击「＋ 添加实验卡」开始');
  }

  global.TPAVisualizer = global.TPAVisualizer || {};
  global.TPAVisualizer.app = { initApp, addCard, exportJson };
})(window);
```

- [ ] **Step 4: 语法与单测校验**

Run: `node --check TPA/visualization/js/main.js`；`node --test TPA/visualization/tests/`
Expected: 语法 OK；JS 全绿。

- [ ] **Step 5: 提交**

```bash
git -C G:\Idea add -- TPA/visualization/index.html TPA/visualization/styles.css TPA/visualization/js/main.js
git -C G:\Idea commit -m "feat(visualization): 实验卡内导入、Modal 组件、顶刊配色与图表布局修复"
```

---

### Task 12: README v3 + 全量回归

**Files:**
- Modify: `TPA/visualization/README.md`

- [ ] **Step 1: 更新 README**

按 v3 说明：实验卡模型（卡内导入/添加数据/重置/改名/删除）、添加实验卡弹窗
（输入名称或导入路径自动命名）、顶刊配色、Modal 组件、图表布局（标题/图例
分离）。手动验证清单补充：标题与图例不重叠、导入只能从实验卡内发起。

- [ ] **Step 2: 全量回归**

Run:
```bash
node --test TPA/visualization/tests/
cd TPA && G:\Idea\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```
Expected: JS 全绿；Python 全量通过。

- [ ] **Step 3: 提交**

```bash
git -C G:\Idea add -- TPA/visualization/README.md
git -C G:\Idea commit -m "docs(visualization): 迭代 3 使用文档与验证清单更新"
```

---

# 迭代 4（选项分组 + 左侧列表 + Bug 修复）任务

对应 spec 第 14 节。基线：迭代 3 已提交于 `feat/attack-viz-demo`。

### Task 13: transforms.js 选项分组（lineOptions / barOptions）

**Files:**
- Modify: `TPA/visualization/js/transforms.js`
- Test: `TPA/visualization/tests/transforms.test.js`

**Interfaces:**
- Consumes: 无变化（输入实验对象结构改为 `lineOptions` / `barOptions`）。
- Produces: `buildMultiLineSeries` 读 `exp.lineOptions`；
  `buildMultiBarSeries` 读 `exp.barOptions`。

- [ ] **Step 1: 更新测试（先失败）**

把 `transforms.test.js` 中多实验用例的 `metricOptions` 分别改为
`lineOptions`（折线用例）与 `barOptions`（直方用例），其余断言不变。

- [ ] **Step 2: 运行确认失败**

Run: `node --test TPA/visualization/tests/transforms.test.js`
Expected: FAIL（实现仍读 `metricOptions`，series 为空/断言不符）。

- [ ] **Step 3: 实现**

`buildMultiLineSeries` 内：
```js
for (const [name, opt] of Object.entries(exp.lineOptions || {})) {
  if (opt && opt.selected && !metricNames.includes(name)) metricNames.push(name);
}
...
const opt = (exp.lineOptions || {})[metric];
```

`buildMultiBarSeries` 内：
```js
for (const [name, opt] of Object.entries(exp.barOptions || {})) {
  if (opt && opt.selected && barNames.has(name) && !metricNames.includes(name)) {
    metricNames.push(name);
  }
}
...
const opt = (exp.barOptions || {})[metric];
```

- [ ] **Step 4: 运行确认通过**

Run: `node --test TPA/visualization/tests/transforms.test.js`
Expected: PASS，6 个用例。

- [ ] **Step 5: 提交**

```bash
git -C G:\Idea add -- TPA/visualization/js/transforms.js TPA/visualization/tests/transforms.test.js
git -C G:\Idea commit -m "feat(visualization): 折线/直方选项分组 lineOptions 与 barOptions（含 Node 单测）"
```

---

### Task 14: 页面重构（左侧列表 + 选中卡面板 + 分组 + Bug 修复）

**Files:**
- Modify: `TPA/visualization/index.html`
- Modify: `TPA/visualization/styles.css`
- Modify: `TPA/visualization/js/main.js`

**Interfaces:**
- Consumes: Task 13 的 `lineOptions` / `barOptions`。
- Produces: `window.TPAVisualizer.app.initApp()`。

- [ ] **Step 1: 重写 index.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>TPA 攻击实验可视化</title>
  <link rel="stylesheet" href="styles.css">
  <script src="lib/echarts.min.js"></script>
</head>
<body>
  <header>
    <div class="brand"><h1>TPA 攻击实验可视化</h1></div>
    <div class="toolbar">
      <button id="add-card">＋ 添加实验卡</button>
      <button id="export-json">导出当前实验卡标准 JSON</button>
    </div>
    <div id="message" class="message"></div>
  </header>
  <main class="layout">
    <aside class="sidebar">
      <h2>实验列表</h2>
      <ul id="card-list"></ul>
    </aside>
    <section class="content">
      <div id="card-panel"></div>
      <section class="charts">
        <div class="chart-panel"><div id="chart-line" class="chart"></div></div>
        <div class="chart-panel"><div id="chart-bar" class="chart"></div></div>
      </section>
    </section>
  </main>
  <div id="modal-overlay" class="modal-overlay hidden">
    <div class="modal">
      <h3 id="modal-title"></h3>
      <div id="modal-body"></div>
      <div class="modal-actions">
        <button id="modal-cancel" class="btn-ghost">取消</button>
        <button id="modal-ok" class="btn-primary">确定</button>
      </div>
    </div>
  </div>
  <script src="js/parser.js"></script>
  <script src="js/transforms.js"></script>
  <script src="js/main.js"></script>
  <script>window.TPAVisualizer.app.initApp();</script>
</body>
</html>
```

- [ ] **Step 2: 重写 styles.css**

在迭代 3 基础上调整：
- 删除 `.brand .hint` 相关样式；
- 增加 `.layout { display:grid; grid-template-columns: 260px 1fr; gap:18px; }`、
  `.sidebar`（细边框圆角列表）、`#card-list`（竖向简要项：
  `[checkbox] 名称 + H✓/C✓ 状态`，选中项高亮）；
- `.content` 纵向布局；`#card-panel` 卡面板（卡头 + 分组）；
- `.metric-group h4` 分组小标题；`.metric-group` 间距。

- [ ] **Step 3: 重写 main.js**

要点：
- 卡数据：`{id, name, checked, epochMetrics, comparison, lineOptions, barOptions}`；
  `buildOptions(card)` 分别由 `listMetrics` / `buildComparisonItems` 生成两组，
  同名字段保留用户已设的 selected/color。
- 图表生命周期：
  ```js
  function showPlaceholder(id, text) {
    const dom = document.getElementById(id);
    const inst = echarts.getInstanceByDom(dom);
    if (inst) inst.dispose();
    dom.innerHTML = `<div class="placeholder">${text}</div>`;
  }
  function chartInstance(id) {
    const dom = document.getElementById(id);
    return echarts.getInstanceByDom(dom) || echarts.init(dom);
  }
  ```
  折线/直方渲染均用 `chartInstance`，无数据用 `showPlaceholder`。
- 左侧列表 `renderCards()`：每项 checkbox + 名称 + `H✓/C✓`，点击选中；
  主面板 `renderPanel()` 只渲染选中卡：卡头（改名/重置 H/重置 C/删除）+
  始终可见导入区 + history/comparison 两个指标分组（勾选/改色只调
  `renderCharts()`）。
- Modal `openModal` 的 `onExtra` 回调注入 `close`：路径导入完成后关闭弹窗。
- 移除 header hint；导出仍为当前选中卡。

- [ ] **Step 4: 语法与单测校验**

Run: `node --check TPA/visualization/js/main.js`；`node --test TPA/visualization/tests/`
Expected: 语法 OK；JS 全绿。

- [ ] **Step 5: 提交**

```bash
git -C G:\Idea add -- TPA/visualization/index.html TPA/visualization/styles.css TPA/visualization/js/main.js
git -C G:\Idea commit -m "feat(visualization): 左侧实验列表、history/comparison 分组与图表实例生命周期修复"
```

---

### Task 15: README v4 + 全量回归

**Files:**
- Modify: `TPA/visualization/README.md`

- [ ] **Step 1: 更新 README**

按 v4 说明：左侧简要列表、选中卡面板、history/comparison 分组选项各管各图、
重置后可重新导入、路径新增自动关闭弹窗。

- [ ] **Step 2: 全量回归**

Run:
```bash
node --test TPA/visualization/tests/
cd TPA && G:\Idea\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```
Expected: JS 全绿；Python 全量通过。

- [ ] **Step 3: 提交**

```bash
git -C G:\Idea add -- TPA/visualization/README.md
git -C G:\Idea commit -m "docs(visualization): 迭代 4 使用文档与验证清单更新"
```

---

# 迭代 5（现代 UI 主题）任务

对应 spec 第 15 节。基线：迭代 4 已提交于 `feat/attack-viz-demo`。

### Task 16: 现代 UI 主题（styles.css 重写 + 状态徽章）

**Files:**
- Modify: `TPA/visualization/styles.css`（全量重写）
- Modify: `TPA/visualization/js/main.js`（状态渲染为药丸徽章）
- Test: 无新增（JS/Python 测试保持通过）

**Interfaces:**
- Consumes: 现有 `index.html` 结构类名（`.brand/.toolbar/.layout/.sidebar/
  #card-list/.content/#card-panel/.import-area/.metric-rows/.chart/.modal-*`）。
- Produces: 现代主题样式与 `badge(text, ok)` 徽章辅助函数。

- [ ] **Step 1: 重写 styles.css**

```css
:root {
  --bg: #f8fafc;
  --surface: #ffffff;
  --surface-soft: #f1f5f9;
  --border: #e2e8f0;
  --border-strong: #cbd5e1;
  --text: #0f172a;
  --text-secondary: #64748b;
  --text-muted: #94a3b8;
  --primary: #6366f1;
  --primary-hover: #4f46e5;
  --primary-soft: #eef2ff;
  --success-soft: #d1fae5;
  --danger: #ef4444;
  --radius: 12px;
  --radius-sm: 8px;
  --shadow-sm: 0 1px 3px rgba(15,23,42,.06), 0 1px 2px rgba(15,23,42,.04);
  --shadow-md: 0 6px 16px rgba(15,23,42,.10);
  --font: "Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, "Microsoft YaHei", sans-serif;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font-family: var(--font); font-size: 14px; line-height: 1.5; -webkit-font-smoothing: antialiased; }
button { font: inherit; }
header { position: sticky; top: 0; z-index: 50; padding: 14px 28px;
  background: rgba(255,255,255,.82); backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.brand h1 { margin: 0; font-size: 17px; font-weight: 700; letter-spacing: .2px; }
.toolbar { display: flex; gap: 8px; }
button { font-size: 13px; font-weight: 500; padding: 8px 14px; border-radius: var(--radius-sm);
  border: 1px solid var(--border-strong); background: var(--surface); color: var(--text);
  cursor: pointer; transition: background .15s, border-color .15s, color .15s, transform .1s, box-shadow .15s; }
button:hover { border-color: var(--primary); color: var(--primary); }
button:active { transform: translateY(1px); }
button:focus-visible { outline: 2px solid var(--primary); outline-offset: 2px; }
.btn-primary { background: linear-gradient(135deg, #6366f1, #8b5cf6); border: none; color: #fff;
  box-shadow: 0 2px 8px rgba(99,102,241,.35); }
.btn-primary:hover { background: linear-gradient(135deg, #4f46e5, #7c3aed); color: #fff; }
.btn-danger:hover { border-color: var(--danger); color: var(--danger); }
.message { width: 100%; margin-top: 6px; font-size: 13px; color: #b45309; }
.message.error { color: var(--danger); }
.layout { display: grid; grid-template-columns: 264px minmax(0,1fr); gap: 18px;
  max-width: 1400px; margin: 0 auto; padding: 20px 28px; align-items: start; }
.sidebar { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  box-shadow: var(--shadow-sm); padding: 14px; position: sticky; top: 76px; }
.sidebar h2 { margin: 0 0 12px; font-size: 13px; font-weight: 600; text-transform: uppercase;
  letter-spacing: .08em; color: var(--text-secondary); }
#add-card { width: 100%; margin-bottom: 12px; }
#card-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; }
.card-item { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border-radius: var(--radius-sm);
  border: 1px solid transparent; cursor: pointer; font-size: 13px; transition: background .15s, border-color .15s; }
.card-item:hover { background: var(--surface-soft); }
.card-item.active { background: var(--primary-soft); border-color: var(--primary); }
.card-item .ci-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.card-item input[type="checkbox"] { accent-color: var(--primary); }
.badge { display: inline-block; padding: 1px 7px; border-radius: 999px; font-size: 11px;
  background: var(--surface-soft); color: var(--text-secondary); white-space: nowrap; }
.badge.ok { background: var(--success-soft); color: #047857; }
.ci-status { display: flex; gap: 4px; }
.content { display: flex; flex-direction: column; gap: 18px; min-width: 0; }
#card-panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  box-shadow: var(--shadow-sm); overflow: hidden; }
.panel-header { display: flex; align-items: center; gap: 12px; padding: 12px 16px;
  border-bottom: 1px solid var(--border); flex-wrap: wrap; }
.panel-name { font-weight: 600; font-size: 15px; }
.panel-status { display: flex; gap: 6px; align-items: center; }
.panel-actions { margin-left: auto; display: flex; gap: 6px; }
.panel-actions button { padding: 5px 10px; font-size: 12px; }
.panel-body { padding: 14px 16px; }
.import-area { border: 1px dashed var(--border-strong); border-radius: var(--radius-sm);
  padding: 12px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
  background: var(--surface-soft); font-size: 13px; margin-bottom: 14px; }
.import-area label { display: inline-flex; align-items: center; gap: 6px; color: var(--text-secondary); }
.metric-group { margin-bottom: 14px; }
.metric-group:last-child { margin-bottom: 0; }
.metric-group h4 { margin: 0 0 8px; font-size: 12px; font-weight: 600; letter-spacing: .05em;
  text-transform: uppercase; color: var(--text-secondary); }
.metric-rows { display: flex; flex-wrap: wrap; gap: 6px 14px; }
.metric-row { display: inline-flex; align-items: center; gap: 6px; font-size: 13px;
  padding: 4px 8px; border: 1px solid var(--border); border-radius: 999px; background: var(--surface);
  transition: border-color .15s; cursor: pointer; }
.metric-row:hover { border-color: var(--primary); }
.metric-row input[type="checkbox"] { accent-color: var(--primary); }
.metric-row input[type="color"] { width: 24px; height: 22px; padding: 0; border: none;
  border-radius: 4px; background: transparent; cursor: pointer; }
.charts { display: flex; flex-direction: column; gap: 18px; }
.chart-panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  box-shadow: var(--shadow-sm); padding: 10px; }
.chart { width: 100%; height: 440px; }
.placeholder { padding: 40px; text-align: center; color: var(--text-muted); }
.modal-overlay { position: fixed; inset: 0; background: rgba(15,23,42,.45);
  backdrop-filter: blur(4px); -webkit-backdrop-filter: blur(4px);
  display: flex; align-items: center; justify-content: center; z-index: 100;
  animation: fadeIn .15s ease-out; }
.modal-overlay.hidden { display: none; }
.modal { background: var(--surface); border-radius: 14px; width: 400px; max-width: 92vw;
  padding: 22px; box-shadow: var(--shadow-md); animation: popIn .18s ease-out; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes popIn { from { opacity: 0; transform: scale(.96) translateY(6px); }
  to { opacity: 1; transform: none; } }
.modal h3 { margin: 0 0 14px; font-size: 16px; font-weight: 600; }
.modal input[type="text"] { width: 100%; padding: 9px 11px; font-size: 14px;
  border: 1px solid var(--border-strong); border-radius: var(--radius-sm); margin-bottom: 12px;
  transition: border-color .15s, box-shadow .15s; }
.modal input[type="text"]:focus { outline: none; border-color: var(--primary);
  box-shadow: 0 0 0 3px var(--primary-soft); }
.modal .sep { text-align: center; color: var(--text-muted); font-size: 12px; margin: 6px 0 10px; }
.modal-actions { display: flex; justify-content: flex-end; gap: 8px; }
.btn-ghost { color: var(--text-secondary); }
```

- [ ] **Step 2: main.js 状态徽章化**

新增辅助函数并替换两处状态渲染：

```js
function badge(text, ok) {
  const s = document.createElement('span');
  s.className = 'badge' + (ok ? ' ok' : '');
  s.textContent = text;
  return s;
}
```

- `renderCards()` 中 `ci-status`：`status.appendChild(badge('H✓', !!card.epochMetrics)); status.appendChild(badge('C✓', !!card.comparison));`
- `renderPanel()` 中 `panel-status`：
  `status.appendChild(badge('history ✓', !!card.epochMetrics)); status.appendChild(badge('comparison ✓', !!card.comparison));`
- 其余逻辑不动。

- [ ] **Step 3: 语法与单测校验**

Run: `node --check TPA/visualization/js/main.js`；`node --test TPA/visualization/tests/`
Expected: 语法 OK；JS 全绿。

- [ ] **Step 4: 提交**

```bash
git -C G:\Idea add -- TPA/visualization/styles.css TPA/visualization/js/main.js
git -C G:\Idea commit -m "feat(visualization): 现代 UI 主题（设计令牌+靛蓝主色+状态徽章）"
```
