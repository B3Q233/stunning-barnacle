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
