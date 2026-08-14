# 数据显示功能（HTML 可视化）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付一个纯前端 HTML 数据可视化工具，支持导入实验数据、自动识别模型/攻击/实验名称、多选对比折线/直方/对比图与展示编辑。

**Architecture:** 静态单页应用（无构建步骤、无后端）。`js/parser.js` 与 `js/transforms.js` 为纯函数模块（Node 单测），`js/app.js` 为薄 UI 装配层，ECharts 负责渲染，内置 `lib/echarts.min.js` 保证离线可用。

**Tech Stack:** 原生 HTML/CSS/JS（ES2020+）、ECharts 5（vendored）、Node 20 标准库 assert 单测。

## Global Constraints

- 不新增第三方 Python 依赖；Python 单测仍为 stdlib unittest。
- JS 测试只用 Node 标准库 `assert`，运行 `node --test TPA/visualization/tests/`。
- 文档与提交信息使用中文；Conventional Commits：`feat(visualization): 中文描述`。
- 每个任务结束运行相关测试；交付前 Python 全量单测 + JS 单测全绿。
- 只新增 `TPA/visualization/` 下文件与文档，不改动其他模块代码。
- 原始数据文件只读，不修改。
- JS 模块采用 UMD 包装：Node 下 `module.exports`，浏览器下挂
  `window.TPAVisualizer.{parser,transforms,app}`；下述代码块为模块内部内容，
  落盘时按此包装。

---

### Task 1: parser.js（路径识别 + 文件解析纯函数）

**Files:**
- Create: `TPA/visualization/js/parser.js`
- Create: `TPA/visualization/tests/test_parser.js`

**Interfaces:**
- Consumes: 无（纯函数，输入路径段数组与文件文本）。
- Produces:
  - `formatRunTag(runTag: string): string` — `2026-08-09-21-54` → `2026年08月09日21时54分`
  - `parseExperiment(segments: string[], files: {name, text}[]): Experiment`
  - `parseHistoryJson(text: string): Array<Record<string, number>>`
  - `parseBest(text: string): object | null`
  - `parseEvalCsv(text: string): Array<Record<string, number>>`
  - `parseConfigYaml(text: string): {dataset, model, attack, epochs, metrics}`
  - `parseComparisonJson(text: string): {model_utility, target_metrics}`
  - `buildExperimentId(kind, method, dataset, model, runTag): string`
  - `buildLabel(kind, method, dataset, model, runTag): string`

- [ ] **Step 1: 写失败测试**

```js
const test = require('node:test');
const assert = require('node:assert');
const {
  formatRunTag, parseExperiment, parseHistoryJson, parseEvalCsv,
  parseConfigYaml, parseComparisonJson, buildLabel,
} = require('../js/parser.js');

test('formatRunTag 转为中文日期时间', () => {
  assert.strictEqual(formatRunTag('2026-08-09-21-54'), '2026年08月09日21时54分');
});

test('识别攻击实验路径', () => {
  const exp = parseExperiment(
    ['attacks', 'pgd', 'outputs', 'ml100k', 'lightgcn', '2026-08-09-21-54'],
    [],
  );
  assert.strictEqual(exp.kind, 'attack');
  assert.strictEqual(exp.method, 'pgd');
  assert.strictEqual(exp.dataset, 'ml100k');
  assert.strictEqual(exp.model, 'lightgcn');
  assert.strictEqual(exp.runTag, '2026-08-09-21-54');
  assert.strictEqual(
    exp.label, 'attack-pgd-ml100k-lightgcn-2026年08月09日21时54分',
  );
});

test('识别模型训练路径并从 config 补全 dataset', () => {
  const exp = parseExperiment(
    ['models', 'lightgcn', 'outputs', '2026-08-09-23-01'],
    [{ name: 'config.yaml', text: 'dataset: ml100k\nmodel:\n  name: lightgcn\n' }],
  );
  assert.strictEqual(exp.kind, 'model');
  assert.strictEqual(exp.dataset, 'ml100k');
  assert.strictEqual(exp.runTag, '2026-08-09-23-01');
});

test('无法识别时按 custom 处理', () => {
  const exp = parseExperiment(['my_data'], [{ name: 'history.json', text: '[]' }]);
  assert.strictEqual(exp.kind, 'custom');
  assert.strictEqual(exp.label, '自定义-my_data');
});

test('解析 history.json', () => {
  const rows = parseHistoryJson(
    JSON.stringify({ history: [{ epoch: 1, train_loss: 0.5, val_loss: 0.4 }] }),
  );
  assert.deepStrictEqual(rows, [{ epoch: 1, train_loss: 0.5, val_loss: 0.4 }]);
});

test('解析 history.json 的 best 字段', () => {
  const best = parseBest(
    JSON.stringify({ best: { 'recall@10': { epoch: 5, value: 0.1 } } }),
  );
  assert.strictEqual(best['recall@10'].value, 0.1);
});

test('解析 eval_log.csv', () => {
  const rows = parseEvalCsv('epoch,recall@10,ndcg@10\n1,0.1,0.2\n2,0.11,0.21\n');
  assert.deepStrictEqual(rows, [
    { epoch: 1, 'recall@10': 0.1, 'ndcg@10': 0.2 },
    { epoch: 2, 'recall@10': 0.11, 'ndcg@10': 0.21 },
  ]);
});

test('解析 config.yaml 快照关键字段', () => {
  const cfg = parseConfigYaml(
    'dataset: ml100k\nmodel:\n  name: lightgcn\nattack:\n  name: pgd\ntraining:\n  epochs: 30\nevaluation:\n  metrics:\n  - recall@10: upper\n',
  );
  assert.strictEqual(cfg.dataset, 'ml100k');
  assert.strictEqual(cfg.attack, 'pgd');
  assert.deepStrictEqual(cfg.metrics, ['recall@10']);
});

test('解析 *_comparison.json', () => {
  const comp = parseComparisonJson(
    JSON.stringify({ model_utility: { clean: { 'recall@10': 0.1 } }, target_metrics: {} }),
  );
  assert.strictEqual(comp.model_utility.clean['recall@10'], 0.1);
});

test('坏 JSON 抛出可读错误', () => {
  assert.throws(() => parseHistoryJson('{oops'), /解析失败/);
});
```

- [ ] **Step 2: 运行确认失败**

Run: `node --test TPA/visualization/tests/test_parser.js`
Expected: FAIL，`Cannot find module '../js/parser.js'`。

- [ ] **Step 3: 实现 parser.js**

```js
'use strict';

const RUN_TAG_RE = /^(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$/;

function formatRunTag(runTag) {
  const m = RUN_TAG_RE.exec(runTag);
  if (!m) return runTag;
  return `${m[1]}年${m[2]}月${m[3]}日${m[4]}时${m[5]}分`;
}

function buildExperimentId(kind, method, dataset, model, runTag) {
  return [kind, method, dataset, model, runTag].filter(Boolean).join('-');
}

function buildLabel(kind, method, dataset, model, runTag) {
  const parts = [kind, method, dataset, model].filter(Boolean);
  if (runTag) parts.push(formatRunTag(runTag));
  return parts.join('-');
}

function parseHistoryJson(text) {
  try {
    const data = JSON.parse(text);
    return Array.isArray(data.history) ? data.history : [];
  } catch (e) {
    throw new Error(`解析失败: history.json 不是合法 JSON（${e.message}）`);
  }
}

function parseBest(text) {
  try {
    const data = JSON.parse(text);
    return data.best && typeof data.best === 'object' ? data.best : null;
  } catch (e) {
    return null;
  }
}

function parseEvalCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return [];
  const headers = lines[0].split(',').map((s) => s.trim());
  return lines.slice(1).map((line) => {
    const values = line.split(',').map((s) => s.trim());
    const row = {};
    headers.forEach((h, i) => {
      row[h] = values[i] === undefined ? null : Number(values[i]);
    });
    return row;
  });
}

function parseConfigYaml(text) {
  const cfg = { dataset: null, model: null, attack: null, epochs: null, metrics: [] };
  let section = null;
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.replace(/\s*#.*$/, '');
    if (!line.trim()) continue;
    const indent = line.match(/^\s*/)[0].length;
    const m = line.match(/^\s*([\w@.-]+):\s*(.*)$/);
    if (!m) continue;
    const key = m[1];
    const value = m[2].trim().replace(/['"]/g, '');
    if (indent === 0) section = key;
    if (key === 'dataset' && !value.includes(':')) cfg.dataset = value;
    if (key === 'name' && section === 'model' && value) cfg.model = value;
    if (key === 'name' && section === 'attack' && value) cfg.attack = value;
    if (key === 'epochs' && section === 'training' && value) cfg.epochs = Number(value);
    if (key === 'metrics' && section === 'evaluation') {
      // metrics 是列表，逐项在后续行以 "- name: direction" 或 "- name" 出现
      cfg._inMetrics = true;
    } else if (cfg._inMetrics && line.trim().startsWith('- ')) {
      const metric = line.trim().slice(2).split(':')[0].trim();
      if (metric) cfg.metrics.push(metric);
    } else if (indent <= 2) {
      cfg._inMetrics = false;
    }
  }
  return cfg;
}

function parseComparisonJson(text) {
  try {
    const data = JSON.parse(text);
    return {
      model_utility: data.model_utility || {},
      target_metrics: data.target_metrics || {},
    };
  } catch (e) {
    throw new Error(`解析失败: comparison JSON 不是合法 JSON（${e.message}）`);
  }
}

function readFile(files, name) {
  const f = files.find((x) => x.name === name);
  return f ? f.text : null;
}

function parseExperiment(segments, files) {
  const joined = segments.join('/');
  let kind = 'custom';
  let method = null;
  let dataset = null;
  let model = null;
  let runTag = null;

  const attacksIdx = segments.indexOf('attacks');
  const modelsIdx = segments.indexOf('models');
  if (attacksIdx !== -1) {
    kind = 'attack';
    method = segments[attacksIdx + 1] || null;
    const rest = segments.slice(attacksIdx + 2);
    const outIdx = rest.indexOf('outputs');
    const tail = outIdx === -1 ? [] : rest.slice(outIdx + 1);
    dataset = tail[0] || null;
    model = tail[1] || null;
    runTag = tail[2] && RUN_TAG_RE.test(tail[2]) ? tail[2] : null;
  } else if (modelsIdx !== -1) {
    kind = 'model';
    model = segments[modelsIdx + 1] || null;
    const rest = segments.slice(modelsIdx + 2);
    const outIdx = rest.indexOf('outputs');
    const tail = outIdx === -1 ? [] : rest.slice(outIdx + 1);
    runTag = tail[0] && RUN_TAG_RE.test(tail[0]) ? tail[0] : null;
  }

  const cfgText = readFile(files, 'config.yaml');
  const cfg = cfgText ? parseConfigYaml(cfgText) : {};
  if (kind === 'model' && !dataset) dataset = cfg.dataset || 'unknown';
  if (kind === 'attack') {
    dataset = dataset || cfg.dataset || 'unknown';
    model = model || cfg.model || 'unknown';
  }
  if (kind === 'custom') {
    const base = segments[segments.length - 1] || '未知';
    const historyText = readFile(files, 'history.json') || '{"history":[]}';
    return {
      id: `custom-${base}`,
      kind, method, dataset, model, runTag,
      label: `自定义-${base}`,
      color: null,
      history: parseHistoryJson(historyText),
      best: parseBest(historyText),
      evalLog: parseEvalCsv(readFile(files, 'eval_log.csv') || ''),
      comparison: null,
      meta: { epochs: cfg.epochs, metrics: cfg.metrics },
    };
  }

  const compFile = files.find((f) => /_comparison\.json$/.test(f.name));
  const historyText = readFile(files, 'history.json') || '{"history":[]}';
  return {
    id: buildExperimentId(kind, method, dataset, model, runTag || 'unknown'),
    kind, method, dataset, model, runTag,
    label: buildLabel(kind, method, dataset, model, runTag),
    color: null,
    history: parseHistoryJson(historyText),
    best: parseBest(historyText),
    evalLog: parseEvalCsv(readFile(files, 'eval_log.csv') || ''),
    comparison: compFile ? parseComparisonJson(compFile.text) : null,
    meta: { epochs: cfg.epochs, metrics: cfg.metrics },
  };
}

module.exports = {
  formatRunTag, buildExperimentId, buildLabel, parseExperiment,
  parseHistoryJson, parseBest, parseEvalCsv, parseConfigYaml, parseComparisonJson,
};
```

- [ ] **Step 4: 运行确认通过**

Run: `node --test TPA/visualization/tests/test_parser.js`
Expected: PASS（10 个用例）。

- [ ] **Step 5: 提交**

```bash
git add TPA/visualization/js/parser.js TPA/visualization/tests/test_parser.js
git commit -m "feat(visualization): parser 路径识别与实验产物解析（含 Node 单测）"
```

---

### Task 2: transforms.js（图表数据变换纯函数）

**Files:**
- Create: `TPA/visualization/js/transforms.js`
- Create: `TPA/visualization/tests/test_transforms.js`

**Interfaces:**
- Consumes: Task 1 的 `Experiment` 对象。
- Produces:
  - `assignPalette(experiments: Experiment[]): Experiment[]`（按序赋色）
  - `PALETTE: string[]`
  - `listMetrics(experiments: Experiment[]): string[]`
  - `buildLineSeries(experiments, metric, xKey='epoch'): {xAxis, series[]}`
  - `buildMetricBars(experiments, metrics): {xAxis, series[]}`
  - `buildComparisonSeries(experiments, metric, mode): {xAxis, series[]}`
  - `buildExportFilename(chartType, metric, now=new Date()): string`

- [ ] **Step 1: 写失败测试**

```js
const test = require('node:test');
const assert = require('node:assert');
const {
  assignPalette, listMetrics, buildLineSeries, buildMetricBars, buildComparisonSeries,
} = require('../js/transforms.js');

const mkExp = (id, history, comparison = null, metrics = ['recall@10', 'ndcg@10']) => ({
  id, label: id, color: null, history, evalLog: [], comparison,
  meta: { metrics },
});

test('调色板按序分配且稳定', () => {
  const exps = assignPalette([mkExp('a', []), mkExp('b', [])]);
  assert.notStrictEqual(exps[0].color, exps[1].color);
  const again = assignPalette([mkExp('a', []), mkExp('b', [])]);
  assert.strictEqual(again[0].color, exps[0].color);
});

test('指标枚举来自 history 数值列', () => {
  const exps = [mkExp('a', [{ epoch: 1, 'recall@10': 0.1 }, { epoch: 2, 'recall@10': 0.2 }])];
  assert.deepStrictEqual(listMetrics(exps), ['epoch', 'recall@10']);
});

test('折线 series：每个实验一条线', () => {
  const exps = [
    mkExp('e1', [{ epoch: 1, loss: 0.5 }, { epoch: 2, loss: 0.3 }]),
    mkExp('e2', [{ epoch: 1, loss: 0.6 }]),
  ];
  const out = buildLineSeries(exps, 'loss');
  assert.strictEqual(out.xAxis.length, 2);
  assert.strictEqual(out.series.length, 2);
  assert.deepStrictEqual(out.series[0].data, [0.5, 0.3]);
});

test('直方图：多指标分组柱', () => {
  const exps = [
    mkExp('e1', [], null, ['recall@10', 'ndcg@10']),
    mkExp('e2', [], null, ['recall@10', 'ndcg@10']),
  ];
  exps[0].best = { 'recall@10': { value: 0.1 }, 'ndcg@10': { value: 0.2 } };
  exps[1].best = { 'recall@10': { value: 0.11 }, 'ndcg@10': { value: 0.21 } };
  const out = buildMetricBars(exps, ['recall@10', 'ndcg@10']);
  assert.strictEqual(out.xAxis.length, 2);
  assert.strictEqual(out.series.length, 2);
  assert.deepStrictEqual(out.series[0].data, [0.1, 0.11]);
});

test('对比图：clean/poisoned 两组柱', () => {
  const exps = [mkExp('pgd-1', [], {
    model_utility: { clean: { 'recall@10': 0.1 }, poisoned: { 'recall@10': 0.2 } },
    target_metrics: {},
  })];
  const out = buildComparisonSeries(exps, 'recall@10');
  assert.strictEqual(out.series.length, 2);
  assert.deepStrictEqual(out.series[0].data, [0.1]);
  assert.deepStrictEqual(out.series[1].data, [0.2]);
});

test('导出文件名自动生成', () => {
  const name = buildExportFilename('line', 'ndcg@10', new Date(2026, 7, 15, 14, 30));
  assert.strictEqual(name, 'tpa-line-ndcg@10-20260815-1430.png');
});
```

- [ ] **Step 2: 运行确认失败**

Run: `node --test TPA/visualization/tests/test_transforms.js`
Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现 transforms.js**

```js
'use strict';

const PALETTE = [
  '#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de',
  '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc', '#2f4554',
];

function assignPalette(experiments) {
  return experiments.map((exp, i) => ({ ...exp, color: exp.color || PALETTE[i % PALETTE.length] }));
}

function numericKeys(rows) {
  const keys = new Set();
  for (const row of rows) {
    for (const [k, v] of Object.entries(row)) {
      if (typeof v === 'number') keys.add(k);
    }
  }
  return [...keys];
}

function listMetrics(experiments) {
  const keys = new Set();
  for (const exp of experiments) {
    for (const k of numericKeys(exp.history)) keys.add(k);
    for (const k of numericKeys(exp.evalLog)) keys.add(k);
  }
  return [...keys];
}

function buildLineSeries(experiments, metric, xKey = 'epoch') {
  const xSet = new Set();
  for (const exp of experiments) {
    for (const row of exp.history) xSet.add(row[xKey]);
    for (const row of exp.evalLog) xSet.add(row[xKey]);
  }
  const xAxis = [...xSet].sort((a, b) => a - b);
  const series = experiments.map((exp) => {
    const dataByX = new Map();
    for (const row of [...exp.history, ...exp.evalLog]) {
      const x = row[xKey];
      if (x !== undefined && typeof row[metric] === 'number') dataByX.set(x, row[metric]);
    }
    return {
      name: exp.label,
      type: 'line',
      color: exp.color,
      data: xAxis.map((x) => (dataByX.has(x) ? dataByX.get(x) : null)),
    };
  });
  return { xAxis, series };
}

function bestValue(exp, metric) {
  if (exp.best && exp.best[metric] && typeof exp.best[metric].value === 'number') {
    return exp.best[metric].value;
  }
  for (const row of [...exp.history, ...exp.evalLog].reverse()) {
    if (typeof row[metric] === 'number') return row[metric];
  }
  return null;
}

function buildMetricBars(experiments, metrics) {
  const xAxis = experiments.map((e) => e.label);
  const series = metrics.map((metric) => ({
    name: metric,
    type: 'bar',
    data: experiments.map((e) => bestValue(e, metric)),
  }));
  return { xAxis, series };
}

function targetAvg(comparison, side, metric) {
  const targets = (comparison.target_metrics || {})[side] || {};
  const values = Object.values(targets).map((t) => t[metric]).filter((v) => typeof v === 'number');
  if (!values.length) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function buildComparisonSeries(experiments, metric, mode = 'model') {
  const xAxis = experiments.map((e) => e.label);
  const pick = (exp, side) => {
    const cmp = exp.comparison || {};
    const src = mode === 'model' ? (cmp.model_utility || {}) : (cmp.target_metrics || {});
    const sideData = src[side] || {};
    if (mode === 'model') return typeof sideData[metric] === 'number' ? sideData[metric] : null;
    return targetAvg(exp.comparison, side, metric);
  };
  return {
    xAxis,
    series: [
      { name: 'Clean', type: 'bar', data: experiments.map((e) => pick(e, 'clean')) },
      { name: 'Poisoned', type: 'bar', data: experiments.map((e) => pick(e, 'poisoned')) },
    ],
  };
}

function pad2(n) {
  return String(n).padStart(2, '0');
}

function buildExportFilename(chartType, metric, now = new Date()) {
  const safeMetric = String(metric || 'summary').replace(/[^\w@%.-]/g, '-');
  const stamp = `${now.getFullYear()}${pad2(now.getMonth() + 1)}${pad2(now.getDate())}-${pad2(now.getHours())}${pad2(now.getMinutes())}`;
  return `tpa-${chartType}-${safeMetric}-${stamp}.png`;
}

module.exports = {
  PALETTE, assignPalette, listMetrics, buildLineSeries, buildMetricBars,
  buildComparisonSeries, buildExportFilename,
};
```

- [ ] **Step 4: 运行确认通过**

Run: `node --test TPA/visualization/tests/test_transforms.js`
Expected: PASS（6 个用例）。

- [ ] **Step 5: 提交**

```bash
git add TPA/visualization/js/transforms.js TPA/visualization/tests/test_transforms.js
git commit -m "feat(visualization): 折线/直方/对比图数据变换（含 Node 单测）"
```

---

### Task 3: app.js（导入、多选、编辑、持久化）

**Files:**
- Create: `TPA/visualization/js/app.js`

**Interfaces:**
- Consumes: Task 1 `parseExperiment`、Task 2 各变换函数。
- Produces（供 Task 4 的 index.html 调用）：
  - `initApp({onStateChange})`
  - `importFiles(fileEntries: {relativePath, name, text}[])`
  - `state`（模块级）：`{ experiments: Experiment[], selected: Set<string>, metric, chartType, title }`

- [ ] **Step 1: 实现导入流程与状态管理**

核心逻辑（含导出快照）：

```js
'use strict';

const { parseExperiment } = require('./parser.js');

const STORAGE_KEY = 'tpa.visualizer.v1';

function normalizeEntry(relPath, name, text) {
  return { segments: relPath.split('/').filter(Boolean), name, text };
}

function groupByExperiment(entries) {
  const groups = new Map();
  for (const entry of entries) {
    const seg = entry.segments;
    const runIdx = seg.findIndex((s) => /^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}$/.test(s));
    const key = runIdx !== -1 ? seg.slice(0, runIdx + 1).join('/') : seg[0] || 'unknown';
    if (!groups.has(key)) groups.set(key, { segments: seg.slice(0, runIdx + 1), files: [] });
    groups.get(key).files.push(entry);
  }
  return [...groups.values()];
}

function importFiles(entries) {
  const groups = groupByExperiment(entries.map((e) => normalizeEntry(e.relativePath, e.name, e.text)));
  const experiments = [];
  const errors = [];
  for (const g of groups) {
    try {
      experiments.push(parseExperiment(g.segments, g.files));
    } catch (e) {
      errors.push(`${g.segments.join('/')}: ${e.message}`);
    }
  }
  return { experiments, errors };
}

function saveState(state) {
  try {
    const payload = {
      experiments: state.experiments,
      selected: [...state.selected],
      metric: state.metric,
      chartType: state.chartType,
      title: state.title,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch (e) {
    console.warn('localStorage 持久化失败（容量或隐私模式），仅本次会话保留', e);
  }
}

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (e) {
    return null;
  }
}

function initApp() {
  // UI 装配在 Task 4 的 index.html 中调用；此处仅暴露纯状态接口
  return {
    importFiles,
    saveState,
    loadState,
    STORAGE_KEY,
  };
}

module.exports = { initApp, importFiles, saveState, loadState, STORAGE_KEY };
```

> 注：浏览器版 app.js 需处理 File System 读取（`webkitdirectory` 文件夹选择 /
> 拖拽 `DataTransferItem.webkitGetAsEntry()`）与 localStorage 的浏览器差异；
> 该部分在 Task 4 与 index.html 集成时补齐（IIFE 挂到 `window`，Node 测试用
> `module.exports` 分支：`if (typeof window === 'undefined') module.exports = ...`）。

- [ ] **Step 2: 补充 importFiles 的 Node 单测（并入 tests/test_parser.js 或新建 test_app.js）**

```js
const { importFiles } = require('../js/app.js');

test('批量导入：父目录下多个实验被分组识别', () => {
  const entries = [
    { relativePath: 'attacks/pgd/outputs/ml100k/lightgcn/2026-08-09-21-54/history.json',
      name: 'history.json', text: '{"history":[]}' },
    { relativePath: 'attacks/pgd/outputs/ml100k/lightgcn/2026-08-09-21-54/config.yaml',
      name: 'config.yaml', text: 'dataset: ml100k\nattack:\n  name: pgd\n' },
    { relativePath: 'attacks/tpa/outputs/ml100k/lightgcn/2026-08-09-22-30/history.json',
      name: 'history.json', text: '{"history":[]}' },
  ];
  const { experiments, errors } = importFiles(entries);
  assert.strictEqual(errors.length, 0);
  assert.strictEqual(experiments.length, 2);
  assert.ok(experiments.some((e) => e.method === 'pgd'));
  assert.ok(experiments.some((e) => e.method === 'tpa'));
});
```

- [ ] **Step 3: 运行测试**

Run: `node --test TPA/visualization/tests/`
Expected: PASS（前两任务用例 + 本任务新增用例）。

- [ ] **Step 4: 提交**

```bash
git add TPA/visualization/js/app.js TPA/visualization/tests/
git commit -m "feat(visualization): 导入分组、状态管理与持久化"
```

---

### Task 4: index.html + styles.css + ECharts 渲染集成

**Files:**
- Create: `TPA/visualization/index.html`
- Create: `TPA/visualization/styles.css`
- Create: `TPA/visualization/lib/echarts.min.js`（vendored，下载 ECharts 5 单文件）

**Interfaces:**
- Consumes: Task 3 `initApp().importFiles`、Task 2 变换函数、ECharts `echarts.init`。

- [ ] **Step 1: 下载 ECharts 到 lib/**

Run（联网一次）:
```bash
curl -L -o TPA/visualization/lib/echarts.min.js https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js
```
校验：文件头部为 ECharts 版权注释，大小约 1MB；提交入库（离线可用）。

- [ ] **Step 2: 写 index.html 骨架**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>TPA 实验数据可视化</title>
  <link rel="stylesheet" href="styles.css">
  <script src="lib/echarts.min.js"></script>
</head>
<body>
  <header>
    <h1 id="chart-title">TPA 实验数据可视化</h1>
    <div class="toolbar">
      <button id="btn-import">导入数据</button>
      <input id="file-input" type="file" webkitdirectory multiple hidden>
      <button id="btn-export">导出图片</button>
      <button id="btn-export-snapshot">导出快照</button>
      <button id="btn-load-snapshot">导入快照</button>
      <input id="snapshot-input" type="file" accept="application/json" hidden>
    </div>
    <div id="drop-zone" class="drop-zone">拖拽实验文件夹到此处</div>
  </header>
  <main class="layout">
    <aside id="experiment-panel" class="panel">
      <h2>实验列表</h2>
      <ul id="experiment-list"></ul>
    </aside>
    <section id="chart-panel" class="panel">
      <div class="chart-controls">
        <select id="chart-type">
          <option value="line">折线图</option>
          <option value="bar">直方图</option>
          <option value="compare">对比图（Clean vs Poisoned）</option>
        </select>
        <select id="metric-select"></select>
      </div>
      <div id="chart" class="chart"></div>
    </section>
  </main>
  <div id="message" class="message"></div>
  <script src="js/parser.js"></script>
  <script src="js/transforms.js"></script>
  <script src="js/app.js"></script>
  <script src="js/main.js"></script>
</body>
</html>
```

- [ ] **Step 3: 实现 js/main.js 渲染装配（UI 薄壳）**

关键渲染函数（完整代码见实现时按此逻辑）：

```js
function renderList() {
  const ul = document.getElementById('experiment-list');
  ul.innerHTML = '';
  state.experiments.forEach((exp) => {
    const li = document.createElement('li');
    li.innerHTML = `
      <label>
        <input type="checkbox" data-id="${exp.id}" ${state.selected.has(exp.id) ? 'checked' : ''}>
        <span class="color-dot" style="background:${exp.color}"></span>
        <input class="label-input" data-id="${exp.id}" value="${exp.label}">
      </label>`;
    ul.appendChild(li);
  });
}

function renderChart() {
  const chart = echarts.getInstanceByDom(document.getElementById('chart'))
    || echarts.init(document.getElementById('chart'));
  const exps = state.experiments.filter((e) => state.selected.has(e.id));
  let option;
  if (state.chartType === 'line') {
    const { xAxis, series } = buildLineSeries(exps, state.metric);
    option = { title: { text: state.title }, tooltip: {}, legend: {},
      xAxis: { type: 'category', data: xAxis }, yAxis: { type: 'value' },
      series, dataZoom: [{ type: 'inside' }, { type: 'slider' }] };
  } else if (state.chartType === 'bar') {
    const metrics = listMetrics(exps).filter((m) => m !== 'epoch');
    const { xAxis, series } = buildMetricBars(exps, metrics);
    option = { title: { text: state.title }, tooltip: {}, legend: {},
      xAxis: { type: 'category', data: xAxis, axisLabel: { rotate: 20 } },
      yAxis: { type: 'value' }, series };
  } else {
    const metrics = listMetrics(exps).filter((m) => m !== 'epoch' && !m.startsWith('target_'));
    const { xAxis, series } = buildComparisonSeries(exps, state.metric);
    option = { title: { text: state.title }, tooltip: {}, legend: {},
      xAxis: { type: 'category', data: xAxis, axisLabel: { rotate: 20 } },
      yAxis: { type: 'value' }, series };
  }
  chart.setOption(option, true);
}

function exportChart() {
  const chart = echarts.getInstanceByDom(document.getElementById('chart'));
  if (!chart) {
    showMessage('当前无图表可导出');
    return;
  }
  const url = chart.getDataURL({
    type: 'png', pixelRatio: 2, backgroundColor: '#fff',
  });
  const a = document.createElement('a');
  a.href = url;
  a.download = buildExportFilename(state.chartType, state.metric);
  document.body.appendChild(a);
  a.click();
  a.remove();
}

document.getElementById('btn-export').addEventListener('click', exportChart);
```

- [ ] **Step 4: 手动验证（README 清单）**

1. `file://` 打开 index.html，导入 `TPA/attacks/pgd/outputs/ml100k/lightgcn/2026-08-09-21-54/`，列表显示 `attack-pgd-ml100k-lightgcn-2026年08月09日21时54分`。
2. 再导入 `models/lightgcn/outputs/2026-08-09-23-01/` 与 `models/lightgcn/outputs/2026-08-09-23-39/`，勾选全部，折线图显示 3 条不同颜色曲线（指标切到 ndcg@10）。
3. 直方图显示多指标分组柱；对比图（攻击实验）显示 Clean/Poisoned 两组柱。
4. 编辑标签/颜色、隐藏一条、改标题，刷新页面后状态保留。
5. 导出快照 → 清空 → 导入快照恢复。
6. 点击「导出图片」，下载 `tpa-line-ndcg@10-*.png`（或其他图表类型），
   图片可打开、内容为当前图表（2x 分辨率、白底）。

- [ ] **Step 5: 提交**

```bash
git add TPA/visualization/index.html TPA/visualization/styles.css TPA/visualization/js/main.js TPA/visualization/lib/echarts.min.js
git commit -m "feat(visualization): HTML 页面与 ECharts 渲染集成"
```

---

### Task 5: README + 全量回归

**Files:**
- Create: `TPA/visualization/README.md`

- [ ] **Step 1: 写 README**

内容：功能简介、目录结构、打开方式（file:// 或 `python -m http.server`）、导入方式、
图表说明、图片导出（PNG、2x、文件名规则）、编辑与快照说明、Node 单测命令、
手动验证清单、已知限制
（纯前端不修改原始数据；localStorage 容量上限）。

- [ ] **Step 2: 全量回归**

Run:
```bash
node --test TPA/visualization/tests/
cd TPA && G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_attack_eval tests.test_attack_fit_consistency tests.test_modes tests.test_training_metrics tests.test_portable_paths -v
```
Expected: JS 全绿；Python 40/40 通过。

- [ ] **Step 3: 提交**

```bash
git add TPA/visualization/README.md
git commit -m "docs(visualization): 使用文档与手动验证清单"
```

---

## 计划自审

- Spec 覆盖：识别命名（Task 1）、多选配色与折线（Task 2/4）、直方图（Task 2/4）、
  对比图（Task 2/4）、编辑（Task 3/4）、导入（Task 3）、离线可用（Task 4）、
  测试与文档（Task 1/2/5）。无缺口。
- 无占位符：所有代码步骤给出可运行代码或明确逻辑。
- 类型一致：`Experiment`、`parseExperiment(segments, files)`、
  `buildLineSeries(experiments, metric)` 等签名在任务间一致。
