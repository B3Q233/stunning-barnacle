'use strict';
// 浏览器全链路回归：按 index.html 顺序加载全部脚本（含 main.js），
// 验证 自定义 JSON → 指纹匹配/设计器建树 的真实路径。
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.join(__dirname, '..');
const SCRIPT_ORDER = [
  'js/parser.js',
  'js/transforms.js',
  'registry/path.js',
  'registry/detector.js',
  'registry/schemas/builtin.js',
  'registry/registry.js',
  'registry/normalize.js',
  'designer/tree_builder.js',
  'designer/designer.js',
  'js/main.js',
];

function makeEl(tag) {
  return {
    tagName: tag, className: '', textContent: '', innerHTML: '', value: '',
    type: '', name: '', checked: false, style: {}, children: [], open: false,
    classList: {
      _s: new Set(),
      add(...cs) { cs.forEach((c) => this._s.add(c)); },
      remove(...cs) { cs.forEach((c) => this._s.delete(c)); },
      contains(c) { return this._s.has(c); },
    },
    appendChild(c) { this.children.push(c); return c; },
    append(...cs) { this.children.push(...cs); },
    addEventListener() {}, removeEventListener() {}, focus() {}, click() {},
    querySelectorAll() { return []; },
  };
}

function buildBrowser() {
  const byId = {};
  for (const id of ['modal-overlay', 'modal-body', 'modal-title', 'modal-ok',
    'modal-cancel', 'message', 'card-list', 'card-panel', 'chart-line',
    'chart-bar', 'add-card', 'export-json']) {
    byId[id] = makeEl('div');
  }
  byId['modal-overlay'].classList.add('hidden');

  const sandbox = {
    console,
    document: {
      getElementById: (id) => byId[id] || makeEl('div'),
      createElement: (t) => makeEl(t),
      createTextNode: (t) => ({ nodeType: 3, textContent: t }),
    },
    localStorage: {
      _m: {},
      getItem(k) { return this._m[k] !== undefined ? this._m[k] : null; },
      setItem(k, v) { this._m[k] = String(v); },
    },
    FileReader: class {
      readAsText(file) { this.result = file._text; this.onload && this.onload(); }
    },
    echarts: {
      getInstanceByDom: () => null,
      init: () => ({ setOption() {}, dispose() {} }),
    },
    Blob: class { constructor(parts) { this.parts = parts; } },
    URL: { createObjectURL: () => 'blob:x', revokeObjectURL() {} },
    setTimeout, clearTimeout,
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);

  for (const f of SCRIPT_ORDER) {
    vm.runInContext(fs.readFileSync(path.join(ROOT, f), 'utf-8'), sandbox,
      { filename: f });
  }
  return { sandbox, byId };
}

function tierStatsJson() {
  const tier = {
    'target_hr@10': { mean: 0.17, std: 0.1, n: 10 },
    'target_ndcg@10': { mean: 0.08, std: 0.06, n: 10 },
    'recall@10': { mean: 0.147, std: 0.001, n: 10 },
    'ndcg@10': { mean: 0.203, std: 0.001, n: 10 },
  };
  return {
    batch_tag: 't', k: 10,
    tiers: { popular: tier, normal: tier, cold: tier },
  };
}

test('未知结构：导入自定义 JSON 弹出设计器并构建树', () => {
  const { sandbox, byId } = buildBrowser();
  const T = sandbox.TPAVisualizer;
  const card = T.app.addCard('t');
  const text = JSON.stringify({ foo: { bar: 1, list: [{ x: 0.5 }] } });
  T.app.applyFileToCard(card, { name: 'custom.json' }, text);
  assert.ok(!byId['modal-overlay'].classList.contains('hidden'),
    '设计器应打开（modal-overlay 不应 hidden）');
  const tree = byId['modal-body'].children
    .find((c) => c.className === 'designer-tree');
  assert.ok(tree, 'modal 中应存在 designer-tree 容器');
  assert.ok(tree.children.length > 0, '树中应存在节点');
});

test('已注册结构：导入自定义 JSON 直接渲染视图', () => {
  const { sandbox } = buildBrowser();
  const T = sandbox.TPAVisualizer;
  const card = T.app.addCard('t');
  T.app.applyFileToCard(card, { name: 'tier_stats.json' },
    JSON.stringify(tierStatsJson()));
  assert.strictEqual(card.registryViews.length, 1);
  assert.strictEqual(card.registryViews[0].view.type, 'bar');
  assert.strictEqual(card.registryViews[0].view.series.length, 6);
});
