'use strict';
const test = require('node:test');
const assert = require('node:assert');

// 加载内置 schema（副作用：注册到 registry）
require('../registry/schemas/history.js');
require('../registry/schemas/comparison.js');
require('../registry/schemas/tier_stats.js');
require('../registry/schemas/meta.js');

const { getSchema } = require('../registry/index.js');
const { buildVisualization } = require('../registry/normalize.js');


test('history schema 匹配并提取 series', () => {
  const json = {
    history: [
      { epoch: 1, train_loss: 0.5, 'recall@10': 0.1 },
      { epoch: 2, train_loss: 0.4, 'recall@10': 0.2 },
    ],
  };
  const schema = getSchema(json);
  assert.strictEqual(schema.name, 'history');
  const view = buildVisualization(json);
  assert.strictEqual(view.type, 'line');
  assert.deepStrictEqual(view.x, [1, 2]);
  const loss = view.series.find((s) => s.name === 'train_loss');
  assert.deepStrictEqual(loss.data, [0.5, 0.4]);
});

test('comparison schema 匹配攻击对比 JSON', () => {
  const json = {
    model_utility: { clean: { 'recall@10': 0.1 }, poisoned: { 'recall@10': 0.2 } },
    target_metrics: { clean: {}, poisoned: {} },
  };
  assert.strictEqual(getSchema(json).name, 'comparison');
  const view = buildVisualization(json);
  assert.strictEqual(view.type, 'metric');
  assert.ok(view.series.some((s) => s.name === 'Clean Recall@10'));
});

test('tier_stats schema 提取各层均值', () => {
  const json = {
    batch_tag: '2026-08-21-18-35',
    k: 10,
    tiers: {
      popular: { 'target_hr@10': { mean: 0.1787, std: 0.12, n: 10 } },
      cold: { 'target_hr@10': { mean: 0.0015, std: 0.003, n: 10 } },
    },
  };
  assert.strictEqual(getSchema(json).name, 'tier_stats');
  const view = buildVisualization(json);
  assert.strictEqual(view.type, 'bar');
  const hr = view.series.find((s) => s.name === 'Popular HR@10');
  assert.strictEqual(hr.data, 0.1787);
});

test('meta schema 匹配批量元信息', () => {
  const json = {
    batch_tag: '2026-08-21-18-35',
    attack: 'random',
    dataset: 'ml100k',
    model: 'lightgcn',
    topk: 10,
    total_runs: 30,
  };
  assert.strictEqual(getSchema(json).name, 'meta');
  const view = buildVisualization(json);
  assert.strictEqual(view.type, 'metric');
  assert.ok(view.series.some((s) => s.name === 'total_runs'));
});
