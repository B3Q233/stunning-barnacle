'use strict';
const test = require('node:test');
const assert = require('node:assert');

const registry = require('../registry/registry.js');
const { normalize } = require('../registry/normalize.js');


test('history schema 归一化为折线', () => {
  const json = {
    history: [{ epoch: 1, train_loss: 0.5, 'recall@10': 0.1 }],
  };
  // 缺省字段的 history 未命中内置指纹 → null（应触发设计器），
  // 这里用完整内置结构验证归一化
  const full = {
    history: [{
      epoch: 1, train_loss: 0.5, val_loss: 0.4,
      'recall@10': 0.1, 'ndcg@10': 0.2,
      'target_hr@10': 0.3, 'target_ndcg@10': 0.4,
    }],
  };
  const view = normalize(full, registry.match(full));
  assert.strictEqual(view.type, 'line');
  assert.ok(view.series.some((s) => s.name === 'train_loss'));
});

test('tier_stats schema 提取各层均值', () => {
  const tier = {
    'target_hr@10': { mean: 0.17, std: 0.1, n: 10 },
    'target_ndcg@10': { mean: 0.08, std: 0.06, n: 10 },
    'recall@10': { mean: 0.147, std: 0.001, n: 10 },
    'ndcg@10': { mean: 0.203, std: 0.001, n: 10 },
  };
  const json = {
    batch_tag: '2026-08-21-18-35', k: 10,
    tiers: { popular: tier, normal: tier, cold: tier },
  };
  const view = normalize(json, registry.match(json));
  assert.strictEqual(view.type, 'bar');
  const hr = view.series.find((s) => s.name === 'Popular HR@10');
  assert.strictEqual(hr.data, 0.17);
});
