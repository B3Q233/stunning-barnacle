'use strict';
const test = require('node:test');
const assert = require('node:assert');

const registry = require('../registry/registry.js');
const { fingerprint, schemaId } = require('../registry/detector.js');
const { normalize } = require('../registry/normalize.js');


test('内置 schema 已加载且按指纹匹配 history', () => {
  const json = {
    history: [{
      epoch: 1, train_loss: 0.5, val_loss: 0.4,
      'recall@10': 0.1, 'ndcg@10': 0.2,
      'target_hr@10': 0.3, 'target_ndcg@10': 0.4,
    }],
  };
  const schema = registry.match(json);
  assert.ok(schema);
  assert.strictEqual(schema.type, 'line');
  const view = normalize(json, schema);
  assert.strictEqual(view.type, 'line');
  assert.deepStrictEqual(view.x, [1]);
});

test('内置 schema 匹配 comparison / tier_stats / meta', () => {
  const cmp = {
    model_utility: {
      clean: { 'recall@10': 0.1, 'ndcg@10': 0.2 },
      poisoned: { 'recall@10': 0.2, 'ndcg@10': 0.15 },
    },
    target_metrics: { clean: {}, poisoned: {} },
  };
  assert.strictEqual(registry.match(cmp).type, 'metric');

  const tier = {
    'target_hr@10': { mean: 0.17, std: 0.1, n: 10 },
    'target_ndcg@10': { mean: 0.08, std: 0.06, n: 10 },
    'recall@10': { mean: 0.147, std: 0.001, n: 10 },
    'ndcg@10': { mean: 0.203, std: 0.001, n: 10 },
  };
  const tiers = {
    batch_tag: '2026-08-21-18-35', k: 10,
    tiers: { popular: tier, normal: tier, cold: tier },
  };
  assert.strictEqual(registry.match(tiers).type, 'bar');

  const meta = { batch_tag: 't', attack: 'random', dataset: 'ml100k',
    model: 'lightgcn', topk: 10, per_tier: 10, total_runs: 30, seed: 42,
    tiers: ['popular', 'normal', 'cold'] };
  assert.strictEqual(registry.match(meta).type, 'metric');
});

test('未注册结构返回 null（触发设计器）', () => {
  assert.strictEqual(registry.match({ foo: { bar: [1, 2] } }), null);
});

test('自定义 schema 保存后可匹配并归一化', () => {
  registry.resetCustom();
  const json = { result: [{ step: 1, asr: 0.5 }] };
  const fp = fingerprint(json);
  const schema = {
    id: schemaId(fp),
    title: 'Custom Attack',
    fingerprint: fp,
    type: 'line',
    x: 'result[].step',
    series: { ASR: 'result[].asr' },
  };
  registry.saveCustom(schema);
  const got = registry.match(json);
  assert.strictEqual(got.id, schema.id);
  const view = normalize(json, got);
  assert.deepStrictEqual(view.x, [1]);
  assert.deepStrictEqual(view.series[0].data, [0.5]);
});
