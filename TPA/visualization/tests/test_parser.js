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
