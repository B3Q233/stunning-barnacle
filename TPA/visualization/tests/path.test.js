'use strict';
const test = require('node:test');
const assert = require('node:assert');

const { extract } = require('../registry/path.js');


test('extract 支持数组节点 history[].epoch', () => {
  const json = {
    history: [
      { epoch: 1, train_loss: 0.5 },
      { epoch: 2, train_loss: 0.4 },
    ],
  };
  assert.deepStrictEqual(extract(json, 'history[].epoch'), [1, 2]);
  assert.deepStrictEqual(extract(json, 'history[].train_loss'), [0.5, 0.4]);
});

test('extract 支持嵌套对象与带 @ 的键', () => {
  const json = {
    summary: { best_hr: 0.61 },
    history: [{ epoch: 1, 'recall@10': 0.1 }],
  };
  assert.strictEqual(extract(json, 'summary.best_hr'), 0.61);
  assert.deepStrictEqual(extract(json, 'history[].recall@10'), [0.1]);
});

test('extract 支持深层 targets.908.ndcg', () => {
  const json = {
    history: [
      { epoch: 1, targets: { '908': { ndcg: 0.15 } } },
      { epoch: 2, targets: { '908': { ndcg: 0.18 } } },
    ],
  };
  assert.deepStrictEqual(extract(json, 'history[].targets.908.ndcg'), [0.15, 0.18]);
});

test('extract 缺失路径返回 null，缺失数组返回 []', () => {
  assert.strictEqual(extract({ a: 1 }, 'a.b.c'), null);
  assert.deepStrictEqual(extract({}, 'history[].epoch'), []);
});
