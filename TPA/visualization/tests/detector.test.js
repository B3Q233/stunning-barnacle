'use strict';
const test = require('node:test');
const assert = require('node:assert');

const { fingerprint, schemaId } = require('../registry/detector.js');


test('fingerprint 只依赖结构，不依赖值', () => {
  const a = { history: [{ epoch: 1, loss: 0.2, acc: 0.8 }] };
  const b = { history: [{ epoch: 5, loss: 0.9, acc: 0.7 }] };
  assert.deepStrictEqual(fingerprint(a), fingerprint(b));
});

test('fingerprint 输出排序后的完整路径', () => {
  const fp = fingerprint({ history: [{ epoch: 1, loss: 0.2 }] });
  assert.deepStrictEqual(fp, [
    'history',
    'history[]',
    'history[].epoch',
    'history[].loss',
  ]);
});

test('fingerprint 覆盖嵌套对象与数组键', () => {
  const fp = fingerprint({ summary: { best_hr: 0.61 }, result: [{ step: 1 }] });
  assert.deepStrictEqual(fp, [
    'result',
    'result[]',
    'result[].step',
    'summary',
    'summary.best_hr',
  ]);
});

test('schemaId 稳定且为 fp_ 前缀短哈希', () => {
  const fp = fingerprint({ a: 1 });
  assert.strictEqual(schemaId(fp), schemaId(fingerprint({ a: 2 })));
  assert.match(schemaId(fp), /^fp_[0-9a-f]{6}$/);
});
