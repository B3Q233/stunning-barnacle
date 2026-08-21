'use strict';
const test = require('node:test');
const assert = require('node:assert');

const { buildTree, collectLeaves, numericLeaves } = require('../designer/tree_builder.js');


test('buildTree 生成对象/数组/叶子节点与完整路径', () => {
  const json = {
    summary: { best_hr: 0.61 },
    history: [{ epoch: 1, loss: 0.2, name: 'x' }],
  };
  const root = buildTree(json);
  const leaves = collectLeaves(root);
  assert.deepStrictEqual(
    leaves.map((l) => l.path).sort(),
    ['history[].epoch', 'history[].loss', 'history[].name', 'summary.best_hr']);
});

test('叶子类型自动识别且 numericLeaves 只含 number', () => {
  const json = { a: 1, b: 'x', c: true, d: null, arr: [1] };
  const root = buildTree(json);
  const leaves = collectLeaves(root);
  const types = Object.fromEntries(leaves.map((l) => [l.path, l.type]));
  assert.strictEqual(types.a, 'number');
  assert.strictEqual(types.b, 'string');
  assert.strictEqual(types.c, 'boolean');
  const nums = numericLeaves(root).map((l) => l.path);
  assert.deepStrictEqual(nums, ['a']);
});
