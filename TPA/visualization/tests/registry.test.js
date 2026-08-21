'use strict';
const test = require('node:test');
const assert = require('node:assert');

const { extract } = require('../registry/path.js');
const registry = require('../registry/index.js');
const { VisualizationSchema } = require('../registry/base.js');
const { normalize, buildVisualization } = require('../registry/normalize.js');


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


test('register/getSchema 按 match 匹配并抛错', () => {
  registry.clear();
  class DummySchema extends VisualizationSchema {
    constructor() {
      super();
      this.name = 'dummy';
      this.series = { A: 'foo' };
    }
    match(json) {
      return json && json.foo !== undefined;
    }
  }
  registry.register(new DummySchema());
  const schema = registry.getSchema({ foo: 1 });
  assert.strictEqual(schema.name, 'dummy');
  assert.throws(() => registry.getSchema({ bar: 1 }), /未找到匹配/);
});


test('normalize 输出统一格式 line', () => {
  const json = { history: [{ epoch: 1, train_loss: 0.5 }] };
  const schema = new VisualizationSchema();
  schema.name = 't';
  schema.title = 'T';
  schema.type = 'line';
  schema.x = 'history[].epoch';
  schema.series = { Loss: 'history[].train_loss' };
  const view = normalize(json, schema);
  assert.deepStrictEqual(view, {
    title: 'T',
    type: 'line',
    x: [1],
    series: [{ name: 'Loss', data: [0.5] }],
  });
});

test('normalize metric 类型无 x', () => {
  const view = normalize({ summary: { best_hr: 0.6 } }, {
    title: 'S', type: 'metric', x: null,
    series: { 'Best HR': 'summary.best_hr' },
  });
  assert.strictEqual(view.x, null);
  assert.deepStrictEqual(view.series[0].data, 0.6);
});
