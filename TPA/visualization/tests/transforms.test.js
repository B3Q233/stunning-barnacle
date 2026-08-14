const test = require('node:test');
const assert = require('node:assert');
const {
  assignPalette, listMetrics, buildLineSeries, buildMetricBars,
  buildComparisonSeries, buildExportFilename,
} = require('../js/transforms.js');

const mkExp = (id, history, comparison = null, metrics = ['recall@10', 'ndcg@10']) => ({
  id, label: id, color: null, history, evalLog: [], comparison,
  meta: { metrics }, best: null,
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
