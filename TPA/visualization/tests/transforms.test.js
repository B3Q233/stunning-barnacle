'use strict';
const test = require('node:test');
const assert = require('node:assert');
const {
  buildLineSeries, buildComparisonItems, buildComparisonSeries,
  buildMultiLineSeries, buildMultiBarSeries,
} = require('../js/transforms.js');

test('折线 series：x 按 epoch 排序，缺指标为 null', () => {
  const em = {
    '1': { train_loss: 0.5, 'recall@10': 0.1 },
    '3': { train_loss: 0.3 },
    '2': { train_loss: 0.4, 'recall@10': 0.2 },
  };
  const { xAxis, series } = buildLineSeries(em, ['train_loss', 'recall@10']);
  assert.deepStrictEqual(xAxis, ['1', '2', '3']);
  assert.strictEqual(series[0].type, 'line');
  assert.deepStrictEqual(series[0].data, [0.5, 0.4, 0.3]);
  assert.deepStrictEqual(series[1].data, [0.1, 0.2, null]);
});

test('直方对比项：模型效用 + 目标多目标取平均', () => {
  const cmp = {
    modelUtility: {
      clean: { 'recall@10': 0.1, 'ndcg@10': 0.2 },
      poisoned: { 'recall@10': 0.2, 'ndcg@10': 0.15 },
    },
    targetMetrics: {
      clean: {
        '251': { 'hr@k': 0.01, 'ndcg@k': 0.005 },
        '252': { 'hr@k': 0.03, 'ndcg@k': 0.015 },
      },
      poisoned: {
        '251': { 'hr@k': 0.05, 'ndcg@k': 0.02 },
        '252': { 'hr@k': 0.07, 'ndcg@k': 0.04 },
      },
    },
  };
  const items = buildComparisonItems(cmp);
  const names = items.map((it) => it.name);
  assert.ok(names.includes('recall@10'));
  assert.ok(names.includes('target_hr@k'));
  const hr = items.find((it) => it.name === 'target_hr@k');
  assert.ok(Math.abs(hr.clean - 0.02) < 1e-9);
  assert.ok(Math.abs(hr.poisoned - 0.06) < 1e-9);
});

test('直方 series：按选中项生成 Clean/Poisoned 两列', () => {
  const items = [
    { name: 'recall@10', clean: 0.1, poisoned: 0.2 },
    { name: 'ndcg@10', clean: 0.2, poisoned: 0.15 },
  ];
  const { xAxis, series } = buildComparisonSeries(items, ['ndcg@10']);
  assert.deepStrictEqual(xAxis, ['ndcg@10']);
  assert.strictEqual(series[0].name, 'Clean');
  assert.deepStrictEqual(series[0].data, [0.2]);
  assert.deepStrictEqual(series[1].data, [0.15]);
});

test('多实验折线：同一指标按实验拆线且颜色可自定义', () => {
  const exps = [
    { name: 'A', epochMetrics: { '1': { 'recall@10': 0.1 }, '2': { 'recall@10': 0.2 } },
      metricOptions: { 'recall@10': { selected: true, color: '#111111' } } },
    { name: 'B', epochMetrics: { '1': { 'recall@10': 0.3 } },
      metricOptions: { 'recall@10': { selected: true, color: '#333333' } } },
  ];
  const { xAxis, series } = buildMultiLineSeries(exps);
  assert.deepStrictEqual(xAxis, ['1', '2']);
  assert.deepStrictEqual(series.map((s) => s.name), ['recall@10-A', 'recall@10-B']);
  assert.strictEqual(series[0].color, '#111111');
  assert.deepStrictEqual(series[0].data, [0.1, 0.2]);
  assert.deepStrictEqual(series[1].data, [0.3, null]);
});

test('多实验直方：x=实验，每指标 Clean/Poisoned 两列并逐点着色', () => {
  const exps = [
    { name: 'A',
      comparison: {
        modelUtility: { clean: { 'recall@10': 0.1 }, poisoned: { 'recall@10': 0.2 } },
        targetMetrics: {},
      },
      metricOptions: { 'recall@10': { selected: true, color: '#111111' } } },
    { name: 'B',
      comparison: {
        modelUtility: { clean: { 'recall@10': 0.3 }, poisoned: { 'recall@10': 0.4 } },
        targetMetrics: {},
      },
      metricOptions: { 'recall@10': { selected: true, color: '#333333' } } },
  ];
  const { xAxis, series } = buildMultiBarSeries(exps);
  assert.deepStrictEqual(xAxis, ['A', 'B']);
  assert.deepStrictEqual(series.map((s) => s.name),
    ['recall@10-Clean', 'recall@10-Poisoned']);
  assert.strictEqual(series[0].data[0].value, 0.1);
  assert.strictEqual(series[0].data[0].itemStyle.color, '#111111');
  assert.strictEqual(series[1].data[1].value, 0.4);
});
