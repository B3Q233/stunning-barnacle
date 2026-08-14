const test = require('node:test');
const assert = require('node:assert');
const {
  formatRunTag, parseExperiment, parseHistoryJson, parseBest, parseEvalCsv,
  parseConfigYaml, parseComparisonJson, buildLabel,
} = require('../js/parser.js');

test('formatRunTag 转为中文日期时间', () => {
  assert.strictEqual(formatRunTag('2026-08-09-21-54'), '2026年08月09日21时54分');
});

test('识别攻击实验路径', () => {
  const exp = parseExperiment(
    ['attacks', 'pgd', 'outputs', 'ml100k', 'lightgcn', '2026-08-09-21-54'],
    [],
  );
  assert.strictEqual(exp.kind, 'attack');
  assert.strictEqual(exp.method, 'pgd');
  assert.strictEqual(exp.dataset, 'ml100k');
  assert.strictEqual(exp.model, 'lightgcn');
  assert.strictEqual(exp.runTag, '2026-08-09-21-54');
  assert.strictEqual(
    exp.label, 'attack-pgd-ml100k-lightgcn-2026年08月09日21时54分',
  );
});

test('识别模型训练路径并从 config 补全 dataset', () => {
  const exp = parseExperiment(
    ['models', 'lightgcn', 'outputs', '2026-08-09-23-01'],
    [{ name: 'config.yaml', text: 'dataset: ml100k\nmodel:\n  name: lightgcn\n' }],
  );
  assert.strictEqual(exp.kind, 'model');
  assert.strictEqual(exp.dataset, 'ml100k');
  assert.strictEqual(exp.runTag, '2026-08-09-23-01');
});

test('无法识别时按 custom 处理', () => {
  const exp = parseExperiment(['my_data'], [{ name: 'history.json', text: '[]' }]);
  assert.strictEqual(exp.kind, 'custom');
  assert.strictEqual(exp.label, '自定义-my_data');
});

test('解析 history.json', () => {
  const rows = parseHistoryJson(
    JSON.stringify({ history: [{ epoch: 1, train_loss: 0.5, val_loss: 0.4 }] }),
  );
  assert.deepStrictEqual(rows, [{ epoch: 1, train_loss: 0.5, val_loss: 0.4 }]);
});

test('解析 history.json 的 best 字段', () => {
  const best = parseBest(
    JSON.stringify({ best: { 'recall@10': { epoch: 5, value: 0.1 } } }),
  );
  assert.strictEqual(best['recall@10'].value, 0.1);
});

test('解析 eval_log.csv', () => {
  const rows = parseEvalCsv('epoch,recall@10,ndcg@10\n1,0.1,0.2\n2,0.11,0.21\n');
  assert.deepStrictEqual(rows, [
    { epoch: 1, 'recall@10': 0.1, 'ndcg@10': 0.2 },
    { epoch: 2, 'recall@10': 0.11, 'ndcg@10': 0.21 },
  ]);
});

test('解析 config.yaml 快照关键字段', () => {
  const cfg = parseConfigYaml(
    'dataset: ml100k\nmodel:\n  name: lightgcn\nattack:\n  name: pgd\n'
    + 'training:\n  epochs: 30\n'
    + 'evaluation:\n  metrics:\n  - recall@10: upper\n  - ndcg@10: upper\n',
  );
  assert.strictEqual(cfg.dataset, 'ml100k');
  assert.strictEqual(cfg.model, 'lightgcn');
  assert.strictEqual(cfg.attack, 'pgd');
  assert.strictEqual(cfg.epochs, 30);
  assert.deepStrictEqual(cfg.metrics, ['recall@10', 'ndcg@10']);
});

test('解析 *_comparison.json', () => {
  const comp = parseComparisonJson(
    JSON.stringify({ model_utility: { clean: { 'recall@10': 0.1 } }, target_metrics: {} }),
  );
  assert.strictEqual(comp.model_utility.clean['recall@10'], 0.1);
});

test('坏 JSON 抛出可读错误', () => {
  assert.throws(() => parseHistoryJson('{oops'), /解析失败/);
});
