const test = require('node:test');
const assert = require('node:assert');
const { importFiles, loadState, STORAGE_KEY } = require('../js/app.js');

test('批量导入：父目录下多个实验被分组识别', () => {
  const entries = [
    {
      relativePath: 'attacks/pgd/outputs/ml100k/lightgcn/2026-08-09-21-54/history.json',
      name: 'history.json',
      text: '{"history":[]}',
    },
    {
      relativePath: 'attacks/pgd/outputs/ml100k/lightgcn/2026-08-09-21-54/config.yaml',
      name: 'config.yaml',
      text: 'dataset: ml100k\nattack:\n  name: pgd\n',
    },
    {
      relativePath: 'attacks/tpa/outputs/ml100k/lightgcn/2026-08-09-22-30/history.json',
      name: 'history.json',
      text: '{"history":[]}',
    },
  ];
  const { experiments, errors } = importFiles(entries);
  assert.strictEqual(errors.length, 0);
  assert.strictEqual(experiments.length, 2);
  assert.ok(experiments.some((e) => e.method === 'pgd'));
  assert.ok(experiments.some((e) => e.method === 'tpa'));
});

test('损坏文件被跳过并记录错误', () => {
  const { experiments, errors } = importFiles([
    {
      relativePath: 'attacks/pgd/outputs/ml100k/lightgcn/2026-08-09-21-54/history.json',
      name: 'history.json',
      text: '{oops',
    },
  ]);
  assert.strictEqual(experiments.length, 0);
  assert.strictEqual(errors.length, 1);
  assert.match(errors[0], /解析失败/);
});

test('无 localStorage 时 loadState 返回 null', () => {
  assert.strictEqual(loadState(), null);
});

test('STORAGE_KEY 固定', () => {
  assert.strictEqual(STORAGE_KEY, 'tpa.visualizer.v1');
});
