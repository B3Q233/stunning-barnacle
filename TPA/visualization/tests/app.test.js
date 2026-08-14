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

test('导入 outputs 根目录：过滤非实验文件并按 latest.json 归并', () => {
  const entries = [
    { relativePath: 'outputs/history.json', name: 'history.json', text: '{"history":[]}' },
    { relativePath: 'outputs/eval_log.csv', name: 'eval_log.csv', text: 'epoch,recall@20,ndcg@20\n1,0.1,0.2\n' },
    { relativePath: 'outputs/latest.json', name: 'latest.json', text: '{"run_tag":"2026-08-09-23-39"}' },
    { relativePath: 'outputs/plot_results.py', name: 'plot_results.py', text: 'x=1' },
    { relativePath: 'outputs/checkpoints/latest.pt', name: 'latest.pt', text: 'bin' },
    { relativePath: 'outputs/2026-08-09-21-39/history.json', name: 'history.json', text: '{"history":[]}' },
    {
      relativePath: 'outputs/2026-08-09-21-39/config.yaml',
      name: 'config.yaml',
      text: 'dataset: ml100k\ncheckpoint_dir: models/lightgcn/outputs/checkpoints\n',
    },
    { relativePath: 'outputs/2026-08-09-23-39/history.json', name: 'history.json', text: '{"history":[]}' },
    {
      relativePath: 'outputs/2026-08-09-23-39/config.yaml',
      name: 'config.yaml',
      text: 'dataset: ml100k\ncheckpoint_dir: models/lightgcn/outputs/checkpoints\n',
    },
  ];
  const { experiments, errors } = importFiles(entries);
  assert.strictEqual(errors.length, 0);
  assert.strictEqual(experiments.length, 2);
  const tags = experiments.map((e) => e.runTag).sort();
  assert.deepStrictEqual(tags, ['2026-08-09-21-39', '2026-08-09-23-39']);
  assert.ok(experiments.every((e) => e.kind !== 'custom'));
});

test('仅含 surrogate_meta.json 的目录不生成实验', () => {
  const { experiments } = importFiles([
    {
      relativePath: 'outputs/surrogate/surrogate_meta.json',
      name: 'surrogate_meta.json',
      text: '{"model":"lightgcn"}',
    },
  ]);
  assert.strictEqual(experiments.length, 0);
});
