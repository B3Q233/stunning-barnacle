'use strict';
(function (global) {
  const parser = global.TPAVisualizer.parser;
  const transforms = global.TPAVisualizer.transforms;
  const state = { epochMetrics: null, comparison: null, lineSelected: [], barSelected: [] };
  let lineChart = null;
  let barChart = null;
  const $ = (id) => document.getElementById(id);

  function showMessage(text, isError) {
    $('message').textContent = text;
    $('message').className = isError ? 'message error' : 'message';
  }

  function readFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(new Error(`读取 ${file.name} 失败`));
      reader.readAsText(file, 'utf-8');
    });
  }

  async function handleFiles(fileList) {
    const errors = [];
    for (const file of fileList) {
      try {
        const text = await readFile(file);
        if (file.name === 'history.json') {
          state.epochMetrics = parser.extractEpochMetrics(text);
        } else if (file.name.endsWith('_comparison.json')) {
          state.comparison = parser.parseComparison(text);
        } else {
          errors.push(`跳过未知文件: ${file.name}`);
        }
      } catch (e) {
        errors.push(`${file.name}: ${e.message}`);
      }
    }
    renderAll();
    if (errors.length) showMessage(errors.join('；'), true);
  }

  function renderSummary() {
    if (!state.epochMetrics) {
      $('summary').textContent = '未加载数据';
      return;
    }
    const epochs = Object.keys(state.epochMetrics).length;
    $('summary').textContent =
      `epoch 数: ${epochs}；指标: ${parser.listMetrics(state.epochMetrics).join(', ')}`;
  }

  function renderLineCheckboxes() {
    const box = $('line-metrics');
    box.innerHTML = '';
    const metrics = state.epochMetrics ? parser.listMetrics(state.epochMetrics) : [];
    state.lineSelected = metrics.slice();
    for (const m of metrics) {
      const label = document.createElement('label');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = true;
      cb.addEventListener('change', () => {
        state.lineSelected = cb.checked
          ? [...new Set([...state.lineSelected, m])]
          : state.lineSelected.filter((x) => x !== m);
        renderLineChart();
      });
      label.appendChild(cb);
      label.appendChild(document.createTextNode(m));
      box.appendChild(label);
    }
  }

  function renderBarCheckboxes() {
    const box = $('bar-metrics');
    box.innerHTML = '';
    const items = state.comparison ? transforms.buildComparisonItems(state.comparison) : [];
    state.barSelected = items.map((it) => it.name);
    for (const it of items) {
      const label = document.createElement('label');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = true;
      cb.addEventListener('change', () => {
        state.barSelected = cb.checked
          ? [...new Set([...state.barSelected, it.name])]
          : state.barSelected.filter((x) => x !== it.name);
        renderBarChart();
      });
      label.appendChild(cb);
      label.appendChild(document.createTextNode(it.name));
      box.appendChild(label);
    }
  }

  function renderLineChart() {
    if (!state.epochMetrics) return;
    lineChart = lineChart || echarts.init($('chart-line'));
    const { xAxis, series } = transforms.buildLineSeries(state.epochMetrics, state.lineSelected);
    lineChart.setOption({
      title: { text: '每轮指标折线图' },
      tooltip: { trigger: 'axis' },
      legend: { data: series.map((s) => s.name) },
      grid: { left: 60, right: 30, bottom: 60 },
      xAxis: { type: 'category', data: xAxis, name: 'epoch' },
      yAxis: { type: 'value' },
      dataZoom: [{ type: 'inside' }, { type: 'slider' }],
      series,
    }, true);
  }

  function renderBarChart() {
    if (!state.comparison) {
      $('chart-bar').innerHTML = '<div class="placeholder">未提供 attack_comparison.json，无对比数据</div>';
      return;
    }
    barChart = barChart || echarts.init($('chart-bar'));
    const items = transforms.buildComparisonItems(state.comparison);
    const { xAxis, series } = transforms.buildComparisonSeries(items, state.barSelected);
    barChart.setOption({
      title: { text: 'Clean vs Poisoned 对比' },
      tooltip: { trigger: 'axis' },
      legend: { data: ['Clean', 'Poisoned'] },
      grid: { left: 60, right: 30, bottom: 60 },
      xAxis: { type: 'category', data: xAxis, axisLabel: { rotate: 20 } },
      yAxis: { type: 'value' },
      series,
    }, true);
  }

  function renderAll() {
    renderSummary();
    renderLineCheckboxes();
    renderBarCheckboxes();
    renderLineChart();
    renderBarChart();
  }

  function exportJson() {
    if (!state.epochMetrics) {
      showMessage('没有可导出的数据', true);
      return;
    }
    const blob = new Blob([JSON.stringify(state.epochMetrics, null, 2)],
      { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'epoch_metrics.json';
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function initApp() {
    $('history-input').addEventListener('change', (e) => handleFiles(e.target.files));
    $('comparison-input').addEventListener('change', (e) => handleFiles(e.target.files));
    $('export-json').addEventListener('click', exportJson);
    const drop = $('drop-zone');
    ['dragenter', 'dragover'].forEach((ev) =>
      drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('active'); }));
    ['dragleave', 'drop'].forEach((ev) =>
      drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('active'); }));
    drop.addEventListener('drop', (e) => handleFiles(e.dataTransfer.files));
    showMessage('请选择 history.json 与 attack_comparison.json');
  }

  global.TPAVisualizer = global.TPAVisualizer || {};
  global.TPAVisualizer.app = { initApp, handleFiles, exportJson };
})(window);
