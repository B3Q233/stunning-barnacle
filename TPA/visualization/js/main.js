/* TPA 可视化页面装配：导入、列表、编辑、渲染、导出（浏览器端） */
(function () {
  'use strict';

  const parser = window.TPAVisualizer.parser;
  const transforms = window.TPAVisualizer.transforms;
  const app = window.TPAVisualizer.app;

  const state = {
    experiments: [],
    selected: new Set(),
    metric: null,
    chartType: 'line',
    title: 'TPA 实验数据可视化',
  };

  const $ = (id) => document.getElementById(id);
  const COMPARE_TARGET_METRICS = ['hr@k', 'ndcg@k'];

  function showMessage(text, isError) {
    const el = $('message');
    el.textContent = text;
    el.className = 'message' + (isError ? ' error' : '');
    clearTimeout(showMessage._timer);
    showMessage._timer = setTimeout(() => {
      el.textContent = '';
      el.className = 'message';
    }, 5000);
  }

  function persist() {
    app.saveState(state);
  }

  // ---- 导入 ----
  async function collectFiles(fileList) {
    const entries = [];
    for (const file of fileList) {
      if (!file || !file.name) continue;
      const relativePath = file.webkitRelativePath || file.name;
      const text = await file.text();
      entries.push({ relativePath, name: file.name, text });
    }
    return entries;
  }

  async function handleImport(fileList) {
    try {
      const entries = await collectFiles(fileList);
      if (!entries.length) {
        showMessage('未选择任何文件');
        return;
      }
      const { experiments, errors } = app.importFiles(entries);
      let added = 0;
      for (const exp of transforms.assignPalette(experiments)) {
        if (state.experiments.some((e) => e.id === exp.id)) continue;
        state.experiments.push(exp);
        state.selected.add(exp.id);
        added += 1;
      }
      if (errors.length) {
        showMessage(`导入 ${added} 个实验，${errors.length} 个失败（${errors[0]}）`, true);
      } else if (added) {
        showMessage(`导入 ${added} 个实验`);
      } else {
        showMessage('未发现新的可识别实验');
      }
      refreshMetrics();
      renderList();
      renderChart();
      persist();
    } catch (e) {
      showMessage(`导入失败：${e.message}`, true);
    }
  }

  // ---- 指标选择 ----
  function refreshMetrics() {
    const select = $('metric-select');
    const all = transforms.listMetrics(state.experiments)
      .filter((m) => m !== 'epoch');
    const options = new Set(all);
    if (state.chartType === 'compare') {
      COMPARE_TARGET_METRICS.forEach((m) => options.add(m));
    }
    select.innerHTML = '';
    for (const m of options) {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m;
      select.appendChild(opt);
    }
    if (state.metric && options.has(state.metric)) {
      select.value = state.metric;
    } else {
      state.metric = select.value || null;
    }
    $('chart-hint').textContent = state.chartType === 'compare'
      ? '对比图支持模型效用指标与目标物品平均 HR/NDCG'
      : '';
  }

  // ---- 实验列表 ----
  function renderList() {
    const ul = $('experiment-list');
    ul.innerHTML = '';
    for (const exp of state.experiments) {
      const li = document.createElement('li');
      li.dataset.id = exp.id;

      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.checked = state.selected.has(exp.id);
      checkbox.title = '显示/隐藏';
      checkbox.addEventListener('change', () => {
        if (checkbox.checked) state.selected.add(exp.id);
        else state.selected.delete(exp.id);
        renderChart();
        persist();
      });

      const dot = document.createElement('span');
      dot.className = 'color-dot';
      dot.style.background = exp.color;

      const colorInput = document.createElement('input');
      colorInput.type = 'color';
      colorInput.className = 'color-input';
      colorInput.value = exp.color || '#5470c6';
      colorInput.title = '修改颜色';
      colorInput.addEventListener('input', () => {
        exp.color = colorInput.value;
        dot.style.background = exp.color;
        renderChart();
        persist();
      });

      const labelInput = document.createElement('input');
      labelInput.className = 'label-input';
      labelInput.value = exp.label;
      labelInput.title = '编辑名称';
      labelInput.addEventListener('change', () => {
        exp.label = labelInput.value || exp.label;
        labelInput.value = exp.label;
        renderChart();
        persist();
      });

      const remove = document.createElement('button');
      remove.className = 'remove-btn';
      remove.textContent = '✕';
      remove.title = '删除该实验';
      remove.addEventListener('click', () => {
        state.experiments = state.experiments.filter((e) => e.id !== exp.id);
        state.selected.delete(exp.id);
        refreshMetrics();
        renderList();
        renderChart();
        persist();
      });

      li.append(checkbox, dot, colorInput, labelInput, remove);
      ul.appendChild(li);
    }
  }

  // ---- 图表渲染 ----
  function chartInstance() {
    const el = $('chart');
    return echarts.getInstanceByDom(el) || echarts.init(el);
  }

  function buildOption(exps) {
    if (state.chartType === 'line') {
      const { xAxis, series } = transforms.buildLineSeries(exps, state.metric);
      return {
        title: { text: state.title },
        tooltip: { trigger: 'axis' },
        legend: { type: 'scroll' },
        xAxis: { type: 'category', name: 'epoch', data: xAxis },
        yAxis: { type: 'value', name: state.metric },
        series,
        dataZoom: [{ type: 'inside' }, { type: 'slider' }],
      };
    }
    if (state.chartType === 'bar') {
      const metrics = transforms.listMetrics(exps)
        .filter((m) => m !== 'epoch' && !m.startsWith('target_'));
      const { xAxis, series } = transforms.buildMetricBars(exps, metrics);
      return {
        title: { text: state.title },
        tooltip: { trigger: 'axis' },
        legend: { type: 'scroll' },
        xAxis: { type: 'category', data: xAxis, axisLabel: { rotate: 20 } },
        yAxis: { type: 'value' },
        series,
      };
    }
    const mode = COMPARE_TARGET_METRICS.includes(state.metric) ? 'target' : 'model';
    const { xAxis, series } = transforms.buildComparisonSeries(
      exps, state.metric, mode,
    );
    return {
      title: { text: state.title },
      tooltip: { trigger: 'axis' },
      legend: {},
      xAxis: { type: 'category', data: xAxis, axisLabel: { rotate: 20 } },
      yAxis: { type: 'value', name: state.metric },
      series,
    };
  }

  function renderChart() {
    let exps = transforms.assignPalette(
      state.experiments.filter((e) => state.selected.has(e.id)),
    );
    const chart = chartInstance();
    if (!exps.length) {
      chart.clear();
      chart.setOption({
        title: { text: state.title, subtext: '请先导入并勾选实验' },
      });
      return;
    }
    if (state.chartType === 'compare' && !exps.some((e) => e.comparison)) {
      state.chartType = 'line';
      $('chart-type').value = 'line';
      $('chart-hint').textContent = '所选实验缺少 *_comparison.json，已切换为折线图';
      refreshMetrics();
      exps = transforms.assignPalette(
        state.experiments.filter((e) => state.selected.has(e.id)),
      );
    }
    try {
      chart.setOption(buildOption(exps), true);
    } catch (e) {
      showMessage(`图表渲染失败：${e.message}`, true);
    }
  }

  function clearAll() {
    state.experiments = [];
    state.selected = new Set();
    state.metric = null;
    state.chartType = 'line';
    state.title = 'TPA 实验数据可视化';
    try {
      localStorage.removeItem(app.STORAGE_KEY);
    } catch (e) { /* 忽略 */ }
    $('title-input').value = state.title;
    $('chart-type').value = 'line';
    refreshMetrics();
    renderList();
    renderChart();
    showMessage('已清空全部数据与本地缓存');
  }

  // ---- 导出图片 ----
  function exportChart() {
    const chart = echarts.getInstanceByDom($('chart'));
    if (!chart) {
      showMessage('当前无图表可导出');
      return;
    }
    const url = chart.getDataURL({
      type: 'png', pixelRatio: 2, backgroundColor: '#fff',
    });
    const a = document.createElement('a');
    a.href = url;
    a.download = transforms.buildExportFilename(state.chartType, state.metric);
    document.body.appendChild(a);
    a.click();
    a.remove();
    showMessage(`已导出 ${a.download}`);
  }

  // ---- 快照导出/导入 ----
  function exportSnapshot() {
    const payload = {
      version: 1,
      savedAt: new Date().toISOString(),
      experiments: state.experiments,
      selected: [...state.selected],
      metric: state.metric,
      chartType: state.chartType,
      title: state.title,
    };
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob(
      [JSON.stringify(payload, null, 2)],
      { type: 'application/json' },
    ));
    a.download = 'tpa-visualizer-snapshot.json';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
  }

  function applySnapshot(payload) {
    if (!payload || !Array.isArray(payload.experiments)) {
      throw new Error('快照格式不正确');
    }
    state.experiments = payload.experiments.map((e) => ({ ...e }));
    state.selected = new Set(payload.selected || []);
    state.metric = payload.metric || null;
    state.chartType = payload.chartType || 'line';
    state.title = payload.title || 'TPA 实验数据可视化';
    $('title-input').value = state.title;
    $('chart-type').value = state.chartType;
    refreshMetrics();
    renderList();
    renderChart();
    persist();
  }

  async function handleSnapshotImport(file) {
    try {
      const text = await file.text();
      applySnapshot(JSON.parse(text));
      showMessage('快照已恢复');
    } catch (e) {
      showMessage(`快照导入失败：${e.message}`, true);
    }
  }

  // ---- 恢复本地状态 ----
  function restoreState() {
    const saved = app.loadState();
    if (saved && Array.isArray(saved.experiments)) {
      state.experiments = saved.experiments;
      state.selected = new Set(saved.selected || []);
      state.metric = saved.metric || null;
      state.chartType = saved.chartType || 'line';
      state.title = saved.title || 'TPA 实验数据可视化';
      $('title-input').value = state.title;
      $('chart-type').value = state.chartType;
    }
  }

  // ---- 事件绑定 ----
  function init() {
    restoreState();
    refreshMetrics();
    renderList();
    renderChart();

    $('btn-import').addEventListener('click', () => $('file-input').click());
    $('file-input').addEventListener('change', (e) => {
      handleImport(e.target.files);
      e.target.value = '';
    });
    $('btn-export').addEventListener('click', exportChart);
    $('btn-export-snapshot').addEventListener('click', exportSnapshot);
    $('btn-load-snapshot').addEventListener('click', () => $('snapshot-input').click());
    $('btn-clear').addEventListener('click', clearAll);
    $('snapshot-input').addEventListener('change', (e) => {
      const file = e.target.files && e.target.files[0];
      if (file) handleSnapshotImport(file);
      e.target.value = '';
    });
    $('chart-type').addEventListener('change', () => {
      state.chartType = $('chart-type').value;
      refreshMetrics();
      renderChart();
      persist();
    });
    $('metric-select').addEventListener('change', () => {
      state.metric = $('metric-select').value;
      renderChart();
      persist();
    });
    $('title-input').addEventListener('change', () => {
      state.title = $('title-input').value || state.title;
      renderChart();
      persist();
    });

    const dropZone = $('drop-zone');
    ['dragenter', 'dragover'].forEach((evt) => {
      dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
      });
    });
    ['dragleave', 'drop'].forEach((evt) => {
      dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
      });
    });
    dropZone.addEventListener('drop', (e) => {
      if (e.dataTransfer && e.dataTransfer.files) {
        handleImport(e.dataTransfer.files);
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
