'use strict';
(function (global) {
  const parser = global.TPAVisualizer.parser;
  const transforms = global.TPAVisualizer.transforms;
  const state = { tabs: [], activeTabId: null };
  let nextTabSeq = 1;
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

  function addTab(name) {
    const tab = {
      id: `tab-${Date.now()}-${nextTabSeq}`,
      name: name || `实验 ${nextTabSeq}`,
      checked: true,
      epochMetrics: null,
      comparison: null,
      metricOptions: {},
    };
    nextTabSeq += 1;
    state.tabs.push(tab);
    state.activeTabId = tab.id;
    return tab;
  }

  function activeTab() {
    return state.tabs.find((t) => t.id === state.activeTabId) || null;
  }

  function tabHasData(tab) {
    return !!(tab.epochMetrics || tab.comparison);
  }

  function buildMetricOptions(tab) {
    const names = [];
    if (tab.epochMetrics) {
      for (const m of parser.listMetrics(tab.epochMetrics)) {
        if (!names.includes(m)) names.push(m);
      }
    }
    if (tab.comparison) {
      for (const it of transforms.buildComparisonItems(tab.comparison)) {
        if (!names.includes(it.name)) names.push(it.name);
      }
    }
    const options = {};
    names.forEach((name, i) => {
      options[name] = { selected: true, color: transforms.colorFor(i) };
    });
    return options;
  }

  async function importFilesInto(tab, files) {
    const errors = [];
    for (const file of files) {
      try {
        const text = await readFile(file);
        if (file.name === 'history.json') {
          tab.epochMetrics = parser.extractEpochMetrics(text);
        } else if (file.name.endsWith('_comparison.json')) {
          tab.comparison = parser.parseComparison(text);
        } else {
          errors.push(`跳过未知文件: ${file.name}`);
        }
      } catch (e) {
        errors.push(`${file.name}: ${e.message}`);
      }
    }
    tab.metricOptions = buildMetricOptions(tab);
    if (errors.length) showMessage(errors.join('；'), true);
  }

  async function handleDirectoryImport(fileList) {
    const files = [...fileList];
    if (!files.length) return;
    const info = parser.parseDirectoryPath(files[0].webkitRelativePath || files[0].name);
    let tab = activeTab();
    if (!tab || tabHasData(tab)) tab = addTab(`实验 ${nextTabSeq}`);
    const autoName = parser.buildAutoName(info);
    if (autoName && !tabHasData(tab)) tab.name = autoName;
    await importFilesInto(tab, files);
    renderAll();
  }

  async function handleManualImport(fileList) {
    const files = [...fileList];
    if (!files.length) return;
    let tab = activeTab();
    if (!tab || tabHasData(tab)) {
      tab = addTab(prompt('选项卡名称', `实验 ${nextTabSeq}`) || `实验 ${nextTabSeq}`);
    }
    await importFilesInto(tab, files);
    renderAll();
  }

  function makeBtn(text, title, onClick) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = text;
    b.title = title;
    b.addEventListener('click', onClick);
    return b;
  }

  function renderTabs() {
    const ul = $('tab-list');
    ul.innerHTML = '';
    for (const tab of state.tabs) {
      const li = document.createElement('li');
      li.className = 'tab-item' + (tab.id === state.activeTabId ? ' active' : '');
      const check = document.createElement('input');
      check.type = 'checkbox';
      check.checked = tab.checked;
      check.title = '是否参与渲染';
      check.addEventListener('change', () => {
        tab.checked = check.checked;
        renderAll();
      });
      const nameSpan = document.createElement('span');
      nameSpan.className = 'tab-name';
      nameSpan.textContent = tab.name;
      nameSpan.title = '点击设为当前选项卡';
      nameSpan.addEventListener('click', () => {
        state.activeTabId = tab.id;
        renderAll();
      });
      const btns = document.createElement('span');
      btns.className = 'tab-btns';
      btns.appendChild(makeBtn('✎', '改名', () => {
        const n = prompt('修改名称', tab.name);
        if (n && n.trim()) {
          tab.name = n.trim();
          renderAll();
        }
      }));
      btns.appendChild(makeBtn('↺H', '重置 history', () => {
        tab.epochMetrics = null;
        tab.metricOptions = buildMetricOptions(tab);
        renderAll();
      }));
      btns.appendChild(makeBtn('↺C', '重置 comparison', () => {
        tab.comparison = null;
        tab.metricOptions = buildMetricOptions(tab);
        renderAll();
      }));
      btns.appendChild(makeBtn('×', '删除选项卡', () => {
        if (confirm(`删除选项卡「${tab.name}」？`)) {
          state.tabs = state.tabs.filter((t) => t.id !== tab.id);
          if (state.activeTabId === tab.id) {
            state.activeTabId = state.tabs.length ? state.tabs[0].id : null;
          }
          renderAll();
        }
      }));
      li.append(check, nameSpan, btns);
      ul.appendChild(li);
    }
  }

  function renderCard() {
    const card = $('metric-card');
    card.innerHTML = '';
    const tabs = state.tabs.filter((t) => t.checked && tabHasData(t));
    if (!tabs.length) {
      card.innerHTML = '<div class="placeholder">勾选左侧选项卡以显示指标选项</div>';
      return;
    }
    for (const tab of tabs) {
      const section = document.createElement('div');
      section.className = 'exp-section';
      const h = document.createElement('h3');
      h.textContent = tab.name;
      section.appendChild(h);
      for (const [name, opt] of Object.entries(tab.metricOptions)) {
        const row = document.createElement('label');
        row.className = 'metric-row';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = opt.selected;
        cb.addEventListener('change', () => {
          opt.selected = cb.checked;
          renderCharts();
        });
        const color = document.createElement('input');
        color.type = 'color';
        color.value = opt.color;
        color.addEventListener('input', () => {
          opt.color = color.value;
          renderCharts();
        });
        const span = document.createElement('span');
        span.textContent = name;
        row.append(cb, color, span);
        section.appendChild(row);
      }
      card.appendChild(section);
    }
  }

  function renderLineChart(tabs) {
    const withHistory = tabs.filter((t) => t.epochMetrics);
    if (!withHistory.length) {
      $('chart-line').innerHTML = '<div class="placeholder">勾选含 history 数据的选项卡</div>';
      return;
    }
    lineChart = lineChart || echarts.init($('chart-line'));
    const { xAxis, series } = transforms.buildMultiLineSeries(withHistory);
    lineChart.setOption({
      title: { text: '每轮指标折线图' },
      tooltip: { trigger: 'axis' },
      legend: { type: 'scroll', data: series.map((s) => s.name) },
      grid: { left: 60, right: 30, bottom: 60 },
      xAxis: { type: 'category', data: xAxis, name: 'epoch' },
      yAxis: { type: 'value' },
      dataZoom: [{ type: 'inside' }, { type: 'slider' }],
      series,
    }, true);
  }

  function renderBarChart(tabs) {
    const withCmp = tabs.filter((t) => t.comparison);
    if (!withCmp.length) {
      $('chart-bar').innerHTML = '<div class="placeholder">勾选含 comparison 数据的选项卡</div>';
      return;
    }
    barChart = barChart || echarts.init($('chart-bar'));
    const { xAxis, series } = transforms.buildMultiBarSeries(withCmp);
    barChart.setOption({
      title: { text: 'Clean vs Poisoned 对比' },
      tooltip: { trigger: 'axis' },
      legend: { type: 'scroll', data: series.map((s) => s.name) },
      grid: { left: 60, right: 30, bottom: 60 },
      xAxis: { type: 'category', data: xAxis, axisLabel: { interval: 0 } },
      yAxis: { type: 'value' },
      series,
    }, true);
  }

  function renderCharts() {
    const tabs = state.tabs.filter((t) => t.checked && tabHasData(t));
    renderLineChart(tabs);
    renderBarChart(tabs);
  }

  function renderAll() {
    renderTabs();
    renderCard();
    renderCharts();
  }

  function exportJson() {
    const tab = activeTab();
    if (!tab || !tab.epochMetrics) {
      showMessage('当前选项卡没有 history 数据', true);
      return;
    }
    const blob = new Blob([JSON.stringify(tab.epochMetrics, null, 2)],
      { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${tab.name}-epoch_metrics.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function initApp() {
    $('new-tab').addEventListener('click', () => {
      addTab(prompt('选项卡名称', `实验 ${nextTabSeq}`) || `实验 ${nextTabSeq}`);
      renderAll();
    });
    $('btn-dir').addEventListener('click', () => $('dir-input').click());
    $('dir-input').addEventListener('change', (e) => handleDirectoryImport(e.target.files));
    $('btn-import').addEventListener('click', () => $('import-input').click());
    $('import-input').addEventListener('change', (e) => handleManualImport(e.target.files));
    $('export-json').addEventListener('click', exportJson);
    const drop = $('drop-zone');
    ['dragenter', 'dragover'].forEach((ev) =>
      drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('active'); }));
    ['dragleave', 'drop'].forEach((ev) =>
      drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('active'); }));
    drop.addEventListener('drop', (e) => handleManualImport(e.dataTransfer.files));
    showMessage('请先新建选项卡，或直接导入实验目录/文件');
  }

  global.TPAVisualizer = global.TPAVisualizer || {};
  global.TPAVisualizer.app = { initApp, handleDirectoryImport, handleManualImport, exportJson };
})(window);
