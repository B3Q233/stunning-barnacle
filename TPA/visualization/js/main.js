'use strict';
(function (global) {
  const parser = global.TPAVisualizer.parser;
  const transforms = global.TPAVisualizer.transforms;
  const state = { cards: [], activeCardId: null };
  let nextCardSeq = 1;
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

  function openModal(title, opts) {
    const o = opts || {};
    const overlay = $('modal-overlay');
    $('modal-title').textContent = title;
    const body = $('modal-body');
    body.innerHTML = '';
    let inputEl = null;
    if (o.input) {
      inputEl = document.createElement('input');
      inputEl.type = 'text';
      inputEl.value = o.inputValue || '';
      body.appendChild(inputEl);
    }
    if (o.extra) {
      const sep = document.createElement('div');
      sep.className = 'sep';
      sep.textContent = '或';
      body.appendChild(sep);
      const extraBtn = document.createElement('button');
      extraBtn.textContent = o.extraLabel || '导入实验路径';
      extraBtn.addEventListener('click', () => o.onExtra && o.onExtra(close));
      body.appendChild(extraBtn);
    }
    const ok = $('modal-ok');
    ok.textContent = o.okText || '确定';
    const cancel = $('modal-cancel');
    overlay.classList.remove('hidden');

    function close() {
      overlay.classList.add('hidden');
      ok.removeEventListener('click', confirmModal);
      cancel.removeEventListener('click', close);
      if (inputEl) inputEl.removeEventListener('keydown', onKey);
    }
    function onKey(e) {
      if (e.key === 'Enter') confirmModal();
    }
    function confirmModal() {
      const value = inputEl ? inputEl.value.trim() : null;
      close();
      o.onOk && o.onOk(value);
    }
    ok.addEventListener('click', confirmModal);
    cancel.addEventListener('click', close);
    if (inputEl) {
      inputEl.addEventListener('keydown', onKey);
      inputEl.focus();
    }
  }

  function addCard(name) {
    const card = {
      id: `card-${Date.now()}-${nextCardSeq}`,
      name: name || `实验 ${nextCardSeq}`,
      checked: true,
      epochMetrics: null,
      comparison: null,
      lineOptions: {},
      barOptions: {},
    };
    nextCardSeq += 1;
    state.cards.push(card);
    state.activeCardId = card.id;
    return card;
  }

  function activeCard() {
    return state.cards.find((c) => c.id === state.activeCardId) || null;
  }

  function cardHasData(card) {
    return !!(card.epochMetrics || card.comparison);
  }

  function isDefaultName(card) {
    return /^实验 \d+$/.test(card.name);
  }

  function buildOptions(card) {
    const nextLine = {};
    if (card.epochMetrics) {
      parser.listMetrics(card.epochMetrics).forEach((name, i) => {
        nextLine[name] = card.lineOptions[name]
          || { selected: true, color: transforms.colorFor(i) };
      });
    }
    card.lineOptions = nextLine;
    const nextBar = {};
    if (card.comparison) {
      transforms.buildComparisonItems(card.comparison).forEach((it, i) => {
        nextBar[it.name] = card.barOptions[it.name]
          || { selected: true, color: transforms.colorFor(i) };
      });
    }
    card.barOptions = nextBar;
  }

  function applyFileToCard(card, file, text) {
    if (file.name === 'history.json') {
      card.epochMetrics = parser.extractEpochMetrics(text);
    } else if (file.name.endsWith('_comparison.json')) {
      card.comparison = parser.parseComparison(text);
    } else {
      throw new Error(`跳过未知文件: ${file.name}`);
    }
  }

  async function importDirIntoCard(card, fileList) {
    const files = [...fileList];
    if (!files.length) return;
    const info = parser.parseDirectoryPath(files[0].webkitRelativePath || files[0].name);
    const autoName = parser.buildAutoName(info);
    if (autoName && isDefaultName(card)) card.name = autoName;
    const errors = [];
    for (const file of files) {
      try {
        applyFileToCard(card, file, await readFile(file));
      } catch (e) {
        errors.push(`${file.name}: ${e.message}`);
      }
    }
    buildOptions(card);
    if (errors.length) showMessage(errors.join('；'), true);
    renderAll();
  }

  async function importFilesIntoCard(card, fileList) {
    const files = [...fileList];
    if (!files.length) return;
    const errors = [];
    for (const file of files) {
      try {
        applyFileToCard(card, file, await readFile(file));
      } catch (e) {
        errors.push(`${file.name}: ${e.message}`);
      }
    }
    buildOptions(card);
    if (errors.length) showMessage(errors.join('；'), true);
    renderAll();
  }

  function renderCards() {
    const ul = $('card-list');
    ul.innerHTML = '';
    for (const card of state.cards) {
      const li = document.createElement('li');
      li.className = 'card-item' + (card.id === state.activeCardId ? ' active' : '');
      const check = document.createElement('input');
      check.type = 'checkbox';
      check.checked = card.checked;
      check.title = '是否参与渲染';
      check.addEventListener('change', () => {
        card.checked = check.checked;
        renderAll();
      });
      const name = document.createElement('span');
      name.className = 'ci-name';
      name.textContent = card.name;
      name.title = '查看该实验卡';
      const status = document.createElement('span');
      status.className = 'ci-status';
      status.appendChild(badge(`H${card.epochMetrics ? '✓' : '—'}`, !!card.epochMetrics));
      status.appendChild(badge(`C${card.comparison ? '✓' : '—'}`, !!card.comparison));
      li.addEventListener('click', (e) => {
        if (e.target !== check) {
          state.activeCardId = card.id;
          renderAll();
        }
      });
      li.append(check, name, status);
      ul.appendChild(li);
    }
  }

  function renderPanel() {
    const panel = $('card-panel');
    const card = activeCard();
    if (!card) {
      panel.innerHTML = '<div class="placeholder">点击「＋ 添加实验卡」开始</div>';
      return;
    }
    const header = document.createElement('div');
    header.className = 'panel-header';
    const name = document.createElement('span');
    name.className = 'panel-name';
    name.textContent = card.name;
    const status = document.createElement('span');
    status.className = 'panel-status';
    status.appendChild(badge(`history ${card.epochMetrics ? '✓' : '—'}`, !!card.epochMetrics));
    status.appendChild(badge(`comparison ${card.comparison ? '✓' : '—'}`, !!card.comparison));
    const actions = document.createElement('span');
    actions.className = 'panel-actions';
    actions.appendChild(btn('改名', () => {
      openModal('修改实验卡名称', {
        input: true, inputValue: card.name, onOk: (v) => {
          if (v) {
            card.name = v;
            renderAll();
          }
        },
      });
    }));
    actions.appendChild(btn('重置 H', () => {
      card.epochMetrics = null;
      buildOptions(card);
      renderAll();
    }));
    actions.appendChild(btn('重置 C', () => {
      card.comparison = null;
      buildOptions(card);
      renderAll();
    }));
    actions.appendChild(btn('删除', () => {
      openModal(`删除实验卡「${card.name}」？`, {
        okText: '删除', onOk: () => {
          state.cards = state.cards.filter((c) => c.id !== card.id);
          if (state.activeCardId === card.id) {
            state.activeCardId = state.cards.length ? state.cards[0].id : null;
          }
          renderAll();
        },
      });
    }, true));
    header.append(name, status, actions);

    const body = document.createElement('div');
    body.className = 'panel-body';
    body.appendChild(buildImportArea(card));
    if (cardHasData(card)) {
      if (card.epochMetrics) {
        body.appendChild(metricGroup('history 指标', card.lineOptions, renderCharts));
      }
      if (card.comparison) {
        body.appendChild(metricGroup('comparison 对比项', card.barOptions, renderCharts));
      }
    }
    panel.innerHTML = '';
    panel.append(header, body);
  }

  function buildImportArea(card) {
    const area = document.createElement('div');
    area.className = 'import-area';
    const dirBtn = btn('导入实验路径', () => {
      const input = document.createElement('input');
      input.type = 'file';
      input.webkitdirectory = true;
      input.multiple = true;
      input.addEventListener('change', (e) => importDirIntoCard(card, e.target.files));
      input.click();
    });
    const histLabel = document.createElement('label');
    histLabel.append('history.json ');
    const histInput = document.createElement('input');
    histInput.type = 'file';
    histInput.accept = '.json';
    histInput.addEventListener('change', (e) => importFilesIntoCard(card, e.target.files));
    histLabel.appendChild(histInput);
    const cmpLabel = document.createElement('label');
    cmpLabel.append('comparison ');
    const cmpInput = document.createElement('input');
    cmpInput.type = 'file';
    cmpInput.accept = '.json';
    cmpInput.addEventListener('change', (e) => importFilesIntoCard(card, e.target.files));
    cmpLabel.appendChild(cmpInput);
    area.append(dirBtn, histLabel, cmpLabel);
    return area;
  }

  function metricGroup(title, options, onChange) {
    const group = document.createElement('div');
    group.className = 'metric-group';
    const h = document.createElement('h4');
    h.textContent = title;
    group.appendChild(h);
    const wrap = document.createElement('div');
    wrap.className = 'metric-rows';
    for (const [name, opt] of Object.entries(options)) {
      const row = document.createElement('label');
      row.className = 'metric-row';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = opt.selected;
      cb.addEventListener('change', () => {
        opt.selected = cb.checked;
        onChange();
      });
      const color = document.createElement('input');
      color.type = 'color';
      color.value = opt.color;
      color.addEventListener('input', () => {
        opt.color = color.value;
        onChange();
      });
      const span = document.createElement('span');
      span.textContent = name;
      row.append(cb, color, span);
      wrap.appendChild(row);
    }
    group.appendChild(wrap);
    return group;
  }

  function btn(text, onClick, danger) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = text;
    if (danger) b.className = 'btn-danger';
    b.addEventListener('click', onClick);
    return b;
  }

  function badge(text, ok) {
    const s = document.createElement('span');
    s.className = 'badge' + (ok ? ' ok' : '');
    s.textContent = text;
    return s;
  }

  function showPlaceholder(id, text) {
    const dom = $(id);
    const inst = echarts.getInstanceByDom(dom);
    if (inst) inst.dispose();
    dom.innerHTML = `<div class="placeholder">${text}</div>`;
  }

  function chartInstance(id) {
    const dom = $(id);
    return echarts.getInstanceByDom(dom) || echarts.init(dom);
  }

  function chartBase(title) {
    return {
      title: { text: title, top: 10, left: 'center', textStyle: { fontSize: 14 } },
      tooltip: { trigger: 'axis' },
      legend: { type: 'scroll', top: 44, left: 'center' },
      grid: { top: 96, left: 70, right: 30, bottom: 60 },
    };
  }

  function renderLineChart(cards) {
    const withHistory = cards.filter((c) => c.epochMetrics);
    if (!withHistory.length) {
      showPlaceholder('chart-line', '勾选含 history 数据的实验卡');
      return;
    }
    const { xAxis, series } = transforms.buildMultiLineSeries(withHistory);
    const option = chartBase('每轮指标折线图');
    option.xAxis = { type: 'category', data: xAxis, name: 'epoch' };
    option.yAxis = { type: 'value' };
    option.dataZoom = [{ type: 'inside' }, { type: 'slider' }];
    option.legend.data = series.map((s) => s.name);
    option.series = series;
    chartInstance('chart-line').setOption(option, true);
  }

  function renderBarChart(cards) {
    const withCmp = cards.filter((c) => c.comparison);
    if (!withCmp.length) {
      showPlaceholder('chart-bar', '勾选含 comparison 数据的实验卡');
      return;
    }
    const { xAxis, series } = transforms.buildMultiBarSeries(withCmp);
    const option = chartBase('Clean vs Poisoned 对比');
    option.xAxis = { type: 'category', data: xAxis, axisLabel: { interval: 0 } };
    option.yAxis = { type: 'value' };
    option.legend.data = series.map((s) => s.name);
    option.series = series;
    chartInstance('chart-bar').setOption(option, true);
  }

  function renderCharts() {
    const cards = state.cards.filter((c) => c.checked && cardHasData(c));
    renderLineChart(cards);
    renderBarChart(cards);
  }

  function renderAll() {
    renderCards();
    renderPanel();
    renderCharts();
  }

  function exportJson() {
    const card = activeCard();
    if (!card || !card.epochMetrics) {
      showMessage('当前实验卡没有 history 数据', true);
      return;
    }
    const blob = new Blob([JSON.stringify(card.epochMetrics, null, 2)],
      { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${card.name}-epoch_metrics.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function initApp() {
    $('add-card').addEventListener('click', () => {
      openModal('添加实验卡', {
        input: true,
        inputValue: `实验 ${nextCardSeq}`,
        extra: true,
        extraLabel: '导入实验路径自动命名',
        onExtra: (closeModal) => {
          const input = document.createElement('input');
          input.type = 'file';
          input.webkitdirectory = true;
          input.multiple = true;
          input.addEventListener('change', (e) => {
            const files = [...e.target.files];
            if (!files.length) return;
            const info = parser.parseDirectoryPath(files[0].webkitRelativePath || files[0].name);
            const card = addCard(parser.buildAutoName(info) || `实验 ${nextCardSeq}`);
            closeModal();
            importDirIntoCard(card, files);
          });
          input.click();
        },
        onOk: (v) => {
          addCard(v || `实验 ${nextCardSeq}`);
          renderAll();
        },
      });
    });
    $('export-json').addEventListener('click', exportJson);
    showMessage('点击「＋ 添加实验卡」开始');
  }

  global.TPAVisualizer = global.TPAVisualizer || {};
  global.TPAVisualizer.app = { initApp, addCard, exportJson };
})(window);
