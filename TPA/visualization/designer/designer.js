'use strict';
(function (global, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    global.TPAVisualizer = global.TPAVisualizer || {};
    global.TPAVisualizer.designer = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {

  const CHART_TYPES = ['line', 'bar', 'pie', 'metric'];
  const TYPE_LABELS = { line: '折线图', bar: '柱状图', pie: '饼图', metric: '指标卡' };

  function lastSegment(path) {
    const parts = String(path).split('.');
    return parts[parts.length - 1];
  }

  // 未识别的 JSON 结构 → 打开设计器：选图类型 + 树状勾选字段 + 别名 → 保存并注册。
  function openSchemaDesigner(json, onSave) {
    const detector = global.TPAVisualizer.detector;
    const registry = global.TPAVisualizer.registry;
    const treeBuilder = global.TPAVisualizer.treeBuilder;

    const fp = detector.fingerprint(json);
    const id = detector.schemaId(fp);
    const root = treeBuilder.buildTree(json);
    const leaves = treeBuilder.collectLeaves(root);
    const numbers = treeBuilder.numericLeaves(root);
    const state = {
      type: 'line',
      title: id,
      x: numbers.length ? numbers[0].path : null,
      checked: new Set(numbers.map((l) => l.path)),
      alias: {},
    };

    const overlay = document.getElementById('modal-overlay');
    const body = document.getElementById('modal-body');
    document.getElementById('modal-title').textContent = '设计新 Schema（未识别结构）';
    body.innerHTML = '';

    // 标题
    const titleRow = document.createElement('label');
    titleRow.className = 'metric-row';
    titleRow.append('标题 ');
    const titleInput = document.createElement('input');
    titleInput.type = 'text';
    titleInput.value = state.title;
    titleInput.addEventListener('input', () => { state.title = titleInput.value; });
    titleRow.appendChild(titleInput);
    body.appendChild(titleRow);

    // 图类型
    const typeRow = document.createElement('div');
    typeRow.className = 'metric-row';
    typeRow.append('图类型 ');
    for (const t of CHART_TYPES) {
      const label = document.createElement('label');
      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = 'chart-type';
      radio.checked = t === state.type;
      radio.addEventListener('change', () => {
        state.type = t;
        xGroup.style.display = t === 'line' ? '' : 'none';
      });
      label.append(radio, document.createTextNode(` ${TYPE_LABELS[t]} `));
      typeRow.appendChild(label);
    }
    body.appendChild(typeRow);

    // X 轴（仅折线）
    const xGroup = document.createElement('div');
    xGroup.className = 'metric-row';
    xGroup.append('X 轴 ');
    for (const l of numbers) {
      const label = document.createElement('label');
      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = 'chart-x';
      radio.checked = l.path === state.x;
      radio.addEventListener('change', () => { state.x = l.path; });
      label.append(radio, document.createTextNode(` ${lastSegment(l.path)} `));
      xGroup.appendChild(label);
    }
    body.appendChild(xGroup);

    // JSON 树：可勾选叶子 + 别名
    const tree = document.createElement('div');
    tree.className = 'designer-tree';
    renderNode(root, tree);
    body.appendChild(tree);

    function renderNode(node, container) {
      if (node.leaf) {
        const row = document.createElement('label');
        row.className = 'metric-row';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = state.checked.has(node.path);
        cb.addEventListener('change', () => {
          if (cb.checked) state.checked.add(node.path);
          else state.checked.delete(node.path);
        });
        const alias = document.createElement('input');
        alias.type = 'text';
        alias.value = lastSegment(node.path);
        alias.style.width = '130px';
        alias.addEventListener('input', () => { state.alias[node.path] = alias.value; });
        const type = document.createElement('span');
        type.className = 'badge';
        type.textContent = node.type;
        row.append(cb, document.createTextNode(` ${lastSegment(node.path)} `), type, alias);
        container.appendChild(row);
        return;
      }
      const details = document.createElement('details');
      details.open = true;
      const summary = document.createElement('summary');
      summary.textContent = `${node.name} (${node.type})`;
      details.appendChild(summary);
      const sub = document.createElement('div');
      for (const c of node.children) renderNode(c, sub);
      details.appendChild(sub);
      container.appendChild(details);
    }

    function close() {
      overlay.classList.add('hidden');
      body.innerHTML = '';
      okBtn.removeEventListener('click', save);
      cancelBtn.removeEventListener('click', close);
      exportBtn.removeEventListener('click', exportSchema);
    }

    function save() {
      const series = {};
      [...state.checked].sort().forEach((p) => {
        series[state.alias[p] || lastSegment(p)] = p;
      });
      const schema = {
        id,
        title: state.title || id,
        fingerprint: fp,
        type: state.type,
        x: state.type === 'line' ? state.x : null,
        series,
      };
      registry.saveCustom(schema);
      close();
      onSave && onSave(schema);
    }

    function exportSchema() {
      const schema = {
        id,
        title: state.title || id,
        fingerprint: fp,
        type: state.type,
        x: state.type === 'line' ? state.x : null,
        series: (() => {
          const s = {};
          [...state.checked].sort().forEach((p) => { s[state.alias[p] || lastSegment(p)] = p; });
          return s;
        })(),
      };
      const blob = new Blob([JSON.stringify(schema, null, 2)],
        { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${id}.schema.json`;
      a.click();
      URL.revokeObjectURL(a.href);
    }

    const exportBtn = document.createElement('button');
    exportBtn.type = 'button';
    exportBtn.textContent = '导出 Schema';
    exportBtn.addEventListener('click', exportSchema);
    body.appendChild(exportBtn);

    const okBtn = document.getElementById('modal-ok');
    okBtn.textContent = '保存并渲染';
    const cancelBtn = document.getElementById('modal-cancel');
    okBtn.addEventListener('click', save);
    cancelBtn.addEventListener('click', close);
    overlay.classList.remove('hidden');
  }

  return { openSchemaDesigner };
});
