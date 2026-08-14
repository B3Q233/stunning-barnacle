/* TPA 可视化应用逻辑：导入分组、状态管理、持久化（Node/浏览器双端） */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory(require('./parser.js'));
  } else {
    root.TPAVisualizer = root.TPAVisualizer || {};
    root.TPAVisualizer.app = factory(root.TPAVisualizer.parser);
  }
})(typeof self !== 'undefined' ? self : this, function (parser) {
  'use strict';

  const STORAGE_KEY = 'tpa.visualizer.v1';
  const RUN_TAG_RE = /^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}$/;

  function normalizeEntry(relPath, name, text) {
    return { segments: String(relPath || '').split('/').filter(Boolean), name, text };
  }

  // 只认 run_tag 子实验目录（YYYY-MM-DD-HH-MM），忽略 output 根目录及其他数据
  function groupByExperiment(entries) {
    const groups = new Map();
    for (const entry of entries) {
      const seg = entry.segments;
      const runIdx = seg.findIndex((s) => RUN_TAG_RE.test(s));
      if (runIdx === -1) continue;
      const key = seg[runIdx];
      if (!groups.has(key)) {
        groups.set(key, { segments: seg.slice(0, runIdx + 1), files: [] });
      }
      groups.get(key).files.push(entry);
    }
    return [...groups.values()];
  }

  function importFiles(entries) {
    const groups = groupByExperiment(
      (entries || []).map((e) => normalizeEntry(e.relativePath, e.name, e.text)),
    );
    const experiments = [];
    const errors = [];
    for (const g of groups) {
      try {
        experiments.push(parser.parseExperiment(g.segments, g.files));
      } catch (e) {
        errors.push(`${g.segments.join('/')}: ${e.message}`);
      }
    }
    return { experiments, errors };
  }

  function saveState(state) {
    if (typeof localStorage === 'undefined') return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        experiments: state.experiments,
        selected: [...state.selected],
        metric: state.metric,
        chartType: state.chartType,
        title: state.title,
      }));
    } catch (e) {
      console.warn('localStorage 持久化失败（容量或隐私模式），仅本次会话保留', e);
    }
  }

  function loadState() {
    if (typeof localStorage === 'undefined') return null;
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      return null;
    }
  }

  function initApp() {
    return { importFiles, saveState, loadState, STORAGE_KEY };
  }

  return { initApp, importFiles, saveState, loadState, STORAGE_KEY };
});
