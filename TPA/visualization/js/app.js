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
  const ARTIFACT_NAMES = new Set([
    'history.json', 'eval_log.csv', 'config.yaml', 'latest.json', 'surrogate_meta.json',
  ]);

  function isArtifact(name) {
    return ARTIFACT_NAMES.has(name) || /_comparison\.(json|md)$/.test(name);
  }

  function normalizeEntry(relPath, name, text) {
    return { segments: String(relPath || '').split('/').filter(Boolean), name, text };
  }

  function groupByExperiment(entries) {
    const groups = new Map();
    for (const entry of entries) {
      if (!isArtifact(entry.name)) continue;
      const seg = entry.segments;
      const runIdx = seg.findIndex((s) => RUN_TAG_RE.test(s));
      const key = runIdx !== -1 ? seg[runIdx] : (seg[0] || 'unknown');
      if (!groups.has(key)) {
        groups.set(key, { segments: seg.slice(0, runIdx + 1), files: [] });
      }
      groups.get(key).files.push(entry);
    }

    // 根目录稳定副本（latest.json 指针）归并到对应 run_tag 分组
    const result = new Map();
    for (const [key, g] of groups) {
      const hasRunTag = g.segments.some((s) => RUN_TAG_RE.test(s));
      if (!hasRunTag) {
        const latest = g.files.find((f) => f.name === 'latest.json');
        if (latest) {
          try {
            const tag = JSON.parse(latest.text).run_tag;
            if (tag && RUN_TAG_RE.test(tag)) {
              if (result.has(tag)) {
                result.get(tag).files.push(...g.files);
              } else {
                result.set(tag, { segments: [tag], files: [...g.files] });
              }
              continue;
            }
          } catch (e) { /* 坏 latest.json 忽略 */ }
        }
      }
      result.set(key, g);
    }

    // 只有含有效实验信号的分组才生成实验（过滤 surrogate_meta / latest.json 等孤立文件）
    return [...result.values()].filter((g) => g.files.some(
      (f) => f.name === 'history.json' || f.name === 'config.yaml'
        || /_comparison\.json$/.test(f.name),
    ));
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
