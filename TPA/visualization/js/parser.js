'use strict';
(function (global, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    global.TPAVisualizer = global.TPAVisualizer || {};
    global.TPAVisualizer.parser = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {

  function parseHistoryRows(text) {
    let data;
    try {
      data = JSON.parse(text);
    } catch (e) {
      throw new Error(`history.json 解析失败: ${e.message}`);
    }
    const rows = Array.isArray(data)
      ? data
      : (data && Array.isArray(data.history) ? data.history : []);
    if (!rows.length) throw new Error('history.json 中没有 history 记录');
    return rows.map((row, idx) => {
      if (!row || typeof row !== 'object') {
        throw new Error(`history.json 第 ${idx + 1} 条记录不是对象`);
      }
      const out = {};
      for (const [k, v] of Object.entries(row)) {
        if (typeof v === 'number') out[k] = v;
      }
      if (!('epoch' in out)) {
        throw new Error(`history.json 第 ${idx + 1} 条记录缺少 epoch 数值字段`);
      }
      return out;
    });
  }

  function extractEpochMetrics(text) {
    const rows = parseHistoryRows(text);
    const out = {};
    for (const row of rows) {
      const metrics = {};
      for (const [k, v] of Object.entries(row)) {
        if (k !== 'epoch') metrics[k] = v;
      }
      out[String(row.epoch)] = metrics;
    }
    return out;
  }

  function listMetrics(epochMetrics) {
    const names = [];
    const seen = new Set();
    for (const metrics of Object.values(epochMetrics)) {
      for (const name of Object.keys(metrics)) {
        if (!seen.has(name)) {
          seen.add(name);
          names.push(name);
        }
      }
    }
    return names;
  }

  function parseComparison(text) {
    let data;
    try {
      data = JSON.parse(text);
    } catch (e) {
      throw new Error(`comparison 解析失败: ${e.message}`);
    }
    return {
      modelUtility: (data && data.model_utility) || { clean: {}, poisoned: {} },
      targetMetrics: (data && data.target_metrics) || { clean: {}, poisoned: {} },
    };
  }

  function parseDirectoryPath(relativePath) {
    const parts = String(relativePath || '').split('/').filter(Boolean);
    const attacksIdx = parts.indexOf('attacks');
    const outIdx = parts.indexOf('outputs');
    const runIdx = parts.findIndex((s) => /^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}$/.test(s));
    if (attacksIdx === -1 || outIdx === -1 || runIdx === -1
        || !(outIdx > attacksIdx && runIdx > outIdx)) {
      return null;
    }
    return {
      method: parts[attacksIdx + 1] || null,
      dataset: parts[outIdx + 1] || null,
      model: parts[outIdx + 2] || null,
      runTag: parts[runIdx],
    };
  }

  function buildAutoName(info) {
    if (!info || !info.method || !info.runTag) return null;
    return `${info.method}-${info.runTag}`;
  }

  return {
    parseHistoryRows, extractEpochMetrics, listMetrics, parseComparison,
    parseDirectoryPath, buildAutoName,
  };
});
