/* TPA 实验数据解析器：路径识别 + 产物文件解析（纯函数，Node/浏览器双端） */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.TPAVisualizer = root.TPAVisualizer || {};
    root.TPAVisualizer.parser = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const RUN_TAG_RE = /^(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$/;

  function formatRunTag(runTag) {
    const m = RUN_TAG_RE.exec(runTag);
    if (!m) return runTag;
    return `${m[1]}年${m[2]}月${m[3]}日${m[4]}时${m[5]}分`;
  }

  function buildExperimentId(kind, method, dataset, model, runTag) {
    return [kind, method, dataset, model, runTag].filter(Boolean).join('-');
  }

  function buildLabel(kind, method, dataset, model, runTag) {
    const parts = [kind, method, dataset, model].filter(Boolean);
    if (runTag) parts.push(formatRunTag(runTag));
    return parts.join('-');
  }

  function parseHistoryJson(text) {
    try {
      const data = JSON.parse(text);
      return Array.isArray(data.history) ? data.history : [];
    } catch (e) {
      throw new Error(`解析失败: history.json 不是合法 JSON（${e.message}）`);
    }
  }

  function parseBest(text) {
    try {
      const data = JSON.parse(text);
      return data.best && typeof data.best === 'object' ? data.best : null;
    } catch (e) {
      return null;
    }
  }

  function parseEvalCsv(text) {
    const lines = text.trim().split(/\r?\n/).filter(Boolean);
    if (lines.length < 2) return [];
    const headers = lines[0].split(',').map((s) => s.trim());
    return lines.slice(1).map((line) => {
      const values = line.split(',').map((s) => s.trim());
      const row = {};
      headers.forEach((h, i) => {
        row[h] = values[i] === undefined || values[i] === ''
          ? null
          : Number(values[i]);
      });
      return row;
    });
  }

  function parseConfigYaml(text) {
    const cfg = {
      dataset: null, model: null, attack: null, epochs: null, metrics: [],
      checkpoint_dir: null,
    };
    let section = null;
    let inMetrics = false;
    for (const rawLine of text.split(/\r?\n/)) {
      const line = rawLine.replace(/\s*#.*$/, '');
      const trimmed = line.trim();
      if (!trimmed) continue;

      if (inMetrics) {
        if (trimmed.startsWith('- ')) {
          const metric = trimmed.slice(2).split(':')[0].trim();
          if (metric) cfg.metrics.push(metric);
          continue;
        }
        inMetrics = false;
      }

      const indent = line.match(/^\s*/)[0].length;
      const m = line.match(/^\s*([\w@.-]+):\s*(.*)$/);
      if (!m) continue;
      const key = m[1];
      const value = m[2].trim().replace(/['"]/g, '');
      if (indent === 0) section = key;
      if (key === 'dataset' && !value.includes(':')) cfg.dataset = value;
      if (key === 'name' && section === 'model' && value) cfg.model = value;
      if (key === 'name' && section === 'attack' && value) cfg.attack = value;
      if (key === 'epochs' && value) cfg.epochs = Number(value);
      if (key === 'checkpoint_dir' && value) cfg.checkpoint_dir = value;
      if (key === 'metrics') inMetrics = true;
    }
    return cfg;
  }

  function inferModelName(checkpointDir) {
    if (!checkpointDir) return null;
    const m = String(checkpointDir).match(/models\/([^/]+)\/outputs/);
    return m ? m[1] : null;
  }

  function parseComparisonJson(text) {
    try {
      const data = JSON.parse(text);
      return {
        model_utility: data.model_utility || {},
        target_metrics: data.target_metrics || {},
      };
    } catch (e) {
      throw new Error(`解析失败: comparison JSON 不是合法 JSON（${e.message}）`);
    }
  }

  function readFile(files, name) {
    const f = (files || []).find((x) => x.name === name);
    return f ? f.text : null;
  }

  function parseExperiment(segments, files) {
    const seg = segments || [];
    let kind = 'custom';
    let method = null;
    let dataset = null;
    let model = null;
    let runTag = null;

    const attacksIdx = seg.indexOf('attacks');
    const modelsIdx = seg.indexOf('models');
    if (attacksIdx !== -1) {
      kind = 'attack';
      method = seg[attacksIdx + 1] || null;
      const rest = seg.slice(attacksIdx + 2);
      const outIdx = rest.indexOf('outputs');
      const tail = outIdx === -1 ? [] : rest.slice(outIdx + 1);
      dataset = tail[0] || null;
      model = tail[1] || null;
      runTag = tail[2] && RUN_TAG_RE.test(tail[2]) ? tail[2] : null;
    } else if (modelsIdx !== -1) {
      kind = 'model';
      model = seg[modelsIdx + 1] || null;
      const rest = seg.slice(modelsIdx + 2);
      const outIdx = rest.indexOf('outputs');
      const tail = outIdx === -1 ? [] : rest.slice(outIdx + 1);
      runTag = tail[0] && RUN_TAG_RE.test(tail[0]) ? tail[0] : null;
    }

    const cfgText = readFile(files, 'config.yaml');
    const cfg = cfgText ? parseConfigYaml(cfgText) : {};
    if (kind === 'model' && !dataset) dataset = cfg.dataset || 'unknown';
    if (kind === 'attack') {
      dataset = dataset || cfg.dataset || 'unknown';
      model = model || cfg.model || 'unknown';
    }

    // 直接选择 run_tag 目录导入时，路径缺少父级段，改用 config.yaml 补全识别
    if (kind === 'custom') {
      const runIdx = seg.findIndex((s) => RUN_TAG_RE.test(s));
      if (runIdx !== -1) runTag = seg[runIdx];
      if (runTag && cfg.attack) {
        kind = 'attack';
        method = cfg.attack;
        dataset = cfg.dataset || 'unknown';
        model = cfg.model || 'unknown';
      } else if (runTag && (cfg.dataset || cfg.checkpoint_dir)) {
        kind = 'model';
        model = cfg.model || inferModelName(cfg.checkpoint_dir) || 'unknown';
        dataset = cfg.dataset || 'unknown';
      }
    }

    const historyText = readFile(files, 'history.json') || '{"history":[]}';
    const compFile = (files || []).find((f) => /_comparison\.json$/.test(f.name));

    if (kind === 'custom') {
      const base = seg[seg.length - 1] || '未知';
      return {
        id: `custom-${base}`,
        kind, method, dataset, model, runTag,
        label: `自定义-${base}`,
        color: null,
        history: parseHistoryJson(historyText),
        best: parseBest(historyText),
        evalLog: parseEvalCsv(readFile(files, 'eval_log.csv') || ''),
        comparison: null,
        meta: { epochs: cfg.epochs, metrics: cfg.metrics },
      };
    }

    return {
      id: buildExperimentId(kind, method, dataset, model, runTag || 'unknown'),
      kind, method, dataset, model, runTag,
      label: buildLabel(kind, method, dataset, model, runTag),
      color: null,
      history: parseHistoryJson(historyText),
      best: parseBest(historyText),
      evalLog: parseEvalCsv(readFile(files, 'eval_log.csv') || ''),
      comparison: compFile ? parseComparisonJson(compFile.text) : null,
      meta: { epochs: cfg.epochs, metrics: cfg.metrics },
    };
  }

  return {
    formatRunTag, buildExperimentId, buildLabel, parseExperiment,
    parseHistoryJson, parseBest, parseEvalCsv, parseConfigYaml, parseComparisonJson,
  };
});
