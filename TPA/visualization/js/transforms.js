/* TPA 图表数据变换：折线/直方/对比 series 构建与调色板（纯函数） */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.TPAVisualizer = root.TPAVisualizer || {};
    root.TPAVisualizer.transforms = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const PALETTE = [
    '#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de',
    '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc', '#2f4554',
  ];

  function assignPalette(experiments) {
    return experiments.map((exp, i) => (
      { ...exp, color: exp.color || PALETTE[i % PALETTE.length] }
    ));
  }

  function numericKeys(rows) {
    const keys = new Set();
    for (const row of rows) {
      for (const [k, v] of Object.entries(row)) {
        if (typeof v === 'number') keys.add(k);
      }
    }
    return [...keys];
  }

  function listMetrics(experiments) {
    const keys = new Set();
    for (const exp of experiments) {
      for (const k of numericKeys(exp.history)) keys.add(k);
      for (const k of numericKeys(exp.evalLog)) keys.add(k);
    }
    return [...keys];
  }

  function buildLineSeries(experiments, metric, xKey = 'epoch') {
    const rowsFor = (exp) => {
      const all = [...exp.history, ...exp.evalLog];
      const sel = exp.selectedEpochs;
      if (!sel || !sel.length) return all;
      const set = new Set(sel);
      return all.filter((row) => set.has(row[xKey]));
    };
    const xSet = new Set();
    for (const exp of experiments) {
      for (const row of rowsFor(exp)) xSet.add(row[xKey]);
    }
    const xAxis = [...xSet].sort((a, b) => a - b);
    const series = experiments.map((exp) => {
      const dataByX = new Map();
      for (const row of rowsFor(exp)) {
        const x = row[xKey];
        if (x !== undefined && typeof row[metric] === 'number') dataByX.set(x, row[metric]);
      }
      return {
        name: exp.label,
        type: 'line',
        color: exp.color,
        data: xAxis.map((x) => (dataByX.has(x) ? dataByX.get(x) : null)),
      };
    });
    return { xAxis, series };
  }

  function bestValue(exp, metric) {
    if (exp.best && exp.best[metric] && typeof exp.best[metric].value === 'number') {
      return exp.best[metric].value;
    }
    for (const row of [...exp.history, ...exp.evalLog].reverse()) {
      if (typeof row[metric] === 'number') return row[metric];
    }
    return null;
  }

  function buildMetricBars(experiments, metrics) {
    const xAxis = experiments.map((e) => e.label);
    const series = metrics.map((metric) => ({
      name: metric,
      type: 'bar',
      data: experiments.map((e) => bestValue(e, metric)),
    }));
    return { xAxis, series };
  }

  function targetAvg(comparison, side, metric) {
    const targets = ((comparison || {}).target_metrics || {})[side] || {};
    const values = Object.values(targets)
      .map((t) => t[metric])
      .filter((v) => typeof v === 'number');
    if (!values.length) return null;
    return values.reduce((a, b) => a + b, 0) / values.length;
  }

  function buildComparisonSeries(experiments, metric, mode = 'model') {
    const xAxis = experiments.map((e) => e.label);
    const pick = (exp, side) => {
      const cmp = exp.comparison || {};
      const src = mode === 'model' ? (cmp.model_utility || {}) : (cmp.target_metrics || {});
      const sideData = src[side] || {};
      if (mode === 'model') {
        return typeof sideData[metric] === 'number' ? sideData[metric] : null;
      }
      return targetAvg(exp.comparison, side, metric);
    };
    return {
      xAxis,
      series: [
        { name: 'Clean', type: 'bar', data: experiments.map((e) => pick(e, 'clean')) },
        { name: 'Poisoned', type: 'bar', data: experiments.map((e) => pick(e, 'poisoned')) },
      ],
    };
  }

  function pad2(n) {
    return String(n).padStart(2, '0');
  }

  function buildExportFilename(chartType, metric, now = new Date()) {
    const safeMetric = String(metric || 'summary').replace(/[^\w@%.-]/g, '-');
    const stamp = `${now.getFullYear()}${pad2(now.getMonth() + 1)}${pad2(now.getDate())}-${pad2(now.getHours())}${pad2(now.getMinutes())}`;
    return `tpa-${chartType}-${safeMetric}-${stamp}.png`;
  }

  return {
    PALETTE, assignPalette, listMetrics, buildLineSeries, buildMetricBars,
    buildComparisonSeries, buildExportFilename,
  };
});
