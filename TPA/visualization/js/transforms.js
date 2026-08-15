'use strict';
(function (global, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    global.TPAVisualizer = global.TPAVisualizer || {};
    global.TPAVisualizer.transforms = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {

  const PALETTE = [
    '#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de',
    '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc',
  ];

  function colorFor(index) {
    return PALETTE[index % PALETTE.length];
  }

  function buildLineSeries(epochMetrics, selectedMetrics) {
    const xAxis = Object.keys(epochMetrics).sort((a, b) => Number(a) - Number(b));
    const series = selectedMetrics.map((name, i) => ({
      name,
      type: 'line',
      color: colorFor(i),
      data: xAxis.map((epoch) => {
        const v = epochMetrics[epoch][name];
        return typeof v === 'number' ? v : null;
      }),
    }));
    return { xAxis, series };
  }

  function avg(values) {
    const nums = values.filter((v) => typeof v === 'number');
    if (!nums.length) return null;
    return nums.reduce((a, b) => a + b, 0) / nums.length;
  }

  function buildComparisonItems(comparison) {
    const items = [];
    const mu = comparison.modelUtility || {};
    const muNames = new Set([
      ...Object.keys(mu.clean || {}),
      ...Object.keys(mu.poisoned || {}),
    ]);
    for (const name of muNames) {
      items.push({
        name,
        clean: mu.clean ? mu.clean[name] ?? null : null,
        poisoned: mu.poisoned ? mu.poisoned[name] ?? null : null,
      });
    }
    const tm = comparison.targetMetrics || {};
    const hr = { clean: [], poisoned: [] };
    const ndcg = { clean: [], poisoned: [] };
    for (const side of ['clean', 'poisoned']) {
      for (const t of Object.values(tm[side] || {})) {
        if (typeof t['hr@k'] === 'number') hr[side].push(t['hr@k']);
        if (typeof t['ndcg@k'] === 'number') ndcg[side].push(t['ndcg@k']);
      }
    }
    items.push({ name: 'target_hr@k', clean: avg(hr.clean), poisoned: avg(hr.poisoned) });
    items.push({ name: 'target_ndcg@k', clean: avg(ndcg.clean), poisoned: avg(ndcg.poisoned) });
    return items;
  }

  function buildComparisonSeries(items, selectedNames) {
    const picked = items.filter((it) => selectedNames.includes(it.name));
    return {
      xAxis: picked.map((it) => it.name),
      series: [
        { name: 'Clean', type: 'bar', data: picked.map((it) => it.clean) },
        { name: 'Poisoned', type: 'bar', data: picked.map((it) => it.poisoned) },
      ],
    };
  }

  return { PALETTE, colorFor, buildLineSeries, buildComparisonItems, buildComparisonSeries };
});
