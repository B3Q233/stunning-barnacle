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
    '#3B4992', '#EE0000', '#008B45', '#631879', '#008280',
    '#BB0021', '#5F559B', '#A20056', '#808180', '#1B1919',
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

  function buildMultiLineSeries(experiments) {
    const xSet = new Set();
    const metricNames = [];
    for (const exp of experiments) {
      for (const e of Object.keys(exp.epochMetrics || {})) xSet.add(Number(e));
      for (const [name, opt] of Object.entries(exp.metricOptions || {})) {
        if (opt && opt.selected && !metricNames.includes(name)) metricNames.push(name);
      }
    }
    const xAxis = [...xSet].sort((a, b) => a - b).map(String);
    const series = [];
    for (const metric of metricNames) {
      for (const exp of experiments) {
        const opt = (exp.metricOptions || {})[metric];
        if (!opt || !opt.selected) continue;
        series.push({
          name: `${metric}-${exp.name}`,
          type: 'line',
          color: opt.color || colorFor(series.length),
          data: xAxis.map((e) => {
            const v = exp.epochMetrics[e] ? exp.epochMetrics[e][metric] : undefined;
            return typeof v === 'number' ? v : null;
          }),
        });
      }
    }
    return { xAxis, series };
  }

  function buildMultiBarSeries(experiments) {
    const xAxis = experiments.map((e) => e.name);
    const barNames = new Set();
    const metricNames = [];
    for (const exp of experiments) {
      for (const it of buildComparisonItems(exp.comparison || {})) barNames.add(it.name);
      for (const [name, opt] of Object.entries(exp.metricOptions || {})) {
        if (opt && opt.selected && barNames.has(name) && !metricNames.includes(name)) {
          metricNames.push(name);
        }
      }
    }
    const series = [];
    for (const metric of metricNames) {
      const cleanData = [];
      const poisonedData = [];
      for (const exp of experiments) {
        const opt = (exp.metricOptions || {})[metric];
        const item = buildComparisonItems(exp.comparison || {})
          .find((it) => it.name === metric);
        const color = opt ? opt.color : undefined;
        const mk = (v) => (v === null || v === undefined
          ? null : { value: v, itemStyle: color ? { color } : undefined });
        cleanData.push(mk(item ? item.clean : null));
        poisonedData.push(mk(item ? item.poisoned : null));
      }
      series.push({ name: `${metric}-Clean`, type: 'bar', data: cleanData });
      series.push({ name: `${metric}-Poisoned`, type: 'bar', data: poisonedData });
    }
    return { xAxis, series };
  }

  return {
    PALETTE, colorFor, buildLineSeries, buildComparisonItems, buildComparisonSeries,
    buildMultiLineSeries, buildMultiBarSeries,
  };
});
