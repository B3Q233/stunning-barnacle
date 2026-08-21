'use strict';
(function (global, factory) {
  if (typeof module === 'object' && module.exports) {
    const registry = require('./index.js');
    const { extract } = require('./path.js');
    module.exports = factory(registry, extract);
  } else {
    global.TPAVisualizer = global.TPAVisualizer || {};
    const registry = global.TPAVisualizer.registry;
    const { extract } = global.TPAVisualizer.path;
    global.TPAVisualizer.normalize = factory(registry, extract);
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function (registry, extract) {

  function normalize(json, schema) {
    const result = {
      title: schema.title || schema.name || '未命名视图',
      type: schema.type || 'line',
      x: null,
      series: [],
    };
    if (schema.x) {
      result.x = extract(json, schema.x);
    }
    for (const [label, path] of Object.entries(schema.series || {})) {
      result.series.push({ name: label, data: extract(json, path) });
    }
    return result;
  }

  function buildVisualization(json) {
    return normalize(json, registry.getSchema(json));
  }

  return { normalize, buildVisualization };
});
