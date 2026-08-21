'use strict';
(function (global, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    global.TPAVisualizer = global.TPAVisualizer || {};
    global.TPAVisualizer.path = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {

  function walk(obj, parts) {
    if (!parts.length) return obj;
    const [current, ...rest] = parts;

    // 数组节点 history[]
    if (current.endsWith('[]')) {
      const key = current.slice(0, -2);
      if (obj == null || !Array.isArray(obj[key])) return [];
      return obj[key].map((item) => walk(item, rest));
    }

    if (obj == null) return null;
    return walk(obj[current], rest);
  }

  function extract(json, path) {
    if (!path) return null;
    return walk(json, String(path).split('.'));
  }

  return { extract };
});
