'use strict';
(function (global, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    global.TPAVisualizer = global.TPAVisualizer || {};
    global.TPAVisualizer.detector = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {

  // 结构指纹：只记录"路径集合"，与具体值无关。
  // history[] 表示数组节点；history[].epoch 表示数组元素内的字段。
  function fingerprint(obj, prefix) {
    const paths = [];

    function dfs(node, path) {
      if (Array.isArray(node)) {
        paths.push(`${path}[]`);
        if (node.length > 0) dfs(node[0], `${path}[]`);
        return;
      }
      if (typeof node === 'object' && node !== null) {
        Object.keys(node).sort().forEach((k) => {
          const p = path ? `${path}.${k}` : k;
          paths.push(p);
          dfs(node[k], p);
        });
      }
    }

    dfs(obj, prefix || '');
    return [...new Set(paths)].sort();
  }

  function fnv1a(str) {
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i += 1) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
  }

  function schemaId(fp) {
    const hex = fnv1a(fp.join('\n')).toString(16).padStart(8, '0');
    return `fp_${hex.slice(0, 6)}`;
  }

  return { fingerprint, schemaId };
});
