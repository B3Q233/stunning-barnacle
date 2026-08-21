'use strict';
(function (global, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    global.TPAVisualizer = global.TPAVisualizer || {};
    global.TPAVisualizer.treeBuilder = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {

  function typeOf(v) {
    if (v === null) return 'null';
    return Array.isArray(v) ? 'array' : typeof v;
  }

  // JSON → 树：对象/数组为分支节点，标量为叶子节点（含完整路径）。
  function buildTree(json) {
    const root = { name: 'root', type: typeOf(json), path: '', leaf: false, children: [] };

    function dfs(node, path, parent) {
      if (!node || typeof node !== 'object') return;
      if (Array.isArray(node)) {
        if (node.length > 0) dfs(node[0], `${path}[]`, parent);
        return;
      }
      for (const k of Object.keys(node).sort()) {
        const p = path ? `${path}.${k}` : k;
        const v = node[k];
        const child = { name: k, path: p, type: typeOf(v), leaf: false, children: [] };
        parent.children.push(child);
        if (Array.isArray(v)) {
          child.type = 'array';
          if (v.length > 0) dfs(v[0], `${p}[]`, child);
        } else if (v && typeof v === 'object') {
          dfs(v, p, child);
        } else {
          child.leaf = true;
          child.value = v;
        }
      }
    }

    dfs(json, '', root);
    return root;
  }

  function collectLeaves(root) {
    const out = [];
    function walk(node) {
      if (node.leaf) {
        out.push(node);
        return;
      }
      for (const c of node.children || []) walk(c);
    }
    walk(root);
    return out;
  }

  function numericLeaves(root) {
    return collectLeaves(root).filter((l) => l.type === 'number');
  }

  return { buildTree, collectLeaves, numericLeaves, typeOf };
});
