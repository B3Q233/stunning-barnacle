'use strict';
(function (global, factory) {
  if (typeof module === 'object' && module.exports) {
    const fs = require('fs');
    const path = require('path');
    const detector = require('./detector.js');
    module.exports = factory(detector, {
      node: true,
      fs,
      path,
      schemaDir: path.join(__dirname, '..', 'schema'),
      builtin: [],
      storage: null,
    });
  } else {
    global.TPAVisualizer = global.TPAVisualizer || {};
    const detector = global.TPAVisualizer.detector;
    global.TPAVisualizer.registry = factory(detector, {
      node: false,
      fs: null,
      path: null,
      schemaDir: null,
      builtin: global.TPAVisualizer.builtin || [],
      storage: typeof localStorage !== 'undefined' ? localStorage : null,
    });
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function (detector, env) {

  const builtin = [];
  let custom = [];

  function loadBuiltin() {
    if (env.node) {
      const files = env.fs.readdirSync(env.schemaDir)
        .filter((f) => f.endsWith('.schema.json'));
      for (const f of files) {
        builtin.push(JSON.parse(
          env.fs.readFileSync(env.path.join(env.schemaDir, f), 'utf-8')));
      }
    } else {
      for (const s of env.builtin) builtin.push(s);
    }
  }

  function loadCustom() {
    if (env.node || !env.storage) return;
    try {
      const raw = env.storage.getItem('tpa_vis_custom_schemas');
      if (raw) custom = JSON.parse(raw);
    } catch (e) {
      custom = [];
    }
  }

  function persistCustom() {
    if (env.node || !env.storage) return;
    try {
      env.storage.setItem('tpa_vis_custom_schemas', JSON.stringify(custom));
    } catch (e) {
      // 存储不可用时忽略（例如隐私模式）
    }
  }

  function fpKey(fp) {
    return JSON.stringify(fp);
  }

  // 按结构指纹匹配；未命中返回 null（调用方进入设计器）
  function match(json) {
    const key = fpKey(detector.fingerprint(json));
    return [...builtin, ...custom]
      .find((s) => fpKey(s.fingerprint) === key) || null;
  }

  function saveCustom(schema) {
    const idx = custom.findIndex((s) => s.id === schema.id);
    if (idx >= 0) custom[idx] = schema;
    else custom.push(schema);
    persistCustom();
    return schema;
  }

  function all() {
    return [...builtin, ...custom];
  }

  function resetCustom() {
    custom = [];
    persistCustom();
  }

  loadBuiltin();
  loadCustom();

  return {
    match, saveCustom, all, resetCustom,
    fingerprint: detector.fingerprint,
    schemaId: detector.schemaId,
  };
});
