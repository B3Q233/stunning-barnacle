'use strict';
(function (global, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    global.TPAVisualizer = global.TPAVisualizer || {};
    global.TPAVisualizer.registry = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {

  const registry = [];

  function register(schema) {
    if (!schema || typeof schema.match !== 'function') {
      throw new Error('schema 必须实现 match(json)');
    }
    registry.push(schema);
    return schema;
  }

  function getSchema(json) {
    for (const schema of registry) {
      try {
        if (schema.match(json)) return schema;
      } catch (e) {
        // match 内部异常视为不匹配
      }
    }
    throw new Error('未找到匹配的 visualization schema');
  }

  function schemas() {
    return registry.slice();
  }

  function clear() {
    registry.length = 0;
  }

  return { register, getSchema, schemas, clear };
});
