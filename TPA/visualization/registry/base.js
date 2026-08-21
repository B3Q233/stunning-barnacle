'use strict';
(function (global, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    global.TPAVisualizer = global.TPAVisualizer || {};
    global.TPAVisualizer.base = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {

  class VisualizationSchema {
    constructor() {
      this.name = '';
      this.title = '';
      this.type = 'line';      // line | bar | metric
      this.x = null;           // 路径，如 history[].epoch
      this.series = {};        // { 显示名: 路径, ... }
    }

    match(json) {
      return false;
    }
  }

  return { VisualizationSchema };
});
