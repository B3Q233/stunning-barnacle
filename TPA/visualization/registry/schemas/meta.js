'use strict';
(function (global, factory) {
  if (typeof module === 'object' && module.exports) {
    const { VisualizationSchema } = require('../base.js');
    const { register } = require('../index.js');
    module.exports = factory(VisualizationSchema, register);
  } else {
    const { VisualizationSchema } = global.TPAVisualizer.base;
    const { register } = global.TPAVisualizer.registry;
    factory(VisualizationSchema, register);
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function (VisualizationSchema, register) {

  class MetaSchema extends VisualizationSchema {
    constructor() {
      super();
      this.name = 'meta';
      this.title = 'Batch Meta';
      this.type = 'metric';
      this.x = null;
      this.series = {
        'attack': 'attack',
        'dataset': 'dataset',
        'model': 'model',
        'topk': 'topk',
        'per_tier': 'per_tier',
        'total_runs': 'total_runs',
        'seed': 'seed',
      };
    }

    match(json) {
      return !!(json && json.batch_tag !== undefined && json.attack);
    }
  }

  register(new MetaSchema());
  return MetaSchema;
});
