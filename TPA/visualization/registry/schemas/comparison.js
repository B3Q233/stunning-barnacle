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

  class ComparisonSchema extends VisualizationSchema {
    constructor() {
      super();
      this.name = 'comparison';
      this.title = 'Attack Comparison';
      this.type = 'metric';
      this.x = null;
      this.series = {
        'Clean Recall@10': 'model_utility.clean.recall@10',
        'Poisoned Recall@10': 'model_utility.poisoned.recall@10',
        'Clean NDCG@10': 'model_utility.clean.ndcg@10',
        'Poisoned NDCG@10': 'model_utility.poisoned.ndcg@10',
      };
    }

    match(json) {
      return !!(json && json.model_utility && json.target_metrics);
    }
  }

  register(new ComparisonSchema());
  return ComparisonSchema;
});
