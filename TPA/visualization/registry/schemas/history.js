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

  class HistorySchema extends VisualizationSchema {
    constructor() {
      super();
      this.name = 'history';
      this.title = 'Training History';
      this.type = 'line';
      this.x = 'history[].epoch';
      this.series = {
        'train_loss': 'history[].train_loss',
        'val_loss': 'history[].val_loss',
        'recall@10': 'history[].recall@10',
        'ndcg@10': 'history[].ndcg@10',
        'target_hr@10': 'history[].target_hr@10',
        'target_ndcg@10': 'history[].target_ndcg@10',
      };
    }

    match(json) {
      return !!(json && Array.isArray(json.history));
    }
  }

  register(new HistorySchema());
  return HistorySchema;
});
