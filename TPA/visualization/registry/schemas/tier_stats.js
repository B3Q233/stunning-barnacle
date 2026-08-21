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

  class TierStatsSchema extends VisualizationSchema {
    constructor() {
      super();
      this.name = 'tier_stats';
      this.title = 'Tier Stats (mean)';
      this.type = 'bar';
      this.x = null;
      this.series = {
        'Popular HR@10': 'tiers.popular.target_hr@10.mean',
        'Normal HR@10': 'tiers.normal.target_hr@10.mean',
        'Cold HR@10': 'tiers.cold.target_hr@10.mean',
        'Popular NDCG@10': 'tiers.popular.target_ndcg@10.mean',
        'Normal NDCG@10': 'tiers.normal.target_ndcg@10.mean',
        'Cold NDCG@10': 'tiers.cold.target_ndcg@10.mean',
      };
    }

    match(json) {
      return !!(json && json.tiers && json.k !== undefined);
    }
  }

  register(new TierStatsSchema());
  return TierStatsSchema;
});
