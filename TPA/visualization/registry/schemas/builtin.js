'use strict';
// 浏览器 file:// 无法 fetch，内嵌 schema/*.json 的镜像（规范源为 schema/ 目录）。
(function (global, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    global.TPAVisualizer = global.TPAVisualizer || {};
    global.TPAVisualizer.builtin = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {

  return [
    {
      "id": "fp_5dd775",
      "title": "Training History",
      "type": "line",
      "x": "history[].epoch",
      "series": {
        "train_loss": "history[].train_loss",
        "val_loss": "history[].val_loss",
        "recall@10": "history[].recall@10",
        "ndcg@10": "history[].ndcg@10",
        "target_hr@10": "history[].target_hr@10",
        "target_ndcg@10": "history[].target_ndcg@10"
      },
      "fingerprint": [
        "history", "history[]", "history[].epoch", "history[].ndcg@10",
        "history[].recall@10", "history[].target_hr@10",
        "history[].target_ndcg@10", "history[].train_loss", "history[].val_loss"
      ]
    },
    {
      "id": "fp_0ecb77",
      "title": "Attack Comparison",
      "type": "metric",
      "x": null,
      "series": {
        "Clean Recall@10": "model_utility.clean.recall@10",
        "Poisoned Recall@10": "model_utility.poisoned.recall@10",
        "Clean NDCG@10": "model_utility.clean.ndcg@10",
        "Poisoned NDCG@10": "model_utility.poisoned.ndcg@10"
      },
      "fingerprint": [
        "model_utility", "model_utility.clean",
        "model_utility.clean.ndcg@10", "model_utility.clean.recall@10",
        "model_utility.poisoned", "model_utility.poisoned.ndcg@10",
        "model_utility.poisoned.recall@10", "target_metrics",
        "target_metrics.clean", "target_metrics.poisoned"
      ]
    },
    {
      "id": "fp_8c7d38",
      "title": "Tier Stats (mean)",
      "type": "bar",
      "x": null,
      "series": {
        "Popular HR@10": "tiers.popular.target_hr@10.mean",
        "Normal HR@10": "tiers.normal.target_hr@10.mean",
        "Cold HR@10": "tiers.cold.target_hr@10.mean",
        "Popular NDCG@10": "tiers.popular.target_ndcg@10.mean",
        "Normal NDCG@10": "tiers.normal.target_ndcg@10.mean",
        "Cold NDCG@10": "tiers.cold.target_ndcg@10.mean"
      },
      "fingerprint": [
        "batch_tag", "k", "tiers", "tiers.cold",
        "tiers.cold.ndcg@10", "tiers.cold.ndcg@10.mean",
        "tiers.cold.ndcg@10.n", "tiers.cold.ndcg@10.std",
        "tiers.cold.recall@10", "tiers.cold.recall@10.mean",
        "tiers.cold.recall@10.n", "tiers.cold.recall@10.std",
        "tiers.cold.target_hr@10", "tiers.cold.target_hr@10.mean",
        "tiers.cold.target_hr@10.n", "tiers.cold.target_hr@10.std",
        "tiers.cold.target_ndcg@10", "tiers.cold.target_ndcg@10.mean",
        "tiers.cold.target_ndcg@10.n", "tiers.cold.target_ndcg@10.std",
        "tiers.normal", "tiers.normal.ndcg@10",
        "tiers.normal.ndcg@10.mean", "tiers.normal.ndcg@10.n",
        "tiers.normal.ndcg@10.std", "tiers.normal.recall@10",
        "tiers.normal.recall@10.mean", "tiers.normal.recall@10.n",
        "tiers.normal.recall@10.std", "tiers.normal.target_hr@10",
        "tiers.normal.target_hr@10.mean", "tiers.normal.target_hr@10.n",
        "tiers.normal.target_hr@10.std", "tiers.normal.target_ndcg@10",
        "tiers.normal.target_ndcg@10.mean", "tiers.normal.target_ndcg@10.n",
        "tiers.normal.target_ndcg@10.std", "tiers.popular",
        "tiers.popular.ndcg@10", "tiers.popular.ndcg@10.mean",
        "tiers.popular.ndcg@10.n", "tiers.popular.ndcg@10.std",
        "tiers.popular.recall@10", "tiers.popular.recall@10.mean",
        "tiers.popular.recall@10.n", "tiers.popular.recall@10.std",
        "tiers.popular.target_hr@10", "tiers.popular.target_hr@10.mean",
        "tiers.popular.target_hr@10.n", "tiers.popular.target_hr@10.std",
        "tiers.popular.target_ndcg@10", "tiers.popular.target_ndcg@10.mean",
        "tiers.popular.target_ndcg@10.n", "tiers.popular.target_ndcg@10.std"
      ]
    },
    {
      "id": "fp_135ed1",
      "title": "Batch Meta",
      "type": "metric",
      "x": null,
      "series": {
        "attack": "attack",
        "dataset": "dataset",
        "model": "model",
        "topk": "topk",
        "per_tier": "per_tier",
        "total_runs": "total_runs",
        "seed": "seed"
      },
      "fingerprint": [
        "attack", "batch_tag", "dataset", "model", "per_tier", "seed",
        "tiers", "tiers[]", "topk", "total_runs"
      ]
    }
  ];
});
