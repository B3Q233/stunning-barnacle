# Batch 批量投毒攻击系统（v1）设计文档

> 设计冻结版见 `docs/superpowers/specs/2026-08-21-batch-poison-attack-design.md`

## 1. 两层架构

- **Atomic Attack**：一个目标物品的一次完整投毒实验（classify → generate → fit →
  evaluation），由攻击自身 config.yaml 驱动；
- **Batch**：配置生成 + 批量调度 + 结果整合，不实现任何攻击算法。

Batch 通过 `registry.py` 获取攻击插件（`AttackSpec(name, config_path, classify,
generate, fit)`），新增攻击只需"建目录 + register + 改 attack.name"。

## 2. 配置继承（Deep Merge）

```
P1 Generator 运行时字段（target_items / run_tag / output.dir）
P2 override（用户覆盖）
P3 Batch 配置（experiment.* 展开后合并）
P4 攻击默认配置（attacks/{attack}/config.yaml）
```

`utils.deep_merge`：嵌套 dict 递归合并，list/标量整体覆盖，不修改入参。
原子配置保持攻击期望的扁平结构（dataset/mode/seed 在顶层）。

## 3. 调度与产物

- classify 只执行一次（公共分类缓存 `cache/classification/{dataset}/{model}/top{k}/`，
  按训练集交互数划分 popular/ordinary/cold，与攻击算法和模型无关；
  攻击侧 `ordinary` 归一化为 `normal`）；
- generate + fit 串行、单 GPU；fit 产物从 staging 移动到分层目录
  `runs/{攻击}_{数据集}_{模型}_top{k}/{层}/item{id}/`（不改 fit.py）；
- 每次 Batch 独立时间目录，互不覆盖；`logs/runner.log` 记录调度过程。

## 4. 结果整合

- `results.csv`：每原子实验一行（attack/dataset/model/tier/item/target_hr@k/
  target_ndcg@k/recall@k/ndcg@k）；
- `summary.md`：按层 mean±std + Clean Model Utility（w_clean 在 clean 数据上的
  recall@k / ndcg@k，投毒代价基线）；
- `meta.json`：实验索引（batch_tag/attack/dataset/model/topk/tiers/per_tier/
  total_runs/seed）。

## 5. v2 迭代方向

- 多数据集 × 多划分（`split_seed` 网格）；
- tpa 分类缓存格式适配；
- 失败重试 / 断点续跑（按 run_tag 跳过已完成原子实验）。
