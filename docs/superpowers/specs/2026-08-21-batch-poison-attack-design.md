# 批量投毒攻击（配置生成器 + 分层采样 + 结果整合）设计文档

> 日期：2026-08-21
> 状态：待用户确认
> 关联计划：`docs/superpowers/plans/2026-08-21-batch-poison-attack.md`

## 1. 背景与目标

仓库已有四个投毒攻击（bandwagon / pgd / random / tpa），均走 classify → data → model
三阶段，每个原子实验由一个攻击 `config.yaml` 驱动、以 run_tag 隔离产物。用户需要：

1. 用**一个批跑配置文件**指定攻击方法、clean 模型权重（w_clean）、数据集、受害模型
   （model_t）与训练策略等基础信息；
2. 批量程序作为**配置文件生成器**：用 clean 模型统计物品出现在用户 top@k 中的频次，
   划分为 popular / normal / cold 三层，每层指定采样数 K，生成对应的原子投毒配置；
3. **结果整合**：批量结果放在批量程序自己的 output 目录，以 `YYYY-MM-DD-HH:MM` 为一次
   批量标签，原子实验命名为 `{攻击方法}_{数据集}_{模型}_top{k}_{层}_item{id}`，
   按层聚合（平均 ± 标准差）。

原子投毒实验语义（用户定义）：

```
输入: model_t（受害模型结构）+ w_clean（clean 权重）+ data_clean + fun_attack + 攻击物品 I_t
流程: Inter = fun_attack(model, data_clean)        # 生成投毒交互
      data_poison = data_clean ∪ Inter             # 形成投毒数据集
      model_t 载入 w_clean，在 data_poison 上训练
输出: 目标物品最优 recall@10 / ndcg@10（取训练过程 BestTracker 最优值）
```

## 2. 敏捷范围

- **v1（本计划，mini 投毒实验）**：单数据集（ml100k）× 单划分 × bandwagon 攻击 ×
  lightgcn 受害模型；分层采样 + 逐目标训练 + 按层平均，跑通全链路。
- **v2（后续迭代，不在本计划）**：多数据集 × 多划分（`training.split_seed` 网格）、
  其他攻击方法（pgd/random/tpa）、更大规模。
- v1 **不改动任何现有攻击代码**，纯新增目录与测试，风险最小。

## 3. 目录结构

```
TPA/attacks/batch/
├── config.yaml          # 批跑配置（= 攻击配置 + sampling 扩展段）
├── generator.py         # 分层采样 + 原子配置生成
├── runner.py            # 调度：classify 一次 + 逐原子 data+model
├── aggregate.py         # results.csv + summary.md 整合
├── run.py               # CLI 编排（generate/run/aggregate/all）
├── docs/USAGE.md        # 使用文档
└── docs/DESIGN.md       # 设计文档
```

批量输出布局：

```
attacks/batch/output/{batch_tag}/            # batch_tag = YYYY-MM-DD-HH-MM
├── config.yaml                              # 批跑配置快照
├── configs/                                 # 生成的原子配置
│   └── {run_tag}.yaml
├── runs/{run_tag}/                          # 原子 fit 输出
│   ├── checkpoints/  history.json  attack_comparison.md  config.yaml
├── results.csv                              # 每原子实验一行
└── summary.md                               # 按层 mean±std + clean 基线
```

## 4. 批跑配置 schema（config.yaml）

```yaml
# 以下为攻击配置（与 attacks/bandwagon/config.yaml 同构，可直接单独运行）
dataset: ml100k
mode: all
seed: 42
model:
  name: lightgcn
  overrides: {}
classification:
  k: 10
  popular_ratio: 0.2
  checkpoint: models/lightgcn/outputs/checkpoints/latest.pt   # w_clean
attack:
  name: bandwagon
  ratio: 0.03
  filler_size: 20
  target_items: {strategy: specified, ids: []}   # 由生成器逐目标覆写
warm_start:
  enabled: true
  checkpoint: models/lightgcn/outputs/checkpoints/latest.pt   # w_clean
training: {epochs: 5, batch_size: 256, lr: 0.001, weight_decay: 0.0001,
           neg_ratio: 1, device: cuda}
evaluation:
  k: 10
  metrics: [target_ndcg@10: upper, target_hr@10: upper, recall@10: upper, ndcg@10: upper]
  report_model_utility: true
output:
  dir: attacks/batch/output

# 以下为批量扩展段（生成器专属，原子配置中剔除）
sampling:
  tiers: [popular, normal, cold]   # 可选子集
  per_tier: 3                      # 每层采样目标物品数 K
  strategy: random                 # random（seed 固定）| first（按层内顺序取前K）
  seed: 42
```

## 5. 生成规则（generator.py）

- 分类缓存：复用攻击自带缓存
  `attacks/{attack}/data/rec_freq/{dataset}/{model}_top{k}.json`
  （由 runner 调 `attacks.{attack}.classify.main` 生成一次，与目标无关，只跑一次；
  缓存与独立运行共享，确定性与 checkpoint/dataset/model/k 绑定）。
  缓存缺失且未生成时，`--mode generate` 报错并提示先跑 classify。
- 层内采样：`strategy=random` 用 `sampling.seed` 的 `random.Random` 固定抽样；
  `strategy=first` 按层列表顺序取前 K。空层跳过并告警。
- 原子配置 = 批跑配置深拷贝，剔除 `sampling` 段，覆写：
  - `attack.target_items = {strategy: specified, ids: [item_id]}`
  - `run_tag = {attack}_{dataset}_{model}_top{k}_{tier}_item{item_id}`（经 sanitize）
  - `output.dir = attacks/batch/output/{batch_tag}/runs`
  - `classification.checkpoint` / `warm_start.checkpoint` 保持 w_clean 路径
- 每个原子实验独立 run_tag → 中毒数据（攻击目录 data/poisoned/，不入库）与
  fit 产物（batch output）互不覆盖。

## 6. 调度规则（runner.py）

- classify 只执行一次（同一 batch 内所有原子实验共享缓存）；
- 对每个原子配置顺序执行 data（generate）+ model（fit）两阶段（in-process 调用
  攻击模块的 `main(config)`，不新建子进程；单进程顺序执行避免 GPU 争用）；
- 支持 `--dry-run`（只生成不执行）、`--skip-classify`（复用已有缓存）、
  `--max-targets`（冒烟限流）。

## 7. 结果整合（aggregate.py）

- 每原子实验读取 `runs/{run_tag}/history.json` 的 `best` 段：
  `target_ndcg@{k}`、`target_hr@{k}`、`recall@{k}`、`ndcg@{k}`；
- `results.csv` 列：`run_tag, tier, target_item, target_hr@{k}, target_ndcg@{k},
  recall@{k}, ndcg@{k}`；
- `summary.md`：按 tier 对 `target_hr@{k}` / `target_ndcg@{k}` 求 mean ± std，
  附 clean 基线（用 w_clean 在 clean 数据上的 recall@{k} / ndcg@{k}，批量开始前算一次）；
- tier / item 从 run_tag 解析（正则 `_top\d+_(popular|normal|cold)_item(\d+)$`）。

## 8. 测试策略（stdlib unittest，CPU，不训练模型）

- `tests/test_batch_generator.py`：给定假分类缓存 → 采样数量/确定性、原子配置字段
  （run_tag、target ids、output.dir、剔除 sampling）、空层跳过、写文件；
- `tests/test_batch_aggregate.py`：给定假 runs 目录 → results.csv 行数/列、
  summary mean±std 正确、缺 best 文件的 run 跳过；
- `tests/test_batch_config.py`：批跑配置 schema 校验（缺 sampling.per_tier 等报错）；
- E2E mini 验证（手工步骤，非单测）：准备 ml100k clean lightgcn checkpoint →
  跑 mini 批量（epochs=1、per_tier=1、tiers=[cold]）→ 校验产物齐全、按层平均正确。

## 9. 验收标准

- 单测全部通过（`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_* -v`）；
- mini 批量跑通：`attacks/batch/output/{batch_tag}/` 下 configs/、runs/、
  results.csv、summary.md 齐全，configs 数量 = Σ各层实际采样数，runs 数量一致；
- summary.md 含按层 target_hr@10 / target_ndcg@10 的 mean±std 与 clean 基线；
- 不修改任何现有攻击/模型代码（v1 纯新增）。
