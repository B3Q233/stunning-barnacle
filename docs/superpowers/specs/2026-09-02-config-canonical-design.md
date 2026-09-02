# 配置文件同义键统一——设计文档

> 日期：2026-09-02
> 状态：待人工审阅（brainstorming 已确认方案 a+b：物理统一 + 兼容层）
> 前置文档：`TPA/docs/config-template.unified.yaml`（canonical 参考模板）

## 1. 背景与目标

仓库 15 份活跃 config.yaml 中同一用途存在多个名字（`use_cuda` vs
`training.device`、`data.dataset` vs `dataset`、`training.eval_every` vs
`evaluation.eval_every`、`output_dir` vs `output.dir`、`lambda_reg` vs
`weight_decay`、`clean_checkpoint`/`classification.checkpoint` vs
`warm_start.checkpoint` 等）。目标：

1. 所有活跃配置物理统一为 `TPA/docs/config-template.unified.yaml` 定义的
   canonical 键，不再出现旧别名。
2. 新增中心兼容层，在配置加载时把历史别名映射为 canonical，保证旧配置
   /历史调用仍可运行（兼容输入允许，但新文件禁止）。
3. AGENTS.md 增加硬性要求：复现/新增模型与攻击时参考统一模板，禁止再引入
   同义键。
4. 新复现模型/攻击若必须新增配置项，必须同步修改
   `TPA/docs/config-template.unified.yaml`（补充 canonical 键；若与历史写法
   相关还需补别名映射），不允许只在单模块 config.yaml 里加键。
5. 全量回归通过。

## 2. Canonical 映射表（实现兼容层与配置文件改写的唯一依据）

| 用途 | Canonical | 旧别名（兼容输入） |
|---|---|---|
| 数据集名 | 顶层 `dataset` | `data.dataset` / `experiment.dataset` / `override.dataset` |
| 设备 | `training.device: cpu\|cuda\|cuda:N` | `use_cuda: bool` / `cuda_id`（CLI）/ 顶层 `device`（batch 注入） |
| 评估频率（模型/攻击训练） | `training.eval_every` | `evaluation.eval_every` |
| 评估频率（surrogate 训练） | `surrogate.training.eval_every` | 无别名（surrogate 域内自有键） |
| 评估 K | 顶层 `k` | `evaluation.k` / `training.k` / `classification.k` |
| 干净模型权重 | `checkpoint.clean`（canonical） | `clean_checkpoint` / `classification.checkpoint`（旧别名） |
| warm-start 专用权重 | `warm_start.checkpoint`（**可选 canonical**，非旧别名；取值 = warm_start.checkpoint or checkpoint.clean） | 无 |
| 假用户数 | `attack.num_fake_users` / `attack.ratio` | `attack.n_fakes`：**值 >1 → `num_fake_users`（绝对数）；0<值≤1 → `ratio`（比例）**。历史实现本身按此混合语义解释，不得无条件并入 ratio |
| filler 数 | `attack.filler_size` | `filler_num`（CLI）/ `profile_size`（显式评分语义）；**冲突规则见 §3.4** |
| 流行度切分（排名比例语义，random/bandwagon/pgd/tpa） | `classification.popular_ratio` / `medium_ratio` | 无（percentile 属另一种语义，不映射） |
| 流行度切分（交互数百分位语义，advinject） | `classification.percentile_head` / `percentile_upper_torso` / `percentile_lower_torso` | `popular_percentile` / `torso_percentile` / `tail_percentile`（纯改名，不做数值换算） |
| surrogate 架构名 | `surrogate.name` | `surrogate.model_name` |
| surrogate L2 | `surrogate.training.weight_decay`（唯一 canonical） | `surrogate.l2` / 平铺 `surrogate.weight_decay`（已废弃，禁止写入新配置） |
| surrogate 训练超参 | `surrogate.training.{epochs,lr,weight_decay,batch_size,hidden_dims,weight_alpha,unroll_steps}` | `surrogate.epochs/lr/l2/batch_size/hidden_dims/weight_alpha` 等平铺旧键 |
| 训练侧 L2 | `training.weight_decay` | `training.lambda_reg`（wmf） |
| 输出目录 | `output.dir` | 顶层 `output_dir`（advinject） |
| 训练/验证划分比例 | `data.val_ratio`（**可选 canonical**，缺省 0.05，不强制每份 config 都写） | 攻击 fit.py 历史硬编码 0.05（改为读配置并保留缺省 0.05） |
| 目标选择 | `attack.target_items.{strategy,category,count,ids}` | `n_target_items` / `attack.target_items`（旧列表式） |
| 目标分层区间（advinject zone 语义） | `attack.target_items.zone: head\|upper_torso\|lower_torso\|tail` | `target_item_popularity`（纯改名，不换算） |
| 对抗优化 | `attack.adv.{epochs,lr,momentum,proj_threshold,click_targets,attack_type}` | `attack.adv_epochs/adv_lr/adv_momentum/proj_threshold/click_targets/attack_type` |
| surrogate unroll | `surrogate.training.unroll_steps`（唯一 canonical） | `attack.unroll_steps` / 平铺 `surrogate.unroll_steps`（已废弃） |

## 3. 兼容层设计（b）：显式 canonicalization pipeline

位置：`TPA/training/config_utils.py`（现有 `apply_k`/指标展开所在模块）。

### 3.1 Pipeline 结构

```text
legacy config
      │
      ▼
┌──────────────────┐
│ canonicalize_config │
└───────┬──────────┘
        │
   ┌────┴────┬──────────┬─────────────┐
   ▼         ▼          ▼             ▼
rename   restructure  semantic     conditional
aliases   nesting    conversion     fallback
   │         │          │             │
   └────┬────┴──────────┴─────────────┘
        ▼
canonical config
        │
        ▼
schema validation（owner/domain 校验）
        │
        ▼
model / attack 业务代码（只读 canonical）
```

`canonicalize_config` 不是"一个巨大的 alias dict"，而是按阶段注册的显式
处理器列表（每个处理器 = 一个小函数，职责单一、可单测）：

1. **rename aliases**：仅做同层改名（如 `surrogate.model_name` →
   `surrogate.name`、`surrogate.l2` → `surrogate.training.weight_decay`、
   `training.lambda_reg` → `training.weight_decay`、顶层
   `output_dir` → `output.dir`）。
2. **restructure nesting**：结构搬迁（如 `data.dataset` /
   `experiment.dataset` → 顶层 `dataset`；`surrogate.epochs` →
   `surrogate.training.epochs`；`attack.adv_*` → `attack.adv.*`）。
3. **semantic conversion**：语义换算（如 `use_cuda: bool` →
   `training.device`；`n_fakes` 按值分支）。percentile/zone 属纯改名
   （rename），**不做数值换算**（见 §2）。
4. **conditional fallback**：条件回退（如 `warm_start.checkpoint` 缺省
   回退 `checkpoint.clean`；`surrogate.training.unroll_steps` 缺省回退旧
   `attack.unroll_steps` / 平铺 `surrogate.unroll_steps`）。

对外只暴露两个函数：

```python
def canonicalize_config(cfg: dict) -> dict:
    """深拷贝输入，按 §3.1 的处理器顺序产出 canonical 配置；不修改入参。"""

def load_config(path: PathLike) -> dict:
    """yaml.safe_load 后调用 canonicalize_config，返回规范化配置。"""
```

### 3.2 优先级规则：canonical 优先

- canonical 与 legacy 同时出现时，**canonical 生效，legacy 被忽略**并打印
  WARNING（`[config] canonical 键已存在，忽略旧键 X`）。
- 多个 legacy 别名指向同一 canonical 且同时出现：
  - 值相同：直接采用；
  - 值不同：**抛错**（`ValueError`），禁止静默二选一。
- 兼容层遇到被消费的旧别名打印一次性 WARNING
  （`[config] 旧键 X 已映射为 Y`），不静默。

### 3.3 n_fakes 语义转换（必须按值分支）

已核验代码（`attacks/advinject/generate.py`）：`n_fakes > 1` 为绝对数量，
`0 < n_fakes <= 1` 为用户比例。因此：

```text
n_fakes > 1   → attack.num_fake_users = int(n_fakes)
0 < n_fakes ≤ 1 → attack.ratio = n_fakes
```

兼容层不需要知道 clean user 数即可完成该分支；比例真正换算为人数发生在
业务代码（`generate.py` 用 `meta["num_users"]` 计算），此逻辑不变。

### 3.4 filler 冲突规则

- `filler_num` → `attack.filler_size`（CLI 时代键）。
- `profile_size`（显式评分语义）→ `attack.filler_size`（若同时存在
  `attack.explicit_rating.profile`，则以 explicit_rating.profile 为准）。
- 多个旧键同时出现且值不同：按 §3.2 抛错；值相同则采用。

### 3.5 兼容层边界：只做输入兼容，不做业务读取 API

- 业务代码（模型/攻击/批量）**只允许读取 canonical 键**。
- legacy 读取只允许出现在 `training/config_utils.py`（及其单测）。
- 禁止业务代码出现：

```python
cfg.get("data", {}).get("dataset")
cfg.get("use_cuda")
cfg.get("clean_checkpoint")
cfg.get("output_dir")
cfg.get("evaluation", {}).get("eval_every")
```

- 所有模块配置加载统一走 `load_config`；散落的 `load_yaml_config` 改为
  调用它（或委托）。

## 4. 物理统一范围（a）

### 4.1 配置文件（16 份）

- `TPA/models/{cml,itemae,itemcf,lightgcn,mf,multvae,ncf,wmf}/config.yaml`
  - 统一：`data.dataset` → 顶层 `dataset`；`evaluation.k` → 顶层 `k`；
    `evaluation.eval_every` → `training.eval_every`；
    `training.lambda_reg` → `training.weight_decay`（wmf）；
    `evaluation.metrics` 保留但 `{k}` 由顶层 k 展开。
  - wmf 额外：`raw_data_path/processed_data_path/val_ratio/source_model`
    保留在 `data.*`（非重复键）。
- `TPA/attacks/{random,bandwagon,pgd,tpa,advinject}/config.yaml`
  - advinject：`use_cuda` 删除；`output_dir` → `output.dir`；
    `n_fakes` 按 §3.3 值分支（>1 → `attack.num_fake_users`；
    0<值≤1 → `attack.ratio`）；`n_target_items` / 列表式 `target_items` →
    `attack.target_items.{count,ids,strategy}`；`target_item_popularity`
    → `attack.target_items.zone`（head/upper_torso/lower_torso/tail）；
    `adv_*` → `attack.adv.*`；`unroll_steps` →
    `surrogate.training.unroll_steps`；
    `surrogate.{epochs,lr,l2,batch_size,hidden_dims,weight_alpha}` →
    `surrogate.training.*`（`surrogate.name` 保持不变）；
    `classification.popular_percentile/torso_percentile/tail_percentile`
    → `classification.percentile_head/percentile_upper_torso/
    percentile_lower_torso`（纯改名）。
  - random/pgd/tpa/bandwagon：`clean_checkpoint` /
    `classification.checkpoint` → `checkpoint.clean`；
    `warm_start.checkpoint` 保留但缺省回退 `checkpoint.clean`。
- `TPA/attacks/batch/config.yaml`、`config.yelp2018_fast.yaml`
  - `experiment.dataset` → 顶层 `dataset`；`experiment.seed` → 顶层 `seed`；
    `override.*` 内部字段同步改 canonical。
- `TPA/docs/config-template.unified.yaml`
  - 按 §2 更新：新增 `checkpoint.clean` 段；surrogate 训练超参收拢到
    `surrogate.training.*`（删除顶层 `surrogate.weight_decay` /
    `surrogate.unroll_steps`）；映射表同步。

### 4.2 代码读取点（按别名出现处逐个修改）

- `attacks/advinject/common.py`：`use_cuda`、`n_fakes`、`n_target_items`、
  `target_item_popularity`、`adv_*`、`proj_threshold`、`click_targets`、
  `surrogate.*` 平铺键、`output_dir` 读取改 canonical。
- `attacks/advinject/{generate.py,data.py,classify.py,run.py}` 同步。
- `attacks/{random,pgd,tpa,bandwagon}/fit.py`、`generate.py`、`classify.py`：
  `clean_checkpoint` / `classification.checkpoint` → `checkpoint.clean`；
  `warm_start.checkpoint` 回退逻辑。
- `attacks/batch/{generator.py,runner.py,aggregate.py,utils.py}`：
  `experiment.dataset/seed` → 顶层 `dataset/seed`。
- `models/wmf/{dataset.py,train.py,report.py,config_keys.py}`：
  `lambda_reg` → `weight_decay`；`evaluation.eval_every` →
  `training.eval_every`；`KEY_*` 常量改为 canonical 名并保留兼容别名读取。
- `models/lightgcn/{dataset.py,train.py}`、`models/mf/{dataset.py,train.py}`
  `evaluation.eval_every` 读取改 canonical。
- `models/revisit_common.py` / `models/revisit_training.py`：
  `data.dataset` 兼容读取（顶层 dataset 优先）。
- `models/registry.py` / 各模型 `dataset.py`：dataset 统一取顶层 `dataset`。
- `evaluation/attack_eval.py`、`training/framework.py`：统一经
  `load_config` 加载与读取 canonical 键。

### 4.3 测试

- 新增 `TPA/tests/test_config_canonical.py`：
  - 对每份活跃 config：`load_config` 后断言不存在旧别名键（集合检查）；
  - legacy 输入兼容：构造含 `use_cuda/lambda_reg/output_dir/...` 的 dict，
    断言 `canonicalize_config` 输出为 canonical；
  - percentile 纯改名（popular_percentile → percentile_head 等，数值
    保持不变）与 `use_cuda` 布尔派生。
  - canonical 与 legacy 同时出现：canonical 生效、legacy 被忽略（WARNING）；
  - 多 legacy 冲突（不同值）抛 `ValueError`；
  - `n_fakes` 按值分支：`0.01 → attack.ratio=0.01`、
    `50 → attack.num_fake_users=50`；
  - **反向 schema 约束（owner/domain）**：每个 canonical 键必须有明确
    归属域，`training.k`、`dataset` 出现在 `training` 下等错误结构直接失败
    （校验"合法叶子路径集合"而非只查扁平 union）。
  - **语义唯一性**：同一语义只允许一个 canonical path；schema 断言
    `surrogate.weight_decay`、`surrogate.unroll_steps` 等已废弃平铺键
    不在合法叶子路径集合内，且每个语义在集合中至多出现一次。
  - `data.val_ratio` 为可选键：缺省 canonicalize 不注入、业务层回退 0.05；
    断言各活跃 config 不会因本改造被机械添加该键。
- 更新受影响旧测试（`test_wmf_dataset/training/als_fit`、
  `test_portable_paths`、`test_history_completeness`、batch 测试等）的
  配置键为 canonical。

## 5. AGENTS.md 增补

§3 代码与工程规范新增一条：

```text
- 配置规范：复现/新增模型与攻击时，config.yaml 必须参照
  `TPA/docs/config-template.unified.yaml` 的 canonical 键组织；禁止新增
  同义键；历史别名仅由 `training/config_utils.py` 的兼容层接受，新文件
  不得使用。新复现模型/攻击如需新增配置项，必须同步更新该模板
  （canonical 键与别名映射），模板未覆盖的新键不允许合入。
```

§6.2 末尾加一行：`config 结构以 TPA/docs/config-template.unified.yaml 为
唯一参考模板（canonical 键见其映射表）。`

## 6. 验证

1. 全量 `unittest`（192+ 新增）通过：PowerShell 下用
   `python -m unittest discover -s tests -p "test_*.py" -v`。
2. 配置冒烟：每个攻击 `--mode data`（或 batch `--mode generate --dry-run`）
   与 wmf/lightgcn preprocess 临时目录冒烟。
3. `git grep` 旧别名（`use_cuda|output_dir:|lambda_reg|popular_percentile|
   n_fakes|clean_checkpoint|evaluation.eval_every|data.dataset:|
   experiment.dataset:`）在活跃 config 内无命中；代码内仅兼容层保留。
4. schema 测试把"每份活跃 config 的全部叶子键 ⊆ 统一模板 union"设为硬约束：
   任何模块新增配置键而不同步模板时，测试直接失败。
5. 最终状态验收（与 §3.5 一致）：
   - 活跃 config.yaml：只出现 canonical；
   - 业务代码（models/attacks/batch/evaluation/training 除 config_utils 外）：
     只读取 canonical；
   - `training/config_utils.py`：允许出现 legacy（兼容层唯一归属）；
   - tests：允许构造 legacy 输入做兼容性测试。

## 7. 明确不做（Out of scope）

- 不修改 outputs/ 历史快照中的 config.yaml。
- 不修改 `.codex/skills/**` 资产 config（另议题）。
- 不做 `model.topk`（itemcf）更名——语义不同，仅作命名提醒。
- 不新增第三方依赖；不改变任何模型/攻击的训练语义与默认数值。

## 8. 风险

- 读取点分散：以“统一 load_config + canonical 常量”为界，逐文件替换并靠
  schema 测试兜底；发现漏读点时先加失败测试再改。
- batch 生成器依赖 `experiment.*` 结构：改造后原子配置顶层仍输出 canonical
  （dataset/mode/seed/k/...），其回归测试需同步更新。
- advinject 与其余攻击配置差异最大，优先处理并单测其 legacy 兼容。

## 9. 实施阶段顺序（供 writing-plans 使用）

按以下顺序逐阶段实施，每阶段结束跑对应测试并提交，禁止"16 份 config 与
代码一起乱改"：

1. **Phase 1 建立规范**：完善 `TPA/docs/config-template.unified.yaml`
   （含 `checkpoint.clean` 段与 `surrogate.training.*`）；实现
   `canonicalize_config` / `load_config`；新增 legacy→canonical 单测与
   canonical schema（owner/domain）单测。此时不改任何业务 config。
   Phase 1 内部再拆为：
   1.1 冻结 canonical schema（模板为唯一事实源，含"一语义一路径"去重）；
   1.2 实现 `canonicalize_config`；
   1.3 实现 `load_config`；
   1.4 legacy compatibility 单测；
   1.5 path/domain schema 单测；
   1.6 对一个旧 config（advinject）做人工验证；
   1.7 确认旧配置 canonicalization 不改变语义（默认值/行为前后一致）后再进入 Phase 2。
2. **Phase 2 AdvInject**：历史键最多、结构最复杂，作为兼容层压力测试；
   改 advinject config 与读取代码，先保证 advinject 全部测试通过。
3. **Phase 3 简单攻击**：random / bandwagon / pgd / tpa——主要是
   `checkpoint.clean`、顶层 `dataset`、`training.eval_every` 等简单迁移。
4. **Phase 4 batch**：单独小心处理；`experiment.*` / `override.*` 承担
   batch generator 内部协议，保留其生成语义，只在读取边界映射 canonical，
   不粗暴删除内部结构。
5. **Phase 5 models**：WMF → LightGCN → MF → revisit 系
   （cml/itemae/itemcf/multvae/ncf）逐个迁移并跑对应测试。
6. **Phase 6 全局收口**：全量回归 + `git grep` 验收（§6.5 的四个"只允许/
   允许"状态），提交收尾。
