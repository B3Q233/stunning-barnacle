# UBA（Uplift-guided Budget Allocation）攻击复现 设计文档

> 论文：*Uplift Modeling for Target User Attacks on Recommender Systems*（WWW '24，
> DOI 10.1145/3589334.3645403）；官方代码：https://github.com/Wcsa23187/UBA
> 理解文档：`papers/UBA/UBA_understanding.md`（v1；本次复现同时补齐其"官方代码交叉验证"章节）
> 状态：设计冻结，开始实施

## 1. 目标与范围

### 1.1 本次实现什么

在仓库既有攻击体系（`TPA/attacks/{random,bandwagon,pgd,tpa,advinject}`）下新增
`TPA/attacks/uba/`，复现论文的 **UBA 框架**：把"给哪些目标用户分配多少个假用户"
建模为 uplift 优化问题，再用后端攻击者的画像规则实例化成假档案 `D_f`。

交付必须满足（硬约束）：

- `AGENTS.md` §6「攻击模块与批量攻击复现硬性模板」：固定目录/入口、统一 config
  结构、run_tag 实验隔离、评估协议、批量适配自查；
- `paper-code-implementation` 技能的攻击模板（classify → data → model 三阶段 +
  每阶段验证门禁 + `docs/DESIGN.md`、`docs/USAGE.md`）；
- `TPA/docs/config-template.unified.yaml` 为唯一配置 schema（新增键必须同步模板，
  否则 `tests/test_config_canonical.py::ActiveConfigsSchemaTest` 失败）。

### 1.2 本次不做什么

- **不实现论文的后端攻击者 AIA / AUSH / Leg-UP**：它们是独立论文的攻击方法，官方
  实现依赖 TensorFlow 1.x + GAN 训练脚本。UBA 是 model-agnostic 的预算分配框架，
  本次以仓库已有的**数据层画像规则**（`template_user` / `popular` / `random` 三种
  filler 语义）作为后端实例化，保证 UBA 核心（处理效应估计 + 预算分配）可独立
  复现与验证。
- **不做多目标物品联合分配**：论文定义单目标物品 i；本版本 `attack.target_items.count`
  必须为 1，>1 直接报错，不静默降级。
- **不复现论文 Table 1 的绝对数值**：后端攻击者不同（AIA/AUSH/Leg-UP vs 本仓库数据层
  画像），数值不可比；本次复现的是**同一后端下的相对增益**：
  `baseline → +Target → +UBA(w/o S_φ)`，并在配置允许时给出 `+UBA(w/ S_φ)` 支路。

## 2. 论文与官方代码的事实核对（本次交叉验证结论）

理解文档 v1 标记"官方代码尚未交叉验证"。本次已克隆官方仓库（`tmp/UBA-official`，
非入库）逐项核对，结论如下表；该表同时回填到 `papers/UBA/UBA_understanding.md`。

| 项目 | 论文写法（B 级） | 官方代码（A 级） | 本次实现取值 |
|---|---|---|---|
| 预算 N | 100（Table 1 说明 / Appendix B.1） | `--attack_num 300`（示例脚本） | 采用论文 N=100（`attack.num_fake_users`） |
| 单用户上限 H | 6（Appendix B.2） | `DPA.py` 中 `x = np.zeros((50, 6))`，实际候选 t=0..5 | 采用论文 H=6（t=0..6），并在 DESIGN 标注差异 |
| 目标用户数 | 50 | `evaluator.py` / `aushplus.py` 硬编码 50 个 ml-1m 用户 id | 采用 50（`attack.uba.target_users.count`） |
| 处理效应（w/o S_φ） | Y ≈ α·((A')³_{u,i})^β，α=β=1 | `DPA.py` 载入 `a3path{item}.npy` 但未参与 DP（遗留） | 实现 A'³ 三跳路径计数（纯数据，α/β 可配） |
| 处理效应（w/ S_φ） | 每个 t_u 等量分配 → 攻击 S_φ → Top-K 命中，重复 E≈10 次 | `DPA.py`：命中判据 `0 < rank <= 20`，最后 `x/10` | 命中判据 `rank ≤ hit_k`（默认 20，与官方一致），重复 `repeats` 次取均值 |
| 预算分配 | Algorithm 1 背包式 DP | `DPA.py::pack5`（分组 0/1 背包 + 回溯） | 标准分组背包 DP，与官方 pack5 在随机小实例上逐例对齐（单测） |
| 三种模式 | 论文对比 AIA/AUSH/Leg-UP、+Target、+UBA | `--way 1/2/3`：1=从 20% 可访问数据随机取模板；2=目标用户均匀分配；3=UBA 分配 | `attack.uba.allocation.strategy ∈ {random_all, uniform_target, uba}` |
| filler 数量 | 论文未给 | `--filler_num` 默认 36 | 采用 36（`attack.filler_size`，[官方代码]） |
| 攻击者可见交互比例 | 默认 20%（Appendix B.1） | 模板池来自 `%s_50user_train.csv`（20% 用户子集） | `random_all` 模板池 = 训练集用户（`accessible_ratio` 可限制比例） |
| 评估指标 | HR@K（K=10/20）、NDCG@K、MRR@K；目标用户群 | `evaluator.py`：逐用户 hr_10/20/50/100 + 目标 50 用户群 HR + pred_shift | 仓库口径 `target_hr@K`/`target_ndcg@K` + UBA 专属「目标用户群 HR@K/NDCG@K」报告段 |
| 数据集 | ML-1M / Amazon-Game / Yelp（显式评分 >3 → 1） | 已处理 csv（user_id,item_id,rating） | 复用仓库 `models/*/data/processed/ml100k/meta.pkl`（隐式 0/1，10-core） |

**关键差异提示（交付时告知用户）**：

1. 论文/官方代码是"显式评分 → 隐式标签"（rating>3 → 1）；仓库 ml100k 预处理已是
   隐式成对数据（`train.txt` 为 `user item`），本次不做评分映射；
2. 官方处理效应估计的命中判据是 `rank ≤ 20`（不是论文主表的 HR@10），本实现用
   `attack.uba.treatment.hit_k` 显式暴露该选择；
3. 官方 H 的候选档位是 6 个（t=0..5），论文写 H=6；本实现取论文值并把该差异写进
   DESIGN/USAGE 与 `stats.json`（`max_per_user`）。

## 3. 架构与数据流

### 3.1 阶段划分（在模板三阶段上增加 estimate 阶段）

```
classify ─→ estimate ─→ data ─→ model
（物品分层）  （处理效应 Y） （分配 T* + 注入 D_f） （中毒训练 + 评估）
```

- `classify`：复用 `attacks/classify_common.py`，按训练集交互数划分
  popular / ordinary / cold，产物
  `attacks/uba/data/rec_freq/{dataset}/{model}_top{k}.json`（共享缓存，不按 run_tag 隔离）。
- `estimate`（**UBA 自定义阶段**，按 AGENTS §6.1 允许，需在 USAGE 说明）：计算处理效应
  矩阵 Y ∈ R^{|U_t|×(H+1)}，产物
  `attacks/uba/data/estimate/{dataset}/{model}/item{target}_h{H}_{method}.json`（共享缓存）。
  `method=path` 为纯数据计算；`method=surrogate` 需要代理模型（import 模型代码）。
- `data`：读 estimate 缓存（`path` 分支缺失时自行计算并回写）→ 预算分配 → 按后端画像
  规则实例化假档案 → 注入 → 产出
  `data/poisoned/{dataset}/{model}/{tag}/{meta.pkl, profiles.json, stats.json}` + `latest.json`。
- `model`：warm-start 投毒训练 + clean/poisoned 对比评估 → `outputs/{dataset}/{model}/{tag}/`
  （checkpoints / history.json / eval_log.csv / uba_comparison.md）。

### 3.2 模块边界（文件职责）

| 文件 | 职责 | 是否 import 模型代码 |
|---|---|---|
| `classify.py` | 物品三档分类缓存 | 否 |
| `uplift.py` | 纯算法：目标用户选择、共现类别代理、A'³ 三跳路径处理效应、分组背包 DP、三种分配策略 | 否 |
| `generate.py` | 数据阶段：目标物品/用户 → 处理效应（读缓存或回退计算）→ 分配 → 画像构造 → 注入 → 产物 | 仅 `method=surrogate` 时惰性 import `estimate.py` |
| `estimate.py` | estimate 阶段：代理模型模拟实验估计 Y（w/ S_φ 支路），写缓存 | 是（torch + registry） |
| `fit.py` | 中毒模型训练 + 对比评估 + 报告 | 是 |
| `evaluate.py` | 薄壳：转出 `evaluation/attack_eval.py` + UBA 专属目标用户群指标 | 是（仅推理） |
| `registry.py` | 薄壳：转出 `models/registry.py` | 间接 |
| `run.py` | CLI `--config/--mode/--tag`，模式 `classify|estimate|data|model|both|all` | 间接 |

### 3.3 UBA 核心算法（uplift.py）

**A. 目标用户选择**（论文 §5；仓库数据无类别元数据，用共现邻域作代理 [ai]）

```
类别代理 C(i) = 与目标物品 i 共现次数最高的 category_size 个物品（含 i 自身）
候选用户 = {u | u 未交互 i，且 0 < |I_u ∩ C(i)| < max_category_interactions}
strategy = cooccurrence（默认，取候选中 |I_u| 降序前 count 个，种子确定）
         | specified（显式 ids）
         | random（未交互 i 的用户中随机 count 个）
```

论文原判据"与目标物品同类别、类别交互数 < 10"在仓库数据（无 genre）下不可直接执行；
共现邻域是同一语义的可计算代理，且保持"轻交互用户更易被撬动"的筛选方向。

**B. 处理效应：三跳路径代理（w/o S_φ）**

A = [[0, D], [Dᵀ, 0]]，A' = [[0, D'], [D'ᵀ, 0]]，D' = [D_r; D_f]。
只计算需要的行/列，避免构造 A'³ 全矩阵：

```
(A'³)_{u,i} = Σ_{u'} <D'_u, D'_{u'}> · D'_{u',i}
            = [(D'[U_t] @ D'.T) @ D'[:, i]]
```

shape 变换：`D'[U_t] (|U_t|, M')` @ `D'.T (M', M')` → `(|U_t|, M')`，再与
`D'[:, i] (M',)` 列乘 → `(|U_t|,)`。Y_{u,t} = α·((A')³_{u,i})^β。

对每个 t ∈ {0..H} 各算一次：给**每个目标用户**都加 t 个假用户（论文的"等量分配"），
D_f 由画像规则生成。

**C. 处理效应：代理模型模拟（w/ S_φ）**

```
for t in 0..H:
    for e in 1..E:                    # E=repeats，每轮换随机种子
        D_f = 每个目标用户等量 t 个假用户（画像规则 + 该轮种子）
        在 D_r ∪ D_f 上重训代理模型 S_φ（可选从 surrogate.checkpoint 热启动）
        rank(u, i) = 干净训练集过滤后全量排序中 i 的位置
        hit[u] = 1 if 0 < rank <= hit_k else 0
    Y[:, t] = mean_e hit
```

**D. 预算分配（Algorithm 1）**

```
max_T Σ_{u∈U_t} v_u(T_u)  s.t.  Σ_u T_u <= N, 0 <= T_u <= H, T_u ∈ Z
分组背包：dp[i][c] = max_{t<=min(H,c)} dp[i-1][c-t] + v_i(t)，回溯得 T*
```

数值与官方 `DPA.py::pack5` 在随机小实例上逐例对齐（单测，A 级代码交叉验证）。

**E. 三种分配策略（复现论文对比行）**

| 策略 | 含义 | 对应论文/官方 |
|---|---|---|
| `uba` | DP 最优 T* | +UBA |
| `uniform_target` | N 在目标用户上尽量均匀分配 | +Target |
| `random_all` | 从可访问用户池随机取 N 个模板用户 | 原始后端攻击者（baseline，官方 way=1） |

### 3.4 画像实例化（generate.py）

每个假用户的画像 = filler 物品集合 + 目标物品（隐式反馈，去重）：

| `attack.uba.profile.filler_source` | filler 来源 | 用途 |
|---|---|---|
| `template_user`（默认） | 该假用户对应模板用户（UBA/uniform_target 即目标用户本人）的历史交互物品 | 论文"后端攻击者基于真实用户画像生成"的等价物 |
| `popular` | classify 缓存的 popular 物品池 | bandwagon 语义 |
| `random` | 全量物品均匀采样 | random 语义 |

模板用户交互数不足 `filler_size` 时，从全局热门池补齐（不重复填充同一物品）。

### 3.5 评估协议

- 仓库口径（沿用 `evaluation/attack_eval.py`）：目标物品 HR@K / NDCG@K（合格用户 =
  训练集未交互目标物品的用户，clean/poisoned 统一用干净训练集过滤）；模型效用
  recall@K / ndcg@K 作投毒代价参考。
- UBA 专属段（论文口径）：**目标用户群** U_t 上的 HR@K / NDCG@K，只在最终报告与
  `stats.json` 中给出（不进 `evaluation.metrics`，避免污染共享 BestTracker 契约）。
- 选优指标仍按仓库硬规范：`target_ndcg@K` 主、`target_hr@K` 副，
  `checkpoint_mode: per_metric`。

## 4. 配置（canonical 键，必须同步 config-template.unified.yaml）

```yaml
dataset / mode / seed / k / run_tag
model: {name, overrides}
classification: {popular_ratio, medium_ratio}
attack:
  name: uba
  num_fake_users: 100          # N（论文默认）
  ratio: null                  # num_fake_users 缺省时才用
  filler_size: 36              # [官方代码]
  target_items: {strategy, category, count, ids}
  uba:
    treatment: {method, max_per_user, repeats, hit_k, alpha, beta}
    allocation: {strategy}
    target_users: {strategy, count, category_size, max_category_interactions, ids, accessible_ratio}
    profile: {filler_source}
checkpoint: {clean}
warm_start: {enabled, checkpoint}
surrogate: {enabled, name, checkpoint, training: {...}}
training: {optimizer, epochs, batch_size, lr, weight_decay, neg_ratio, eval_every, device, num_workers, persistent_workers}
evaluation: {metrics, checkpoint_mode, report_model_utility}
output: {dir}
```

新增 canonical 键只有 `attack.uba.*` 一块（禁止另起顶层同名键）；其余全部复用模板
既有 canonical 键。

## 5. 错误处理与边界

- `attack.target_items.count > 1` → 明确报错（本版本只支持单目标）。
- `method=surrogate` 且 `surrogate.enabled=false` → 报错并提示配置代理模型。
- 目标用户候选为空 / 少于 `count` → 用可用数量 + 打印告警（不静默改变语义）。
- `num_fake_users` 为 0 或负数 → 报错。
- 假用户 uid 从 `num_users` 起连续编号；注入时同步扩展按 uid 索引的容器，且假用户
  不会落入真实用户统计。
- 注入后硬断言：`train_pairs_after - train_pairs_before == Σ|profile|`。

## 6. 测试策略（stdlib unittest，`TPA/tests/`）

| 测试文件 | 覆盖 |
|---|---|
| `test_uba_uplift.py` | DP 与暴力枚举最优值一致；DP 与官方 pack5 口径一致；三跳路径与朴素 A'³ 稠密计算一致；目标用户选择约束/确定性；分配策略预算与语义 |
| `test_uba_generate.py` | 合成 meta 上的 data 阶段：注入数量断言、profiles/stats schema、假 uid 起始、目标物品出现在每个画像、filler 不重复 |
| `test_batch_registry.py`（改） | `uba` 已注册且 classify/generate/fit 可调用 |
| `test_attack_fit_consistency.py`（改） | `attacks/uba/fit.py::main` 定义 `attack_name` |
| `test_config_canonical.py`（既有，不新增） | 新增 config 叶子路径必须在统一模板内（自动覆盖） |

## 7. 交付清单

- [ ] `TPA/attacks/uba/` 固定入口 9 文件 + `uplift.py` / `estimate.py`
- [ ] `config.yaml`（canonical，含来源标注）与 `docs/USAGE.md` / `docs/DESIGN.md`
- [ ] `TPA/docs/config-template.unified.yaml` 同步 `attack.uba.*`
- [ ] classify / estimate(path) / data / model 四阶段各跑通一次
- [ ] `evaluation/attack_eval.py` 增加 `uba` 报告名映射
- [ ] batch registry 注册 `uba`（path 分支可批量；surrogate 分支需预建缓存并写入 DESIGN）
- [ ] 全量回归 `python -m unittest` 通过
- [ ] 回填 `papers/UBA/UBA_understanding.md` 的官方代码交叉验证表
