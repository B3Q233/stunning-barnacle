# PGD（投影梯度上升投毒攻击）设计文档

> 对应论文理解：本攻击实现的是 Li et al., *Data Poisoning Attacks on
> Factorization-Based Collaborative Filtering* (NIPS 2016) 的 PGA（投影梯度上升，
> 工程上常称 PGD）算法。本项目把论文 §3-§4.1 的攻击模型、梯度公式与参考仓库
> （fuying-wang/Data-poisoning-attacks-on-factorization-based-collaborative-filtering，
> A 级代码）落地为"指定受害模型（MF / LightGCN）+ 指定目标物品 + 对比评估"的
> 可运行攻击。

## 0. 事实来源与参考层级

| 来源 | 层级 | 用途 |
|------|------|------|
| PGD.md（论文原文） | B 级 | 攻击模型（§3）、PGA 更新式（Eq.10）、链式法则（Eq.11）、KKT 隐式梯度（§4.1）、效用定义（Eq.7-9）|
| fuying-wang GitHub 仓库（main.py / compute_grad.py / ALS_optimize.py）| A 级 | 梯度逐元素实现、ALS 求解细节、超参（s_t=0.2、Λ=1、λ=5e-2、m_iters=10）|
| 本项目 bandwagon / tpa 攻击 | 已验收 | 三阶段模板（classify → data → model）、yaml 配置结构、HR@K/NDCG@K 评估协议 |

## 1. 攻击定义

【论文明确写出】攻击者加入 αm 个恶意用户，每个恶意用户最多给 B 个物品打分，
评分有界 [−Λ, Λ]（§3，Eq.6 可行集 M）。攻击者最大化自身效用
R(M̂, M)（Eq.6），其中 M̂ = Θ_λ(M̃; M) 是联合数据上训练出的模型预测。

【论文明确写出】本文采用 PGA 更新（Eq.10，Algorithm 1）：

```
M̃^(t+1) = Proj_M( M̃^(t) + s_t · ∇_{M̃} R )
```

投影算子 = 把每个恶意用户的评分向量截断到 [−Λ, Λ]（l∞ 球），支撑集（B 个物品）
初始化时随机选取并固定。梯度按链式法则（Eq.11）：

```
∇_{M̃} R = ∇_{M̃} Θ · ∇_Θ R
```

- ∇_Θ R（易求，Appendix A）：availability（Eq.7）与 integrity（Eq.8）对预测矩阵
  M̂ 的梯度，混合效用 μ1·R^av + μ2·R^in（Eq.9）。
- ∇_{M̃} Θ（难求）：用交替最小化（ALS）的 KKT 条件近似（§4.1）：
  ∂v_j/∂M̃_ij ≈ (λ_V I + Σ_V^(j))^{-1} ũ_i。

## 2. 本项目实现要点

### 2.1 效用（utility）

- integrity（push 指定目标物品）：R^in = Σ_u Σ_{j∈J0} w(j)·M̂_uj（Eq.8），
  默认 w=2（论文实验值），J0 由用户指定（`attack.target_items.ids`）。
- availability：R^av = ‖R_{Ω^C}(M̂ − M̄)‖_F²（Eq.7），M̄ 为干净模型预测。
- 混合：μ1·av + μ2·in（Eq.9），默认 μ1=-1、μ2=1（论文 Fig.1(d) "light trace"：
  push 目标的同时尽量少扰动其他推荐，兼顾隐蔽性）。

### 2.2 梯度引擎（两个引擎共用论文 §4.1 的代数形式）

基础梯度（论文式）：

```
∇_{r̃_fj} R = (Σ_m ∂R/∂M̂_mj · u_m)^T (λ_V I + Σ_{i∈N_j} u_i u_i^T
              + Σ_{f∈F_j} ũ_f ũ_f^T)^{-1} ũ_f
```

- `engine: als`（model.name=mf 时 auto 选择）：在每个 PGD 外层迭代内，用 ALS
  在（正常数据 ∪ 恶意数据）上求解 Eq.(4) 的联合固定点（与参考仓库
  ALS_optimize.py 一致；正则采用论文 KKT 的无 n 缩放形式），再用上式求梯度。
  该引擎与论文 §4.1 完全对应（【A/B 级】）。
- `engine: neighbor`（model.name=lightgcn 时 auto 选择）：【AI 推断补全】
  LightGCN 不是 ALS 学习者，无法直接套用 KKT。本实现把干净 LightGCN 的最终
  嵌入（U, V，即"模型权重"）作为代理：
  - 假用户嵌入 ũ_f = (λ_u I + Σ_{j∈S_f} v_j v_j^T)^{-1} Σ_j r̃_fj v_j
    （ALS 式用户闭式解，v_j 用干净物品嵌入）——模拟 warm-start 后新用户的嵌入；
  - 物品侧一阶响应 V_j^resp = V_j + A_j^{-1} Σ_f r̃_fj ũ_f ——模拟投毒后物品
    嵌入沿恶意评分方向的线性化移动。

交叉项（默认开启，config `utility.cross_target: true`）：

```
∇_{r̃_fj'} R ⊇ (Σ_{j∈S_f} r̃_fj h_j)^T A_f^{-1} v_j'
  其中 h_j = A_j^{-1} Σ_m ∂R_mj u_m，A_f = λ_u I + Σ_{j∈S_f} v_j v_j^T
```

【AI 推断补全】论文 §4.1 在计算 ∂v_j/∂M̃_ij 时把 ũ_i 视为常数（"approximately
compute"），因此纯 integrity 效用下 filler 评分的梯度恒为 0，PGD 只能把目标
评分推向 +Λ（参考仓库同样如此）。本项目补上一阶链式交叉项（∂v_j/∂r̃_fj' 经 ũ_f
传播），让 PGD 对 filler 的选择/评分有真实优化信号——即"让假用户嵌入对准真实
用户方向，从而更有效地推动目标物品"。关闭可设 `cross_target: false` 回到论文式。

### 2.3 与论文的差异（需人工确认）

| 差异点 | 论文 | 本项目 | 依据 |
|--------|------|--------|------|
| 支撑集 | B 个物品完全随机（Algorithm 1）| `include_target: true` 时固定包含目标物品，其余 B−1 个从流行池初始化 | [AI] 适配"指定目标物品 push"需求；隐式反馈评估下目标必须出现在画像中才会被推动 |
| 评分语义 | 显式评分（MovieLens 移轴到 [−2,2]）| 隐式反馈：正常交互=1，恶意评分由 PGD 优化（[−Λ,Λ]）；注入时只保留物品集合 | [AI] 与 bandwagon/tpa 统一评估协议（BPR + HR@K/NDCG@K）|
| learner | 交替最小化（显式）| 受害模型为 BPR-MF / LightGCN（warm-start 投毒训练）| 用户指定 MF/LightGCN；PGD 画像在 ALS/代理梯度下生成，再用于 BPR 训练（代理差距是投毒文献常规做法）|
| 梯度 | 论文近似 | 论文近似 + 一阶交叉项 | [AI] 见 2.2 |

## 3. 参数与来源等级

| 参数 | 默认值 | 来源等级 | 说明 |
|------|--------|----------|------|
| 假用户数 | ratio=0.01（6 个）| [AI] | 与 bandwagon 一致；论文 α 可调 |
| filler_size B | 20 | [AI] | 参考仓库 B=25、bandwagon filler=20，取 20 |
| iterations | 10 | [paper]/[A] | 参考仓库 m_iters=10 |
| step_size s_t | 0.2 | [A] | 参考仓库 s_t=0.2 |
| lambda_rating Λ | 1.0 | [A] | 参考仓库 Lamda=1（论文为 [−Λ,Λ]）|
| lambda_u / lambda_v | 0.05 | [A] | 参考仓库 lamda_u=lamda_v=5e-2 |
| utility μ1 / μ2 | -1.0 / 1.0 | [paper] | Eq.9；Fig.1(d) light-trace 设置 |
| w_target | 2.0 | [paper] | 论文实验 w_j0=2 |
| cross_target | true | [AI] | 一阶交叉项开关 |
| engine | auto | [AI] | mf→als（论文精确），lightgcn→neighbor 代理 |
| warm_start | 开 | 用户指定 | 用干净模型权重初始化中毒模型 |

## 4. 模块设计（三阶段，与攻击模板一致）

| 模式 | 职责 | 是否新建模型 |
|------|------|--------------|
| classify（classify.py）| 加载干净模型 → 全量评分 → 每用户 Top-K → 推荐频次 → 流行/普通/冷门缓存 | 否 |
| data（generate.py）| 选目标（默认指定）→ PGD 梯度上升生成假画像 → 注入训练集 | 否（PGD 需要加载干净模型权重计算梯度）|
| model（fit.py）| 按 model.name 新建受害模型（可选 warm-start）→ 投毒训练 → 对比评估 | 是 |

数据流：

```
干净模型 checkpoint ─┬─ classify → rec_freq.json（流行/普通/冷门）
                     ├─ PGD（模型权重 + 目标物品）→ poisoned meta.pkl
                     └─ warm-start → 投毒模型 → 对比报告 pgd_comparison.md
```

## 5. 评估协议

- 模型效用：all-ranking recall@K / ndcg@K（与 LightGCN 一致，过滤训练集已交互），
  用于投毒代价检查（不得显著下降）。
- 攻击效果：对"训练集未交互过目标物品"的合格用户，报告目标物品的 HR@K /
  NDCG@K / 命中用户数 / 平均排名；clean 与 poisoned 统一用干净训练集过滤。

## 6. 实测结果（ml100k，目标物品 251，假用户 6，B=20）

| 模型 | Clean HR@20 | Poisoned HR@20 | Clean NDCG@20 | Poisoned NDCG@20 | 平均排名 Clean→Poisoned |
|------|------------|----------------|---------------|------------------|------------------------|
| MF（engine=als）| 0.0087 | 0.0173（+99%）| 0.0037 | 0.0063（+70%）| 676.5 → 497.9 |
| LightGCN（engine=neighbor）| 0.0208 | 0.0329（+58%）| 0.0095 | 0.0131（+38%）| 856.6 → 541.8 |

投毒代价：MF recall@20 0.2092→0.2091、LightGCN 0.2356→0.2434，均无显著下降。

## 7. 缺口与风险（需人工确认）

- 【论文未提及】filler 初始化池：默认取模型推荐频次 Top 20% 流行物品（与 bandwagon
  一致）；`init: random` 可改为全物品均匀采样（更接近论文 Algorithm 1）。
- 【AI 推断】neighbor 引擎是 LightGCN 的一阶线性化代理，不是 GCN 训练过程的精确
  隐式梯度；如需精确需对 BPR-SGD 训练做 unrolled/implicit 微分（超出本任务范围）。
- 【AI 推断】交叉项推导基于一步响应模型，忽略高阶耦合；数值上已确认其提供 filler
  梯度并驱动收敛。
- 目标物品选择：`strategy: specified` 由用户指定；`category/coldest/random` 可用。

## 8. 配置键一致性审计（交付前检查）

全项目扫描 `attacks/pgd/*.py` 中所有 `config.get()` / `config[...]` 调用，汇总
使用的键如下，均与 `attacks/pgd/config.yaml` 的定义一致，无同义多拼写
（如 `data_dir` 与 `processed_data_path` 并存）：

- 顶层：`dataset` / `mode` / `seed` / `model.name` / `model.overrides` /
  `clean_checkpoint` / `output.dir` / `run_tag`（实验隔离，缺省当前时间）
- `classification`：`k` / `popular_ratio` / `batch_size` / `checkpoint`
- `attack`：`num_fake_users`（可选）/ `ratio` / `filler_size` /
  `target_items.{strategy,category,count,ids}` / `pgd.*`
- `attack.pgd`：`iterations` / `step_size` / `lambda_rating` / `lambda_u` /
  `lambda_v` / `als_iterations` / `engine` / `init` / `include_target` /
  `utility.{mu1,mu2,w_target,cross_target}`
- `warm_start`：`enabled` / `checkpoint`
- `training`：`lr` / `epochs` / `batch_size` / `weight_decay` / `neg_ratio` /
  `device` / `k` / `eval_every`
- `evaluation`：`report_model_utility` / `metrics`（支持 upper/lower 方向标注，
  指标名中的 @K 是评估 K 的唯一权威）/ `checkpoint_mode`（per_metric 默认 | single）

`models/mf` 的键（`data.dataset` / `model.emb_dim` / `model.init_method` /
`training.*` / `evaluation.k|eval_every`）同样与 `models/mf/config.yaml` 一致。

## 9. 动态数据结构验证（仅投毒类需要）

- 假用户注入后 `num_users` 由 608 → 614，所有按 uid 索引的容器（`user_items`、
  `embedding` 表）均在构造时使用扩展后的行数，实际训练通过
  （`uid >= original_n_users` 的边界 batch 已覆盖）。
- warm-start 嵌入迁移前断言：干净 checkpoint 行数 == 干净用户数 + 物品数，
  迁移后物品行整体偏移假用户数，假用户行保持随机初始化（fit.py
  `transfer_clean_embeddings`），两次真实运行均通过。
