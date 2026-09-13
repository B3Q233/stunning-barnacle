# UBA（Uplift-guided Budget Allocation）攻击 —— 设计文档

> 论文：*Uplift Modeling for Target User Attacks on Recommender Systems*（WWW '24，
> DOI 10.1145/3589334.3645403）
> 官方代码：https://github.com/Wcsa23187/UBA（本次复现已克隆核对，见 §5）
> 理解文档：`papers/UBA/UBA_understanding.md`

## 1. 攻击定义

UBA 是**目标用户攻击的预算分配框架**：攻击者能注入的假用户总数固定为 N，目标是
让目标物品 i 尽可能多地进入**指定目标用户群** U_t 的 Top-K 列表。它不自己发明画像
生成器，而是"给定后端攻击者 + 每个目标用户分到的假用户数 t_u"，由后端攻击者实例化
假档案 D_f。

$$\max_{T}\ \sum_{u\in U_t} Y_{u,i}^{\theta^*}\!\big(D_f(T)\big)\quad
\text{s.t.}\ \sum_{u\in U_t} t_u \le N,\ t_u \ge 0,\ t_u \in \mathbb{Z},\ t_u \le H$$

其中 $Y_{u,i}^{\theta^*}(D_f(T))$ 是"注入 D_f(T) 后在 $D=[D_r;D_f]$ 上重训得到的
模型 $\theta^*$ 下，目标物品 i 进入用户 u 的 Top-K"的指示量（取期望即命中概率）。

## 2. 模块结构与本仓库的对应

| 论文组件 | 本仓库实现 | 说明 |
|---|---|---|
| 目标用户选择（§5） | `uplift.select_target_users` | 三种策略：cooccurrence/specified/random |
| 处理效应 w/o $S_\phi$（§4.1 方法二） | `uplift.three_hop_path_effect` | A'³ 三跳路径计数，稀疏化简 |
| 处理效应 w/ $S_\phi$（§4.1 方法一） | `estimate.treatment_effect_surrogate` | 代理模型模拟实验，写共享缓存 |
| 预算分配（Algorithm 1） | `uplift.dp_allocate` | 分组背包 DP + 回溯 |
| 三种对比行 | `uplift.allocate` | uba / uniform_target / random_all |
| 后端攻击者实例化 | `generate.build_fake_profiles` | filler_source：template_user/popular/random |
| 中毒重训 + 评估 | `fit.py` + `evaluate.py` | warm-start 投毒训练 + 双口径评估 |

阶段：`classify`（物品分层）→ `estimate`（处理效应 Y）→ `data`（分配 T* + 注入 D_f）
→ `model`（投毒训练 + 评估）。`estimate` 是按 AGENTS §6.1 允许的攻击自定义阶段。

## 3. 关键实现决策

### 3.1 三跳路径只用需要的行列（内存安全）

直接构造 $A'$ 再求 $A'^3$ 需要 $(M+K+N)^2$ 稠密矩阵（ml100k 已 $6906^2$，
Gowalla 量级会直接爆内存）。由二部图结构可化简为

$$(A'^3)_{u,i}=\sum_{u'}\langle D'_u,D'_{u'}\rangle\,D'_{u',i}
=\big[(D'[U_t]\,D'^{\top})\,D'_{:,i}\big]$$

矩阵变换：`D'[U_t] (|U_t|, N)` ⨯ `D'ᵀ (N, M+K)` → `(|U_t|, M+K)`；再与
`D'[:, i] (M+K,)` 列乘 → `(|U_t|,)`。单测用朴素稠密 $A'^3$ 逐元素比对（
`tests/test_uba_uplift.py::ThreeHopPathTest`）。

### 3.2 目标用户选择：用共现邻域代理"类别"

论文 §5 的判据是"与目标物品**同类别**、类别交互数 < 10、未交互目标物品"。仓库
`meta.pkl` 只有 user/item 二元组（无 genre/类别字段），因此用**共现邻域**作为类别
代理：$\mathcal{C}(i)$ = 与 i 共现次数最高的 `category_size` 个物品，用户 u 的
"类别交互数" = $|I_u \cap \mathcal{C}(i)|$。

保留的语义方向：候选用户必须"在该类别有交互但不多"（$0 < |I_u\cap\mathcal{C}(i)| < 10$），
即论文刻意挑选的轻交互、易被撬动用户。`strategy: specified` 可完全绕开该代理，
用于复现官方代码里硬编码的目标用户列表。

### 3.3 Y 矩阵的 t=0 列是"跳过该用户"的基线选项

论文 Eq.2 对每个用户取 $t_u\in\{0..H\}$ 中的**一个**值，因此 $Y_{u,i}(t_u=0)$ 表示
"不给他分配假用户时的命中概率"，必须参与比较：

- `path` 支路：$t=0$ 即干净图上的 A'³ 计数（一般非 0）；
- `surrogate` 支路：$t=0$ 即"训练在干净数据上的代理模型"的命中率。

官方 `DPA.py` 把 $x[:,0]$ 恒置 0（模拟实验只填 $t=1..5$）。两者在 $Y[:,0]=0$ 时
**逐例等价**（本仓库单测用 300 个随机实例验证：`test_matches_official_pack5` +
本地批量比对），差别只在"基线命中非 0 时是否还值得投放预算"这一语义。

### 3.4 分配策略统一为"模板用户序列"

官方 way 1/2/3 最终都归约为 `idx = [uid, uid, ...]`，再 `train_data_array[idx]`
取真实用户画像作为生成器输入。本实现把三种策略统一成 `template_users` 序列：

| strategy | template_users | 语义 | 对应 |
|---|---|---|---|
| `uba` | 每个目标用户重复 t_u 次（DP 输出） | 最优分配 | +UBA |
| `uniform_target` | 每个目标用户重复 ⌊N/\|U_t\|⌋ 次（余量随机补齐） | 平均分配 | +Target |
| `random_all` | 从可访问用户池有放回抽 N 次 | 随机模板 | 原始后端攻击者（官方 way=1） |

这样注入代码只有一条路径，不会出现"某种策略绕过画像规则"的静默差异。

### 3.5 评估双口径

| 口径 | 指标 | 统计对象 | 用途 |
|---|---|---|---|
| 仓库统一 | `target_hr@K` / `target_ndcg@K` | 所有未交互目标物品的合格用户 | checkpoint 选优（`target_ndcg@K` 主） |
| 仓库统一 | `recall@K` / `ndcg@K` | 测试集全体用户 | 投毒代价（模型效用） |
| 论文 Table 1 | `target_user_hr@K` / `target_user_ndcg@K` | 目标用户群 U_t | `target_user_metrics.md`（不影响 BestTracker） |

## 4. 参数来源等级

| 参数 | 值 | 来源 |
|---|---|---|
| `attack.num_fake_users`（N） | 100 | [paper] Table 1 说明 / Appendix B.1 |
| `attack.uba.treatment.max_per_user`（H） | 6 | [paper] Appendix B.2 |
| `attack.uba.treatment.repeats`（E） | 10 | [paper] §4.1 |
| `attack.uba.treatment.hit_k` | 20 | [官方代码] `DPA.py` 命中判据 `rank <= 20` |
| `attack.uba.treatment.alpha` / `beta` | 1.0 / 1.0 | [paper] Appendix B.1（调参 {0.5,1}、{0.3,1}） |
| `attack.uba.target_users.count` | 50 | [paper] §5 |
| `attack.uba.target_users.max_category_interactions` | 10 | [paper] §5 |
| `attack.uba.target_users.category_size` | 20 | [ai] 仓库数据无类别字段，共现邻域规模的工程取值 |
| `attack.uba.target_users.accessible_ratio` | 1.0 | [ai]；论文 Appendix B.1 考察 20%（设为 0.2 即对齐） |
| `attack.filler_size` | 36 | [官方代码] `--filler_num` 默认值（论文未给出） |
| `attack.uba.profile.filler_source` | template_user | [paper] "由后端攻击者基于真实用户画像生成"的等价物 |
| `surrogate.name` | mf | [paper] "代理取最简 MF"（官方 `--surrogate` 默认 WMF，同为最简分解模型） |
| `surrogate.training.*` | 30 epoch / lr 1e-3 / wd 1e-4 | [ai] 论文未给，沿用仓库 MF 默认量级 |

## 5. 与官方代码的交叉验证（本次复现新增）

| 项目 | 论文（B 级） | 官方代码（A 级） | 本实现 | 差异处置 |
|---|---|---|---|---|
| 预算 N | 100 | 示例脚本 300 | 100 | 取论文值，配置可改 |
| 单用户上限 H | 6 | `x = np.zeros((50, 6))` → t=0..5 | 6（t=0..6） | 取论文值，`stats.json` 记录 `max_per_user` |
| 命中判据 | 主表 HR@10/20 | `0 < rank <= 20` | `hit_k: 20` | 与官方一致，显式暴露为配置 |
| 预算分配 | Algorithm 1（分组背包） | `DPA.py::pack5`（in-place 0/1 背包 + 回溯） | 分组背包 DP | $Y[:,0]=0$ 时**逐例等价**（单测 + 300 随机实例批量比对）；$Y[:,0]\neq0$ 时本实现按论文 Eq.2 保留基线项 |
| 三种模式 | +Target / +UBA | `--way 1/2/3` | `allocation.strategy` | 语义一一对应 |
| 后端攻击者 | AIA / AUSH / Leg-UP | 各自 GAN 训练脚本 | 数据层画像规则 | **范围外**：后端攻击者属独立论文，见 §6 |
| filler 数 | 未给 | 默认 36 | 36 | 与官方一致 |

## 6. 已知限制

1. **后端攻击者范围**：本版本用数据层画像规则（template_user/popular/random）实例化
   假档案，不含 AIA/AUSH/Leg-UP 的 GAN/白盒优化。因此论文 Table 1 的**绝对数值不可比**，
   可复现的是同一后端下的相对增益（baseline → +Target → +UBA）。
2. **单目标物品**：`attack.target_items.count > 1` 直接报错（多目标需要跨物品联合分配，
   论文未定义）。
3. **w/ S_φ 支路成本**：训练次数 = (H+1) × E（默认 70 次代理重训），与论文
   Appendix B.2 报告的 52–64 分钟量级一致；用小配置冒烟（`epochs=1, repeats=1`）。
4. **显式评分语义**：论文的数据集是显式评分映射为隐式（rating>3 → 1）；仓库
   ml100k 预处理已是隐式成对数据，本实现不做评分映射（AGENTS §6.6 的显式评分槽位
   仅在后端攻击者需要评分画像时才启用，本版本未触发）。
5. **批量适配**：`attacks/batch/registry.py` 已注册 uba；`treatment.method=path` 可直接
   批量跑；`method=surrogate` 需要先跑 `--mode estimate` 生成缓存（否则 generate 会在
   缺失缓存时惰性重算，耗时很长）。

## 7. 产物

| 阶段 | 路径 | 内容 |
|---|---|---|
| classify | `data/rec_freq/{dataset}/{model}_top{k}.json` | 三档分类 + 交互数（共享缓存） |
| estimate | `data/estimate/{dataset}/{model}/item{i}_h{H}_...json` | 处理效应 Y + 实验元数据（共享缓存） |
| data | `data/poisoned/{dataset}/{model}/{tag}/` | meta.pkl / profiles.json / stats.json / config.yaml 快照 |
| model | `outputs/{dataset}/{model}/{tag}/` | checkpoints / history.json / eval_log.csv / uba_comparison.md / target_user_metrics.md |
