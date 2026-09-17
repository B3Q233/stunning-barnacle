# pre 跨数据集放大实验（Batch A）设计文档

> 上游：brainstorming（逐节评审，2026-09-16 全部冻结）
> 下游：`docs/superpowers/plans/2026-09-16-pre-scaleup.md`（P0–P10 实施计划）
> 关联：`TPA/pre/docs/SERVER_RUNBOOK.md`（运行手册）、`TPA/pre/docs/ANALYSIS_PROTOCOL.md`（分析协议）
> 状态：设计冻结

## 0. 背景与目标

### 0.1 已确立的结论（ml100k / LightGCN / 单 seed）

目标物品的真实训练交互被逐级删除后，其嵌入相对干净嵌入产生位移；投毒注入可以把位移
部分推回。现有证据（`tmp/uba_ablation/`）：

- L2 口径下，删除比例 >= 0.5 时 `bandwagon` / `UBA` 的恢复为正，`random` 为负；
- 余弦口径下恢复全为负 —— 恢复以模长位移为主，方向未被拉回；
- 假用户嵌入的集中度排序 `bandwagon > UBA > random` 在 6 个比例上稳定；
- UBA 的处理效应缓存存在缺陷，导致 Y 被跨退化程度复用；修好后主结论不变
  （`estimated_value` 随退化单调下降，但恢复幅度几乎不变）。

### 0.2 本次要解决的三个问题

1. **外部效度**：上述现象在更大、更稀疏的数据集上是否成立；
2. **可信性**：现有结论建立在单 seed、且分析层只存在于一次性脚本（`tmp/`）之上，
   无法审计、无法跨数据集复用；
3. **可比性**：不同数据集之间"什么叫同等实验条件"从未定义过。

### 0.3 范围外

- 不实现新的投毒算法（属 Batch B，见 §6.2）；
- 不改动 `attacks/*` 的算法语义（只修 §4 列出的缺陷）；
- 不重训 ml100k 的既有锚点结论（目标集保持连续，见 §1.3）。

### 0.4 Protocol Freeze Principle（总约束）

**Gate 0a 开始后，任何会改变实验定义、数据生成、预算、目标集、分析主口径或 Gate 判据
的修改，都必须回到 spec 重新评审；普通工程修复可以继续，但必须通过对应单元测试并记录
provenance。**

这条用于制度性堵死最危险的情形：实验已经跑了一部分，发现某处不合理，于是顺手改一下
再继续跑。

---

## 1. 数据与预算标定

### 1.1 数据集基础

| | ml100k | gowalla | amazon-book |
|---|---|---|---|
| 用户 / 物品 | 608 / 6,298 | 29,858 / 40,981 | 52,643 / 91,599 |
| train / test 交互 | 38,614 / 9,965 | 810,128 / 217,242 | 2,380,730 / 603,378 |
| 密度 | 0.01008 | 0.000662 | 0.000494 |
| 平均用户度数 | 63.51 | 27.13 | 45.22 |
| 用户度数中位 | 32 | 16 | 26 |
| 物品流行度中位 | 2 | 12 | 15 |
| 最热物品交互数 | 214 | 1,415 | 1,741 |
| train/test 重复对 | 0 / 0 | 0 / 0 | 0 / 0 |

三档数据集的 raw 与 processed（`models/lightgcn/data/processed/<ds>/meta.pkl` + pairs）
均已入库；`pipeline.py::clean_meta_path()` 在模型自身 processed 缺失时回退 lightgcn，
因此 MF 在新数据集上不需要额外预处理。

### 1.2 交互衰减定义

- 删除对象是**目标物品自身**的训练交互，不是数据集比例；
- 比例轴 `r` 取 `{0.1, 0.2, 0.3, 0.5, 0.8, 0.9}`；
- "同等条件"只能是**相对口径**：同一 ratio 的绝对删除量在数据集之间差一个量级
  （r=0.9 时 ml100k 现有目标删 162–193 条，gowalla 最热物品要删 1,274 条）；
- `deleted_interaction_count` / `remaining_interaction_count` 必须逐条落盘并作为协变量
  进入论文表格，不允许只报比例；
- 主口径 K=10（与既有结果连续）；K=20 作为补充稳健性（S4）。

### 1.3 Target qualification

**门槛由两个方向共同决定**：最小比例要删得动、最大比例要剩得下。取「r=0.1 删 >=18 条」
且「r=0.9 剩 >=18 条」，解出 `N_min = 180`。

| 门槛 | ml100k | gowalla | amazon-book |
|---|---|---|---|
| >= 100 | 26 | 728 | 2,806 |
| >= 150 | 7 | 363 | 1,237 |
| **>= 180** | **5** | **237** | **855** |
| >= 200 | 2 | 189 | 674 |

`N_min = 180` 使 ml100k 的候选池恰好等于现有 5 个目标（交互数 180–214），因此
**ml100k 的全部既有结论可以原样延续**。

筛选分两层，**均在观察任何退化结果之前完成**：

1. 资格：交互数 >= 180，且来自 >= 2 个不同用户；
2. 排序：干净模型 native HR@10 降序取前 K=5。

已知代价（必须写进论文）：绝对门槛让目标物品的相对中心度在数据集间不一致（180 是平均
用户度数的 2.8 倍在 ml100k、6.6 倍在 gowalla）。改用相对分位口径会让 ml100k 只剩 2 个
候选、既有结论断裂。折中做法是保留绝对门槛，同时为每个目标记录 `N_i`、
`q_i = N_i / N_items`、交互数分位、`clean HR@10`、`clean NDCG@10`，把差异摆在明面上。

### 1.4 Feasibility pilot（只查可执行性，不筛结果）

`pilot` 只回答"当前代码、数据、模型与攻击配置能否在三个数据集上稳定完成整个实验链路"，
**不根据 HR 或嵌入退化幅度淘汰任何目标**。

检查项：嵌入是否出现 NaN/Inf、`remaining >= 1`、退化模型是否正常收敛、HR/NDCG 是否可算、
L2 是否为有限值、攻击是否可执行、`analyze`/`verify` 是否闭环。

**禁止**：以"HR 明显下降"或"嵌入明显位移"为条件筛选目标。目标选择发生在观察
degradation outcome 之前，这是本设计的核心不变量；HR 下降是被解释的因变量，用它筛目标
属于因变量选择（post-outcome selection）。

### 1.5 Attack budget（双归一化）

基准取 ml100k 现有配置：`A=18`（2.96% 用户）、`P=20`（0.315 倍平均用户度数），派生注入
交互占比 `rho_E = A(P+1)/n_train_pairs = 0.979%`。

主口径 = **双归一化 `(rho_A, rho_P)`**：`A = round(rho_A * n_users)`、
`P = max(1, round(rho_P * mean_user_degree))`。

| 数据集 | 双归一化（主） | 固定 P=20 | 固定 rho_E |
|---|---|---|---|
| ml100k | A=18 P=20 -> 0.979% | A=18 P=20 -> 0.979% | A=18 P=20 -> 0.979% |
| gowalla | A=884 P=9 -> 1.091% | A=884 P=20 -> 2.291% | A=378 P=20 -> 0.980% |
| amazon-book | A=1559 P=14 -> 0.982% | A=1559 P=20 -> 1.375% | A=1111 P=20 -> 0.980% |

理由：① rho_A 是与文献对齐的常用口径；② rho_P 决定假用户画像像不像该数据集的正常用户，
固定 P=20 会让 gowalla 上的假用户达到平均度数的 1.8 倍，把"攻击能力"与"画像异常度"
混在一起；③ 固定 `(rho_A, rho_P)` 会自动让 `rho_E` 稳定在约 1%，总预算可比。

取整抖动需记录在案不隐藏：gowalla 因 P 取整为 9，其 `rho_E` 为 1.091%，比其他两档高约
11%。该偏差进入论文表格的预算列，不做平滑处理。

`rho_E` 作为实际预算记录字段写入 provenance。

### 1.6 Seed

现状为假多 seed：`pipeline.py:46-47` 的 `_seed()` 只取 `seeds[0]`。约定改为：

| 随机源 | 取值 | 作用 |
|---|---|---|
| `selection_seed` | 42（固定） | 目标物品选择，全局唯一 |
| `deletion_seed` | 42（固定，与 model seed 解耦） | 每个物品的删除排列 |
| `model_seed` | {42, 43, 44} | 模型初始化 + 负采样 |
| `attack_seed` | = `model_seed` | 攻击生成与之配对 |

删除集合是实验的**自变量**，不是噪声源。固定后同一 `(dataset, item, ratio)` 下所有条件
看到逐字节相同的退化数据，条件差异只来自模型与攻击，baseline 对比为配对比较。

代价：该设计覆盖模型侧方差，不覆盖删除侧方差；以 S3 补充实验弥补。

### 1.7 Deletion 规则

- 只删 `train_pairs`，test 侧完全不动；
- **去重断言**：`train_pairs` 必须唯一（三数据集实测 dup=0）；现实现
  `p not in deleted_set` 遇重复会一次删光全部重复项，属静默偏差；
- 采样：`stable_seed(deletion_seed, item_id)` 抽**一次**排列，比例 r 删前缀，保证
  `D(0.1)` 是 `D(0.2)` 的子集，依此类推；
- 边界：`n_delete = floor(r * N)`，断言 `remaining >= 1`；
- 落盘：每个 `(item, ratio)` 一份 `deleted_interactions.json`（含明细与计数）。

### 1.8 Shared degradation artifact + fingerprint

退化数据与模型无关（已用 sha256 验证 lightgcn 与 mf 的 ml100k meta 逐字节相同）。因此：

- 改 `degraded/shared/item_<id>/ratio_<pp>/` 单份，模型只影响训练；
- 运行时校验 meta fingerprint，不一致直接报错，不允许静默共用。

---

## 2. 实验矩阵

### 2.1 因子与取值

| 因子 | 取值 | 说明 |
|---|---|---|
| dataset | ml100k / gowalla / amazon-book | §1.1 |
| victim model | lightgcn（主轴）、mf（补充 S1） | pre 已适配，`--mode doctor` 自检 |
| target item | 5 / dataset，**冻结产物** | §2.6 |
| deletion ratio | 6 档；降档时 4 档 `0.1/0.3/0.5/0.9` 或 3 档 `0.1/0.5/0.9` | |
| arm | `degradation-only` + `random` / `bandwagon` / `pgd` / `uba` | 5 条臂 |
| seed | `model_seed` 取 {42,43,44} | `deletion_seed` 固定 |

主轴条件数 = `5 items * R ratios * 5 arms * S seeds`。

### 2.2 成本模型（实测 -> 外推）

本地 RTX 3050 实测（落盘 `metadata.json`，30 epoch/次）：

| 数据集 | batch 数/epoch | 每 epoch | 每 batch | 相对 ml100k |
|---|---|---|---|---|
| ml100k | 151 | 4.2 s | 0.028 s | 1x |
| gowalla | 3,165 | 179.6 s | 0.057 s | 2.0x |
| yelp2018 | 4,834 | 371.6 s | 0.077 s | 2.7x |
| amazon-book | 9,300 | 1138.2 s | 0.122 s | 4.4x |

（yelp2018 仅作成本标定，用于检验"epoch 时间随数据规模超线性外推"是否成立，
**不进入实验矩阵**，实验数据集只有 §1.1 的三档。）

代价约等于 `batch 数 * 图规模`，两者随数据集增长，因此 epoch 时间超线性。amazon-book 单次
训练（30 epoch）= 9.5 h，是 ml100k 的 271 倍。

4090 按 **约 5 倍** 估算（瓶颈在图传播，属 GPU-bound；per-batch 中 CPU 加载占 10–16%，
理论上限约 7 倍）。该系数由 Gate 1 实测回填；档位之间的相对比较不依赖它。

### 2.3 规模档位与分阶段（预注册决策规则）

| 档 | ml100k | gowalla | amazon-book | runs | GPU-h | 4 卡墙钟 |
|---|---|---|---|---|---|---|
| **A** | 5x6x5x3 | 5x6x5x3 | 5x6x5x3 | 1,350 | 993 | 约 10.3 天 |
| **B** | 5x6x5x3 | 5x6x5x3 | 5x4x5x3 | 1,200 | 706 | 约 7.4 天 |
| **C** | 5x6x5x3 | 5x6x5x3 | 5x3x5x2 | 1,050 | 423 | 约 4.4 天 |
| **D** | 5x6x5x3 | 5x6x5x3 | 暂不跑 | 900 | 138 | 约 1.4 天 |

执行路径固定为 **D -> 按 §5.7 的 Rule A / Rule B 决定 B 或 C**，默认进入 **B**。理由是
"少 ratio 优于少 seed"：三 seed 本身是主结论的一部分，不应为省 GPU 小时削掉。

决策规则在任何结果产生之前固定，禁止"先看 gowalla 结果好不好再决定 amazon 跑多少"。

### 2.4 补充实验

| 编号 | 内容 | 构成（items x ratios x arms x seeds） | 规模 | 回答 |
|---|---|---|---|---|
| S1 | backbone 泛化：mf x {ml100k, gowalla} | ml100k `5x6x5x3`；gowalla `5x3(0.1/0.5/0.9)x5x3` | 450 + 225 | 换掉图结构后排序是否保持 |
| S2 | 预算敏感性：gowalla，`rho_A` 取 {1%, 3%} | `5 x 1(ratio=0.5) x 3 臂(random/bandwagon/uba) x 3 x 2 档预算` | 90 | 恢复是否依赖预算大小 |
| S3 | deletion-seed 方差：gowalla，3 个 deletion seed | `5 x 3(0.1/0.5/0.9) x 2 臂(bandwagon/uba) x 3 deletion seed` | 90 | 结论不来自某一个删除排列 |
| S4 | K 稳健性：k=20 | 复用主轴已有产物，只重算指标 | 0 | 绝对水平是否受 K 影响 |
| S5 | 目标 5->10（gowalla） | 追加 `5 items x 6x5x3` | +450 | robustness reserve，出现"某个目标主导"质疑时启动 |
| S6 | TPA / AdvInject 修复后纳入 | 修复结果确定后另定 | 视修复结果 | 已有攻击是否符合同一规律 |

S4 零成本必做；S5/S6 后置。**S6 不进主轴**：两个攻击在 pre 里从未完整跑通，不应让未
验证的臂决定主结论。

### 2.5 分片

- 分片单元 `(dataset, model, seed, item)`，主轴 45 个 shard；
- `run_tag`：`batchA-<dataset>-<model>-s<seed>-item<id>`，ratio 粒度追加 `-r<ratio>`；
- 预估 wall-clock 超过 24 h 时自动降级到 ratio 粒度（amazon-book 会命中）；
- 调度用 LPT 贪心（按预估成本降序），避免 ml100k 的 shard 先占满卡；
- `reuse` 语义使重复执行同一命令等价于续跑；shard 之间文件隔离，单点崩溃不污染其他
  shard；
- clean 模型只依赖 `(dataset, model, seed)`，放 batch 级共享目录，45 个 shard 复用同一份
  （否则 clean 成本从 0.7% 涨到 6.4%）。

### 2.6 目标集冻结（工程保证）

- 产物 `TPA/pre/targets/<dataset>.json`，由 `--mode targets` 生成并 **commit 进 git**；
- 内容：`dataset / strategy / selection_seed / exposure_source_model / sha256(meta.pkl) /
  items[]`；
- 每个 item 记录 `N_i`、`q_i`、交互数分位、`clean HR@10`、`clean NDCG@10`；
- 后续所有阶段只读该文件，缺失即报错，**不允许现场重选**。

这样"目标集在看结果之前定死"可经 git 历史审计；`sha256(meta.pkl)` 同时承担 §1.8 的
指纹校验。

### 2.7 provenance 链

每个 manifest 记录：

```
experimental_protocol_version, dataset, victim_model, target_file_sha256,
deletion_seed, model_seed, attack_seed, ratio, rho_A, rho_P, rho_E,
attack, K, code_commit
```

使论文数字可反向追踪：`Figure -> analysis output -> raw CSV -> shard -> config ->
target.json -> git commit`。

---

## 3. 分析层（`pre/analysis/` 入仓）

### 3.1 记号与参考帧

`z0` = 干净模型中目标物品的嵌入；`z_deg` / `z_att` 为退化后 / 攻击后。每个
`(dataset, model, seed)` 有**自己的参考帧**。

**硬规则**：一切嵌入层面的比较只在同一 `(dataset, model, seed)` 内进行；跨 seed 只在
指标层聚合（均值 +- 标准差），绝不对嵌入做平均。

### 3.2 对齐（gauge）

实测 `mean_row_cos_raw` 仅 **0.74–0.79**，`aligned` **0.988**，残差 0.14 —— 两次独立
训练得到的嵌入空间之间相差一个近似全局旋转；`raw` 口径下的位移绝大部分是这个旋转。

因此：

1. **主口径固定为 Procrustes 对齐**，`raw` 仅作诊断；
2. **拟合掩码排除目标物品**，避免对齐变换吸收目标自身位移（leakage into alignment）；
3. **同一个 `R` 同时作用于物品行与用户行**（用户与物品共享同一嵌入空间），`F`、
   audience 的对齐必须复用 `R`；
4. **对齐质量是门槛**：`mean_row_cos_aligned < 0.95` 时标记 `unresolved_frame`。

### 3.3 Placebo 对照与超噪声位移

删除只影响目标物品的边，其余物品的位移即该条件的噪声底：

```
d_ctrl(c) = median over j in C of d_j     C = 未受影响的物品集合
E(c)      = d_tgt(c) - d_ctrl(c)          超噪声位移
RR_pc     = (E_deg - E_att) / E_deg
```

该定义能自检坐标系是否合格：`raw` 下 `d_ctrl` 约 2.15，与目标位移同量级，于是 `E` 约为
0，指标自己判定该坐标系无分辨力；Procrustes 下 `d_ctrl` 约 0.19、`d_deg` 约 1.02，
分离度 5 倍。

已用 ml100k 现有产物验证（目标 402，Procrustes 拟合排除目标）：

| 条件 | d_tgt | d_ctrl (median) | z | 目标分位 |
|---|---|---|---|---|
| deg-only r=0.1 | 0.391 | 0.187 | +5.25 | 99.8% |
| deg-only r=0.5 | 1.016 | 0.192 | +22.0 | 100% |
| deg-only r=0.9 | 2.087 | 0.188 | +50.8 | 100% |
| random r=0.5 | 1.093 | 0.379 | +11.9 | 100% |
| bandwagon r=0.5 | 0.731 | 0.380 | +6.67 | 99.9% |
| uba r=0.5 | 0.817 | 0.377 | +8.42 | 99.9% |

**约束**：安慰剂对照只用于报告与归一化，**不允许用于挑选目标物品**（§1.4 不变量优先）。

### 3.4 恢复量三件套

```
Delta_abs = d_deg - d_att                  绝对恢复量（仅同数据集内可比）
RR        = Delta_abs / d_deg              归一化恢复率（原始主指标）
RR_pc     = (E_deg - E_att) / E_deg        安慰剂控制恢复率（机制主指标）
d_rel     = ||z - z0|| / ||z0||            相对位移（跨数据集可比）
```

表格固定同时给出 `Delta_abs`、`RR`、`E_deg`、`E_att`、`RR_pc`。机制解释优先 `RR_pc`；
原始实验结果保留 `RR`；绝对效果看 `Delta_abs`；判断 placebo correction 是否可靠看
`E_deg`。

**`pc_valid` flag**：`RR_pc` 在 `E_deg` 趋近 0 时会爆炸。因此设
`pc_valid = E_deg > threshold`，阈值**不拍脑袋取整数**，而由 placebo 分布与数值精度
导出（规则写进代码与 `ANALYSIS_PROTOCOL.md`）。`pc_valid = False` 的条件**照常报告**
`RR` / `Delta_abs`，不静默丢弃。

### 3.5 径向/切向分解

现有 `norm_recovery = (||z_att|| - ||z_deg||) / (||z0|| - ||z_deg||)` 只分解模长，方向变化
大时引入交叉项。改用严格恒等式：令 `u0 = z0 / ||z0||`、`r(z) = <z, u0>`、
`t(z) = z - r(z) * u0`，则

```
||z - z0||^2 = (r(z) - ||z0||)^2  +  ||t(z)||^2
               径向偏差平方          切向偏差平方
```

对 `z_deg`、`z_att` 做**状态分解**，对攻击位移 `z_att - z_deg` 做**作用分解**。已有的
"L2 正、余弦负"结论用该恒等式重述（径向改善、切向恶化）。`norm_recovery` 保留为 legacy
字段。

### 3.6 力结构

`F = U_att[fake_ids]`（攻击后模型的假用户行，经同一 `R` 对齐）：

| 量 | 定义 |
|---|---|
| `concentration` | `||mean(F)|| / mean(||f||)`，1 表示完全同向 |
| `pairwise_cos` | 两两 `cos(f_i, f_j)` 的均值 |
| `cos_F_z0` | `cos(mean(F), z0)` |
| `cos_F_aud` | `cos(mean(F), mean(U[aud]))` |

**audience 两套定义**：主口径为**固定受众** `A_i = {u : (u,i) 属于干净训练集}`，使比例轴
上的曲线变化只来自 force 变化；辅口径为**剩余受众** `A_i(r) = {u : (u,i) 属于 D_r 训练集}`，
仅作 supplementary diagnostic，不与固定受众混在同一主图中。

需在文档中写明：此处的"力"是训练后假用户在实际模型中的位置，**不是攻击者的意图向量**，
不可解读为因果量。

### 3.7 暴露侧验证

嵌入恢复不等于曝光恢复。分析阶段已重建条件模型，顺带计算 per-condition
`target HR@K / NDCG@K`（K=10 与 20），同时承载 S4。

口径约束：曝光评估必须在**模型自身坐标系**里做，**不得使用对齐后的嵌入**（`align.py`
已声明对齐只服务距离分析）。

### 3.8 聚合与统计

- **两级聚合**：先在 `(dataset, model, attack, ratio, seed)` 内对物品求均值，再对 seed
  求均值和标准差；
- **必须同时输出 per-item 表**，防止某个物品主导均值；
- **配对比较**：同一 `(seed, item, ratio)` 内两臂配对；报告符号一致率与配对差值；
  不引入新依赖（符号检验用 stdlib 实现）；
- **覆盖率**：`analyze` 不得静默跳过缺失条件。四态状态机：
  `resolved / unresolved_frame / missing / invalid_numeric`，输出 coverage 表，覆盖率
  写入 `summary.json` 与每张图表注。

### 3.9 跨数据集可比性（schema 级约束）

只有无量纲量可跨数据集比较。跨数据集的表**禁止出现** `Delta_abs`、`d_deg`、`d_ctrl`；
只允许 `RR`、`RR_pc`、`d_rel`、`cos`、分解占比。该约束**用单元测试强制**（构造含绝对
距离的跨数据集表应报错），不只写文档。

### 3.10 模块划分与数据流

```
pre/analysis/
  metrics.py         d_l2 / d_cos / d_rel / RR / Delta_abs（纯函数，无 IO）
  decomposition.py   径向-切向恒等式、norm_recovery（legacy）
  align.py           Procrustes（新增"排除目标物品"选项）
  placebo.py         对照集合构造、d_ctrl、超噪声位移 E、pc_valid
  concentration.py   F 的集中度、pairwise cos、cos(F,z0)
  audience.py        固定受众 / 剩余受众 + cos(F,audience)
  exposure.py        target HR@K/NDCG@K（扩展到 K=10/20、per-condition）
  aggregate.py       两级聚合 + coverage 表 + legacy/corrected 双输出
  plots.py           固定图式：绝对偏差 / 归一化恢复 / 力结构 / 分解占比
```

数据流严格分层：

```
outputs/<run_tag>/
  raw/        训练与攻击产物（每 shard 一份）
  analysis/   legacy/ 与 corrected/ 两套派生结果
  tables/     跨条件 / 跨 seed / 跨数据集汇总
  figures/    *.pdf + *.svg
  manifest.json
```

**关键性质**：`analysis/` 到 `tables/` 到 `figures/` 全程不需要重新训练。

---

## 4. 硬伤修复

### 4.0 分类

| 类别 | 定义 |
|---|---|
| **A 类 结果口径 bug** | 跑出来的数与你声称的实验定义不一致（静默降级、错误预算、被污染的参考系、被隐藏的覆盖率） |
| **B 类 工程可靠性 bug** | 不改变定义，但在 45 shard x 3 卡 x 断点续跑下会让实验无法可靠完成 |

| # | 项目 | 类别 | 阻塞 ml100k 复现 |
|---|---|---|---|
| 1 | 多 seed 静默降级 | A | 是 |
| 2 | UBA 处理效应缓存（key + 行身份） | A + B | 是 |
| 3 | TPA `path_builder` 数据源不一致 | B | 否（仅 S6） |
| 4 | advinject 未接入批量 | B | 否（仅 S6） |
| 5 | Procrustes 含目标物品 | A | 是 |
| 6 | `analyze` 静默跳过缺失条件 | A | 是 |
| 7 | `norm_recovery` 只分解模长 | A | 是 |
| 8 | `degraded` 按模型分层 | B | 是 |
| 9 | 重复交互未断言 | A（潜在） | 是 |
| 10 | `uba --mode all` 无条件跑 estimate | B | 否 |
| 11 | manifest 缺 provenance 链 | B | 是 |
| 12 | `align.py` 文档论断被推翻 | A（文档） | 是 |

### 4.1 修复 1：多 seed 静默降级（A 类）

- **现象**：`seeds: [42,43,44]` 跑出来逐字节相同、输出目录只有一个，不报错也不警告。
- **根因**：`pre/runners/pipeline.py:46-47` 的 `_seed()` 返回 `seeds[0]`；六个 phase 全
  调用它，输出路径无 seed 维度。
- **修改**：拆成 `model_seeds()` / `deletion_seed()`；phase 接受显式 `seed`；`run.py`
  新增 `--seed` 与 `--seeds`；路径插入 `seed_<s>` 段；`degraded/` 不含 seed。
- **单元测试**：`--seeds 42,43` 产生两个独立目录且 `metadata.json["seed"]` 正确；同一
  `(item, ratio)` 两次运行读到同一份 `deleted_interactions.json`；模型权重哈希不同。
- **验收**：ml100k 上 `--seeds 42,43,44 --limit-items 1 --ratios 0.5` 通过；三套退化 meta
  的 sha256 相同、三套模型权重 sha256 不同；单 seed 行为与 run-k5v2 对齐不变。

### 4.2 修复 2：UBA 处理效应缓存（A + B 类）

- **现象**：同一目标物品在 6 个比例下 `estimated_value` 完全相同（402=28121、16=43534、
  339=43411、139=44435、28=38978）；缓存 Y 追踪到 `run-k5@0.1`；跨条件 `target_users`
  位置重合度仅 **7.8%**。
- **根因**：① `attacks/uba/generate.py::effect_cache_path` 的缓存名只含
  `item/max_per_user/alpha/beta`，**不含数据指纹**；② `generate.py:498` 只校验行数
  `effect.shape != (len(target_users), max_per_user+1)`，而 `uplift.allocate` 按位置索引
  `values[r, t]`，于是行数相同但用户集合不同时，Y 的行被贴到**错误的用户**上。
- **修改**：缓存 key 加 `sha256(num_users, num_items, sorted(train_pairs))[:16]`；缓存内写
  有序 `target_users` + `meta_fingerprint` + `git_commit`；命中后逐元素校验
  `target_users`，不一致则重算；`allocate` 改为显式 `{user_id: row_index}` 映射。
- **单元测试**：同行数不同用户集合必须触发重算；乱序 `target_users` 下按 user_id 取到的
  Y 行与原顺序一致。
- **验收**：5 目标 x 3 比例的 `estimated_value` 不再恒定；`target_users` 一致率 100%；
  重复命中缓存结果逐字节一致。

### 4.3 修复 3：TPA `path_builder` 数据源不一致（B 类）

- **现象**：共现路径基于干净数据构建，中毒 meta 是退化数据，两阶段看到不同的图且不报错。
- **根因**：`path_builder.py:350` 用 `load_meta(raw_meta_path(config))`，而 `raw_meta_path`
  （`generate.py:32`）把路径写死成 processed 目录；对比 `generate.py:183` 的
  `main(config, raw_meta=None)` 支持重定向，同一模块内两阶段能力不一致。
- **修改**：`path_builder.main(config, raw_meta=None)`；`run_attack.py` 调用 `pre_stage` 时
  同样传退化 meta；两阶段共用同一处数据源解析。
- **验收**：TPA 在 pre 里跑通一次闭环（1 item x 1 ratio），产物记录的 meta 与 degraded
  meta 逐字节一致；不传 `raw_meta` 时回退行为与旧版一致。

### 4.4 修复 4：advinject 未接入批量（B 类）

- **现象**：`attacks/batch/registry.py::_register_builtin` 注册
  `("bandwagon","random","pgd","tpa","uba")`，**无 advinject**；而 pre 的 `ATTACK_SPECS`
  有 advinject。同一攻击在两个调度层可用性不一致。
- **根因**：批量注册表漏登记；advinject 入口名是 `generate` 而非 `main`，未做适配。
- **修改**：注册表支持 `entry` / `meta_kwarg` 差异；按 AGENTS §6.5 四项自查逐条核对。
- **验收**：`registered_names()` 含 advinject；ml100k 上一次 `classify -> generate -> fit`
  闭环且指标可被 aggregate 解析。

### 4.5 其余八项（同格式）

| # | 项目 | 类别 | 现象 -> 根因 | 修改 | 单元测试 / 验收 |
|---|---|---|---|---|---|
| 5 | Procrustes 含目标物品 | A | 对齐变换部分吸收目标自身位移 | 拟合掩码排除目标物品；`R` 同时作用于用户行 | 构造已知旋转 + 目标偏移，断言 `R` 与目标偏移无关 |
| 6 | `analyze` 静默跳过缺失条件 | A | 跑到一半与跑完看起来一样（`continue`） | 四态状态机 + coverage 表，覆盖率入 summary 与图表注 | 删掉一个条件目录，coverage=29/30 且状态 `missing` |
| 7 | `norm_recovery` 只分解模长 | A | 方向变化大时引入交叉项 | 改用径向/切向恒等式（§3.5），旧字段转 legacy | 断言 `d^2 = radial^2 + tangent^2`（1e-10） |
| 8 | `degraded` 按模型分层 | B | 内容与模型无关却存三份，可能漂移 | 改 `degraded/shared/` 单份 + 运行时指纹校验 | 两份模型 meta 指纹不一致时报错 |
| 9 | 重复交互未断言 | A（潜在） | `p not in deleted_set` 会一次删光全部重复项 | 入口断言 `train_pairs` 唯一 | 注入重复对，断言抛错 |
| 10 | `uba --mode all` 无条件跑 estimate | B | `estimate.main` 不检查 `treatment.method` | estimate 入口检查 method，非 surrogate 跳过并打印 | `method=path` 时 `--mode all` 不训练代理模型 |
| 11 | manifest 缺 provenance 链 | B | 论文数字无法反查 | 按 §2.7 字段表写入 + `experimental_protocol_version` | 字段齐全性断言 |
| 12 | `align.py` 文档论断被推翻 | A（文档） | 文档称"同初始化重训余弦 0.956–0.975 可比"，实测 **0.74–0.79**；照文档用 `raw` 会得出"所有攻击都是负恢复" | 改写 docstring 并把实测值写进去；代码默认 `procrustes` | 文档与 §3.2 一致 |

### 4.6 验收判据（L1–L4）与 legacy 并行输出

ml100k 复现**不以"修复前后数字相同"为标准**（修复本身必然改变数字）：

| 层级 | 内容 | 性质 |
|---|---|---|
| **L1** | clean / degraded / targets 等不受该修复影响的产物保持一致 | 硬约束 |
| **L2** | 数据指纹、seed、删除集合、provenance、coverage 100% 正确 | 硬约束 |
| **L3** | 新旧口径并行计算，**所有数值差异能够定位到具体修复** | 硬约束 |
| **L4** | 攻击方向、排序、placebo 对照作为**独立科学结果验证** | 不作代码复现硬约束 |

L3 要求每个发生变化的指标都能追溯到具体口径变化（例：`d_deg` 变化对应 Procrustes 排除
目标物品；`norm_recovery` 对应径向/切向分解），而不是"偏离不能太大"。

**legacy / corrected 双输出**：

```
analysis/
  legacy/     results.csv + summary.json    （v1 口径，审计/迁移用）
  corrected/  results.csv + summary.json    （唯一主口径）
manifest: analysis_protocol: {primary: corrected, legacy: legacy}
```

legacy 仅作审计与结果迁移，**不得进入论文主结果**；所有论文图表、统计检验与结论只允许
读取 corrected。

---

## 5. 服务器执行

### 5.1 交付物

| 交付物 | 路径 | 入库 |
|---|---|---|
| 运行手册 | `TPA/pre/docs/SERVER_RUNBOOK.md` | 是 |
| 分析协议 | `TPA/pre/docs/ANALYSIS_PROTOCOL.md` | 是 |
| 分片计划器 | `TPA/pre/schedule/plan.py` | 是 |
| 调度器 | `TPA/pre/schedule/run.py` | 是 |
| 完整性校验 | `TPA/pre/schedule/verify.py` | 是 |
| 分片清单 | `outputs/<batch>/plan.json` | 是 |
| 论文资产 | `outputs/<tag>/{analysis,tables,figures}` | 是（§5.9） |
| 训练/攻击产物 | `outputs/<tag>/raw/`、`attacks/*/data/` | 否，可重生成 |

命名约定：`<batch>` 指一次批量实验（如 `batchA`），`<tag>` 指其下的 shard 级目录；
`plan.json` / `coverage.json` / `coverage.md` 位于 batch 级，`analysis` / `tables` /
`figures` 位于 shard 级，跨 shard 的汇总目录亦在 batch 级。

`pre/runners/` 负责"怎么跑一个实验"，`pre/schedule/` 负责"什么时候、在哪张 GPU、跑哪些
实验"；GPU 调度、checkpoint、shard 状态不得塞回 `runners`。

### 5.2 clone 后检查（preflight）

`python pre/run.py --mode doctor` 扩展为完整 preflight，产出 `preflight.json`：

1. 环境：Python 3.12、torch 2.5.1+cu121、`torch.cuda.is_available()`、可见 GPU 列表
2. 依赖：`requirements.txt` 与已安装版本一致
3. 数据：三个数据集 `meta.pkl` 存在且 sha256 与 `pre/targets/<dataset>.json` 一致
4. 目标集冻结文件存在、可解析、含 `N_i / q_i / clean HR@10 / clean NDCG@10`
5. 攻击 `ATTACK_SPECS` 全部可解析到 callable
6. 磁盘空间、输出目录可写

**任何一项失败即拒绝启动大矩阵**。

### 5.3 GPU 探测与动态绑定

`GPU 1 被占 6.4 GB` 不写死进 runbook：

```
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  可用卡 = 显存占用 < 2 GB 且 util < 20%
  保留 1 张 debug 卡（默认取可用卡中显存占用最低者，--reserve-debug 可覆盖）
  主实验卡 = 剩余可用卡，默认 3 张
  写入 plan.json 的 gpu_assignment，逐个用 CUDA_VISIBLE_DEVICES 绑定子进程
```

探测与分配结果均落盘。可用卡少于 2 张时拒绝启动并打印当前 `nvidia-smi`。

### 5.4 日志与运行方式

- 调度器前台运行 + 终端复用（`tmux` / `screen` / `nohup`），命令写进 runbook；
- 每 shard 结束打印 `[shard] tag=... status=... seconds=...`；
- 定期打印进度：完成 X/Y shard、已耗 GPU-小时、预计剩余时间。

### 5.5 断点续跑

1. **pipeline 层**：`reuse` 语义（产物齐则跳过）；
2. **调度器层**：完成的 shard 写 `<run_tag>.done`，重启时跳过；
3. **日志层**：每 shard 一份 `logs/<run_tag>.log`。

两个幂等隐患一并处理：`run.py` 每次重写 `manifest.json`，改为保留最后一次并追加
`manifest_history.jsonl`；`analyze` 覆盖 CSV 的问题由 §3.10 的 legacy/corrected 双份承担，
并要求重复执行结果一致。

### 5.6 输出完整性校验

`pre/schedule/verify.py` 逐 shard 检查：期望条件数（`5 items x R ratios x 5 arms`）、关键
文件存在性、四态状态机、数值有限性（NaN/Inf）、meta fingerprint。产出 `coverage.json` +
`coverage.md`。

**退出码语义**：`missing` 或 `invalid_numeric` 报非零；`unresolved_frame` 返回零但计入
覆盖率并报警（**shard 级语义**）。

**Gate 级语义**：Gate 2 / Gate 3 要求 `unresolved_frame = 0`，否则 gate fail，不允许它
悄悄混进最终论文资产。

### 5.7 Gate 与 Rule（预注册判据）

| 门 | 内容 | 判据 |
|---|---|---|
| **Gate 0a** | 本地 ml100k 快速验收：`1 item x 3 ratio x 5 arms x 2 seed` | L1 / L2 / L3；失败直接回 P1–P3，不进入 0b |
| **Gate 0b** | 本地 ml100k 全量主轴：`5 x 6 x 5 x 3` | L1 / L2 / L3 + coverage 100% |
| **Gate 1** | 服务器 preflight + dry-run：三数据集各 `1 item x 1 ratio x {random, UBA, TPA}` | preflight 全绿、verify 全绿、实测 epoch 回填成本模型 |
| **Gate 2** | ml100k + gowalla 主轴 + S1–S4 | Rule A + Rule B1–B4 |
| **Gate 3** | amazon-book 档位 | 默认 B（4 ratio x 3 seed）；明确 feasibility 问题才进 C |

Gate 1 选 `{random, UBA, TPA}` 是因为这三臂覆盖三类执行路径：`random` 是基础 pipeline
sanity；`UBA` 验证 effect-cache 与 `target_users` provenance 修复；`TPA` 验证 `raw_meta`
重定向与 `path_builder` 数据源一致性。advinject 另做轻量 import/registry smoke test，
真正闭环留到后续。

**Rule 定义**

- **Rule A（跨数据集方向一致）**：ml100k 与 gowalla 在 `ratio >= 0.5` 上，各预注册
  primary contrast `Delta_RR_pc(attack, random)` 的 `RR_pc` **同号**。攻击臂之间的相对
  排序**记录为结果，不作为 Gate 通过条件**（与 L4 一致）。
- **Rule B1（跨 seed 稳定）**：`bandwagon` / `uba` 的 `RR_pc` 在 3 个 model seed 中
  至少 2/3 同号。
- **Rule B2（退化可测）**：三个比例下 `E_deg > 0` 且目标位移 z-score 至少 5。
- **Rule B3（攻击非噪声）**：`Delta_i = RR_pc(attack)_i - RR_pc(random)_i`，**配对**要求同
  一 `item x ratio x seed`；`sign consistency = max(正号数, 负号数) / N >= 2/3`。
- **Rule B4（无工程 artifact）**：coverage 100%、`invalid_numeric = 0`、
  `unresolved_frame = 0`。

### 5.8 本地 / 服务器一致性

- 服务器建**独立 venv**（不复用现有 conda 环境），按 `requirements.txt` 安装；CUDA 12.8
  驱动向下兼容 `cu121` 轮子，不需要换 torch；
- 每个 manifest 记录 `git_commit` + `experimental_protocol_version`；
- 分析层（对齐、分解、placebo）跑在 CPU，显存全部留给训练。

### 5.9 产物回传

`.gitignore` 白名单（**已获授权**），保持"raw / checkpoint / log 不入 Git，可重生成的论文
最终资产允许入 Git"：

```
!outputs/*/
!outputs/*/analysis/
!outputs/*/analysis/**
!outputs/*/tables/
!outputs/*/tables/**
!outputs/*/figures/
!outputs/*/figures/**
!outputs/*/plan.json
!outputs/*/coverage.json
!outputs/*/coverage.md
```

具体规则需按当前 `.gitignore` 的层级调整，避免父目录被 ignore 导致 `!` 失效。figures 输出
**PDF**（论文插图 / LaTeX / 最终提交）与 **SVG**（网页 / 人工检查 / 后续编辑）；PNG 只作
临时 debug，不作为正式论文资产。

---

## 6. 最终验收与落文件

### 6.1 验收时间线

```
Gate 0a -> Gate 0b -> Gate 1 -> Gate 2 -> Gate 3 -> 论文资产
```

各 Gate 的判据见 §5.7。

### 6.2 Batch B 的启动门槛

| 动作 | 允许时点 | 理由 |
|---|---|---|
| 移植代码（模块、adapter、单测、sanity harness） | **Gate 1 通过后** | 接口已在 §4 冻结、分析层已定型；服务器矩阵在后台跑 |
| 入矩阵跑实验 | **Gate 3 通过后** | 否则新攻击要同时适配两套口径 |
| 进入论文主轴表格 | 不允许 | Batch B 独立成批，以独立章节/表呈现 |

**红线**：Batch B 以**独立目录 / 分支式新增**方式进行，**不得修改已冻结的 Batch A 实验
协议、数据口径、主轴配置与分析接口**。若 Batch B 确实需要改动这些，必须回到 spec 评审，
而不是边跑 Batch A 边修改协议。

每个新攻击入矩阵前必须通过**官方设定 sanity check**（例如 ml100k 上验证 AUSH+ 相对
Random 的增益方向与量级对得上），未通过只能进附录。

### 6.3 落文件

| 文件 | 内容 | 何时提交 |
|---|---|---|
| `docs/superpowers/specs/2026-09-16-pre-scaleup-design.md` | 第 1–6 节（本文） | 现在 |
| `docs/superpowers/plans/2026-09-16-pre-scaleup.md` | P0–P10 实施计划 | spec 评审通过后 |
| `TPA/pre/docs/SERVER_RUNBOOK.md` | 操作手册（§5 + Gate 判据 + 两层 unresolved_frame 语义） | P0 |
| `TPA/pre/docs/ANALYSIS_PROTOCOL.md` | §3 的数学口径 + `analysis_protocol` 版本号 | P3 |
| `docs/superpowers/specs/2026-09-16-batchB-attack-porting-design.md` | Batch B（Leg-UP / TrialAttack / SUI-Attack） | Gate 1 通过后 |

### 6.4 实施计划分段（P0–P10）

| 阶段 | 内容 | 退出判据 |
|---|---|---|
| **P0** | spec 落盘、runbook 骨架、`experimental_protocol_version` 定义 | 文档入库 |
| **P1** | A 类修复 1 -> 2 -> 5 -> 6 -> 7 -> 9 -> 12 + 单元测试 | 单测全绿 |
| **P2** | B 类修复 3 / 4 / 8 / 10 / 11 + 单元测试 | 单测全绿 |
| **P3** | 分析层入仓（§3.10 模块）+ legacy/corrected 双输出 | §3 单测全绿、legacy 能复现旧数 |
| **P4** | preflight + `pre/schedule/{plan,run,verify}` + 单测 | 1 卡本地跑通一个 2-shard 计划 |
| **P5** | Gate 0a -> Gate 0b | L1–L3 |
| **P6** | 服务器 preflight + Gate 1 dry-run | 全绿 + 成本回填 |
| **P7** | Gate 2：ml100k + gowalla 主轴 + S1–S4 | Rule A / B1–B4 |
| **P8** | Gate 3：amazon-book 按预定规则 | 档位确定并跑完 |
| **P9** | 论文资产：tables / figures（PDF+SVG） | 只读 corrected |
| **P10** | Batch B：spec -> 移植（Gate 1 后）-> sanity check -> 入矩阵（Gate 3 后） | 独立章节产出；**独立目录/分支式新增，不改 Batch A 冻结协议** |

P1/P2 分开的理由：先消灭会改变实验定义的 bug，再处理工程可靠性问题。

每阶段结束的通用要求（AGENTS.md）：`python -m unittest discover -s tests -t . -v` 全绿、
文档同步更新、Conventional Commits 中文提交、`git status` 自查。

### 6.5 红线

- Gate 0 未通过，不允许碰服务器大矩阵；
- preflight 有红项，不允许启动；
- verify 出现 `missing` / `invalid_numeric`，不允许进入下一 Gate；
- Gate 级 `unresolved_frame > 0`，Gate 失败，不允许进入论文资产；
- Rule A / B 未通过，不允许跑 amazon 的 B 档（否则浪费约 10 天）；
- Batch B 未过官方 sanity check，不允许入表。

### 6.6 Protocol Freeze Principle

见 §0.4。

---

## 附录 A：本文引用的实测证据清单

| 证据 | 出处 | 用途 |
|---|---|---|
| ml100k 5 目标交互数 180–214 | `outputs/run-k5v2/targets.json` | §1.3 门槛定标 |
| 三数据集规模/密度/度数分布 | `models/lightgcn/data/processed/*/meta.pkl` | §1.1、§1.3 |
| 单 epoch 成本 4.2 / 179.6 / 371.6 / 1138.2 s | `tmp/pre-timing/outputs/timing-*/original/lightgcn/clean/metadata.json` | §2.2 |
| 跨条件 `target_users` 位置重合 7.8% | `tmp/uba_ablation/contamination_run_k5v2.py` | §4.2 |
| `mean_row_cos_raw` 0.74–0.79 对比 `aligned` 0.988 | `outputs/run-k5v2/results/embedding_distances.csv` | §3.2、§4.5(#12) |
| placebo 对照（d_ctrl 约 0.19，目标 z 至少 +5.25） | 设计期用 run-k5v2 产物复算 | §3.3 |
| 力集中度排序 bandwagon > UBA > random | `tmp/uba_ablation/force_compare.json` | §0.1、§3.6 |

**证据文件的可见性（重要）**：上表出处分三类，服务器端 clone 后**只有第一类存在**。

| 出处类别 | 例子 | clone 后是否可得 | 复核方式 |
|---|---|---|---|
| 随代码入库 | `models/lightgcn/data/processed/*/meta.pkl` | 是 | 直接读 |
| 本地实验产物 | `outputs/run-k5v2/…` | 否（`outputs/` 被 .gitignore 覆盖） | 重跑对应 tag，或从本地手动拷贝 |
| 本地中间产物 | `tmp/pre-timing/…`、`tmp/uba_ablation/…` | 否（`tmp/` 被 .gitignore 覆盖，且禁止 `git add -f`） | 从本地手动拷贝，或按 §4 修复后重测 |
 
这些数字的**结论已全部写入正文**（§1.1 规模表、§2.2 成本表、§3.2 对齐实测、§3.3 placebo
表、§4.2 重合度），因此不依赖原始文件即可执行本设计；原始文件仅用于事后逐项复核。
P1/P2 完成后，§2.2 与 §4.2 的证据将由新流水线重新产出并按 §5.9 的白名单入库
（对应计划 Task 24 的 `schedule/verify.py` 产出 batch 级 `coverage.*`，以及 Task 31 的白名单回传）。
