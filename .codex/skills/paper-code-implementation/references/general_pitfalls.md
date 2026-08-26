# 论文代码实现常见陷阱（非 API 层面）

复现论文时除 API 语义问题外的其他高频错误。在实现过程中逐项检查。

## 1. 数据格式假设

**典型错误**：论文数据集的原始文件格式是 `user_id item1 item2 item3 ...`（每行一个用户 + 其所有物品），但预处理代码假设为 `user_id item_id`（每行一个交互对），导致每行只解析出第一个物品，丢失 ~97% 的交互数据。不报错，模型能跑，但邻接矩阵几乎为空。

**检测方式**：
- 预处理前 `head -5 train.txt` 确认真实格式
- 预处理后打印 `len(train_pairs)` 与论文 Table 2 对比（Gowalla = 1,027,370）
- 偏差 > 5% 立即报警

## 2. 拿第三方参考实现当 ground truth

**典型错误**：论文有官方代码，但找了第三方实现做对照。第三方可能有自己的 tuning（不同初始化、额外 dropout、不同架构选择）。基于它做修复会把代码往远离论文的方向改。

**检测方式**：
- 区分 A 级（官方代码）、B 级（论文原文）、C 级（第三方）
- 任何修复前先回论文逐句核实
- C 级代码和论文冲突时，以论文为准

## 3. 预处理后不验证交互数

**典型错误**：预处理脚本运行成功，没报错，就开始训练。但实际产出的 train.npz 只有 29K 条交互（应 1M+），没人注意到。模型训练很多轮，loss 正常下降，但 recall 始终是随机水平——因为图里没边，GCN 传播不了信息。

**检测方式**：
- 步骤①完成后必须读 `stats.json` 打印 `num_train`
- 与论文报告值对比（Gowalla: 1,027,370; Yelp2018: 1,561,406; Amazon-Book: 2,984,108）
- 这是不过就白跑的硬门禁

## 4. 正则化公式少一个平方

**典型错误**：论文写 L2 penalty = `||W||^2`，代码写 `W.norm(p=2)`。`.norm(p=2)` 返回开根号后的值，少一步平方导致正则化弱 10-100 倍。

**检测方式**：
- 步骤⑤公式-API 对照验证时手工验算
- 正确写法：`.norm(p=2).pow(2)` 或 `.pow(2).sum()`

## 5. eval_step 返回张量而非标量，导致 Trainer 聚合崩溃

**典型错误**：`eval_step` 返回大批量预测张量（如 shape `(B, N)` 的全量排序评分），Trainer 的 `sum(v)/len(v)` 聚合逻辑对不同 batch 的张量求和时，尾批 batch_size 不同导致 shape 不匹配，crash：`RuntimeError: The size of tensor a (256) must match the size of tensor b (175)`。

**为什么隐蔽**：只要 `batch_size` 整除数据集大小就不会触发（所有 batch 尺寸相同），一换数据集/调 batch_size 才暴露。

**检测方式**：
- 步骤⑤实现 `eval_step` 后，在 `test_snippets.md` ⑤验证脚本中**必须**包含标量返回值断言：`assert isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.numel() == 1)`
- 用 `drop_last=False` 的 DataLoader 故意产生不等长尾批测试
- 如果 eval_step 确实需要返回大批量张量（如推荐系统的全量排序评分），则必须走独立通道（单独的方法如 `predict_full_ranking()`），不要通过 eval_step 返回

## 6. 攻击/投毒类论文的动态数据结构同步

**典型错误**：攻击算法动态注入假用户（ID 从原始 M 开始递增），但预处理阶段创建的 `_train_matrix` 尺寸固定为 `(M_original, N)`。假用户的负采样 collate 访问 `_train_matrix[fake_uid]` 时索引越界：`IndexError: index 943 is out of bounds for dimension 0 with size 943`。

**为什么隐蔽**：正常 train 模式永远只访问 uid ∈ [0, M-1]，不会越界。一开 attack 模式就 crash。

**检测方式**：
1. 步骤①预处理后，**在理解文档中标注**哪些数据结构是静态的（如 N=物品总数）、哪些会在攻击流程中动态扩展（如 M=用户总数、训练交互矩阵）
2. 所有访问"按用户/物品索引的容器"的地方（如 `_train_matrix[uid]`、Embedding 表查找），**必须加边界检查**：`if uid < container.size(0)` 或使用 `.get(uid, default)` 模式
3. 步骤⑤完成后，不仅要测正常训练的 1-batch，还要测**边界 batch**：构造一个最大 uid = M + fake_count 的假 batch，断言不 crash

## 7. 评估字典污染：假用户数据混入正常用户统计

**典型错误**：构建"训练集已交互物品字典"（`train_interacted[uid] = {items}`）时，把假用户的交互记录也一起统计了。后续 HR@K 评估过滤训练集物品时，假用户的 target_item 可能被错误地从正常用户候选中排除。

**检测方式**：
- 所有需要区分"正常用户 vs 假用户/注入用户"的统计字典，**必须**在构建时过滤 uid ≥ original_n_users 的记录
- 步骤④评估验证中，手工构造一个"假用户 target_item 正好是某正常用户 test_item"的边界用例，断言评估结果不受影响
