# AdvInject 实现对照与复现边界

## 实现入口

- `classify.py`：统计 clean train interaction 的 item 频次，生成 `popular / ordinary / cold` 分类缓存。
- `data.py`、`generate.py`：读取连续 ID 的 `meta.pkl`，初始化 fake users，在 surrogate 上执行有限步可微优化，并保存投毒 `meta.pkl`。
- `fit.py`、`evaluate.py`：分别在 clean 与 poisoned train pairs 上重新训练 victim，计算 `TargetAvgRank`、`TargetHR@K`、`TargetNDCG@K`。
- `run.py`：按 `classify → data → model` 编排；单独执行 `model` 时读取 `data/poisoned/.../latest.json`，不会重新生成投毒数据。

## 上游对应关系

实现依据 `graytowne/revisit_adv_rec`（已核对 commit `0e50743712810a515e35642d65399541e2553445`）中的 `BlackBoxAdvTrainer`、`ItemAETrainer`、`WMFTrainer`、`BaseTrainer`、`losses.py`、`generate_attack.py` 和 `evaluate_attack.py`。

- 上游 fake-user 初始化与目标频次选择对应 `data.py`。
- 上游 surrogate reconstruction / weighted loss 对应 `surrogate.py` 与 `train_surrogate.py`。
- 上游 outer attack 更新对应 `generate.py` 的 fake tensor 梯度、归一化更新和投影。
- 上游 victim 重训及攻击效果评估对应 `evaluate.py` 与 `fit.py`。

## 已知差异

1. 当前仓库默认不强制安装 `higher`，有限 unroll 使用 PyTorch 手工可微更新；因此 optimizer state 与上游 `higher.innerloop_ctx` 不是逐行等价。
2. WMF victim 的默认路径复用本项目已有 ALS 实现；surrogate 的可微路径使用 SGD 近似，不能声称等价于上游所有 WMF trainer 细节。
3. 当前投毒数据以项目 `meta.pkl`、`fake_data.npz`、`target_items.json`、`history.json`、`stats.json` 为主，不复制上游未影响训练/评估的 profile 展示文件。
4. 结果指标按本项目统一的目标物品过滤训练交互后排名口径计算；论文最终数值仍需在相同数据预处理、随机种子、模型超参数和硬件条件下核对。

## 验证证据

- AdvInject 契约测试：4 项通过。
- 项目全量单元测试：187 项通过。
- victim model 接口 smoke：`itemcf`、`itemae`、`ncf`、`multvae`、`cml`、`mf`、`wmf`、`lightgcn` 全部完成一次训练并输出全量排名。
- `ml100k` WMF smoke：完成 classify、data、model 三阶段，并生成 clean/poisoned 对比报告。

上述 smoke 结果仅证明流程和接口可运行，不代表论文最终实验结果。