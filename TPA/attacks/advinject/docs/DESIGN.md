# AdvInject 设计说明

攻击输出不是 surrogate loss 本身。surrogate loss 只用于更新伪用户交互；最终效果必须在拼接伪用户后的 victim model 上重新训练，并报告目标物品排序指标。

流程边界：

1. `data.py` 负责读取 clean meta、目标选择和投毒 meta 保存。
2. `surrogate.py` 负责代理模型目标损失和伪数据优化。
3. `evaluate.py` 负责 victim 重训与目标指标。
4. `run.py` 负责 run_tag、配置快照和阶段编排。

后续扩展有限 unroll 时，应保持这四个边界，不把 victim model 重复放入 attack 目录。
