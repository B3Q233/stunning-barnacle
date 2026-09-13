"""UBA（Uplift-guided Budget Allocation）—— 目标用户攻击的 uplift 预算分配框架。

论文：Uplift Modeling for Target User Attacks on Recommender Systems（WWW '24）
官方代码：https://github.com/Wcsa23187/UBA

模块分离设计（与仓库攻击模板一致，另加 estimate 阶段）：
- classify.py  纯数据层：按训练集交互数划分物品三档（流行/普通/冷门）
- uplift.py    纯算法层：目标用户选择 / A'³ 三跳路径处理效应 / 分组背包预算分配
- estimate.py  数据+模型层：w/ S_φ 支路用代理模型做模拟实验估计处理效应（写共享缓存）
- generate.py  数据层：预算分配（读缓存）→ 按画像规则实例化假档案 → 注入中毒数据
- fit.py       模型层：warm-start 投毒训练 + clean/poisoned 对比评估
- evaluate.py  评估薄壳：共享实现见 evaluation/attack_eval.py（另加目标用户群指标）
- run.py       编排入口：classify | estimate | data | model | both | all
"""
