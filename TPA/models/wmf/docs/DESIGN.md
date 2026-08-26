# WMF 设计

WMF 使用用户因子与物品因子建模隐式反馈。默认 `training.optimizer: als` 保留原有闭式交替最小二乘；`training.optimizer: sgd` 使用可微加权平方损失和 L2 正则更新因子，供后续 AdvInject surrogate/unroll 阶段调用。两条路径共享构造、全量评分、checkpoint 与注册表接口。