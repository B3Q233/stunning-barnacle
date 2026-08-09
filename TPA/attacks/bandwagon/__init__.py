"""Bandwagon（从众）攻击基线。

模块分离设计：
- generate.py   纯数据层：生成并注入假用户画像（不依赖 torch / 模型）
- fit.py        中毒模型拟合：新建 LightGCN 并在中毒数据上训练（可选 warm-start）
- evaluate.py   攻击效果评估：目标物品曝光指标 + clean vs poisoned 对比
"""
