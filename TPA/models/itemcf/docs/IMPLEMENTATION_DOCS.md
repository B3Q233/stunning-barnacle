# ItemCF 实现说明

## 数据协议

通过 models.revisit_common.load_meta 读取 models/wmf/data/processed/{dataset}/meta.pkl。元数据至少包含 
um_users、
um_items、	rain_pairs 和 	est_pairs。

## 构造协议

模型统一支持：

`python
model_cls(config, num_users, num_items, edge_index=None)
`

其中 edge_index 是兼容现有模型和后续攻击调用的可选参数；ItemCF 不依赖图结构时会忽略该参数。

## 训练与评估

模型提供 	rain_step 和 eval_step，并通过 predict_full_ranking 输出用户对全部物品的分数。models.revisit_training.ranking_report 基于测试交互计算 Recall@K 和 NDCG@K。

## 配置与产物

唯一配置入口是对应目录的 config.yaml。un_tag 用于隔离实验，训练后保存配置快照、history.json、metrics.json 和 checkpoints/last.pt。

## 边界处理

用户 ID、物品 ID 和输入矩阵维度在模型或数据辅助模块中校验；非法输入会抛出明确异常。

## 目录职责

本目录只实现 victim model，不在 TPA/attacks/advinject/ 中重复实现模型。
