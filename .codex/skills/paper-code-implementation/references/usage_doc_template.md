# USAGE.md 模板

## 1. 项目结构说明

| 文件夹/文件 | 用途 |
|---|---|
| `main.py` | 唯一运行入口 |
| `config.yaml` | 唯一需要手动修改的配置文件 |
| `data/raw/` | 存放原始数据集（需手动下载） |
| `data/processed/` | 数据预处理后的产出，由预处理脚本生成 |
| `datasets/` | 数据集类与数据加载器 |
| `models/` | 模型结构定义 |
| `evaluation/` | 评估指标计算 |
| `training/` | 训练框架核心代码 |
| `outputs/` | 训练曲线、checkpoint、对比表等产出 |
| `docs/` | 实现文档与本使用文档 |

## 2. 环境准备

```bash
cd <项目根目录>
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / Mac
source .venv/bin/activate

pip install -r requirements.txt
```

## 3. 数据集准备

- 数据集名称：{}
- 官方下载地址：{}
- 下载后请将文件放入：`data/raw/{具体子路径}`
- 目录结构示例：
  ```
  data/raw/
  └── {dataset_name}/
      ├── train.csv
      └── test.csv
  ```

## 4. 复现完整流程（按操作顺序，非代码实现顺序）

```
① 下载数据集 → 放入 data/raw/{具体路径}
② 编辑 config.yaml 中 data.raw_data_path 为实际路径
③ 运行预处理：python -m scripts.preprocess --config config.yaml
   → 生成 data/processed/ 下的文件
④ 编辑 config.yaml 中 data.processed_data_path（如有变化）
⑤ 开始训练：python main.py --config config.yaml --model {model_name}
⑥ 训练完成后查看：
   - outputs/training_curve.png（训练曲线）
   - outputs/comparison_table.md（与论文指标对比）
```

## 5. 配置文件详解

### data 模块
| 参数 | 含义 | 默认值 | 可调范围 | 调整影响 |
|---|---|---|---|---|
| raw_data_path | 原始数据路径 | - | 任意有效路径 | 改变后需重新运行①②确认能正确读取 |
| negative_sampling_ratio | 负采样比例 | 4 | 正整数 | 影响训练数据量和训练速度，过大可能导致训练变慢 |

### model 模块
| 参数 | 含义 | 默认值 | 可调范围 | 调整影响 |
|---|---|---|---|---|
| embedding_dim | 嵌入维度 | 64 | 正整数，建议 8/16/32/64/128 | 改变后需重新验证③模型结构的输出 shape |

### training 模块
| 参数 | 含义 | 默认值 | 可调范围 | 调整影响 |
|---|---|---|---|---|
| optimizer | 优化器 | adam | 仅限 adam / sgd | 其他值会报错，因为 train_step 只实现了这两种分支逻辑 |
| batch_size | 批大小 | 1024 | 正整数 | 显存不足时调小；过小会拖慢收敛 |

### evaluation 模块
| 参数 | 含义 | 默认值 | 可调范围 | 调整影响 |
|---|---|---|---|---|
| metrics | 评估指标列表 | ["recall@20","ndcg@20"] | metrics.py 中已实现的指标名 | 加入未实现的指标名会报错 |

## 6. 常见问题

- **显存不足（CUDA out of memory）**：调小 `training.batch_size`，或在 `config.yaml` 中将 `training.device` 改为 `cpu`（速度会显著变慢）
- **训练中 loss 变为 NaN**：先检查 `data/processed/` 中的数据是否存在异常值（如除零导致的 inf），再检查 `training.lr` 是否过大
- **复现指标与论文差距较大**：优先检查 `evaluation` 模块的负采样/过滤策略是否与论文一致，这是最常见的指标偏差来源，其次再排查超参数
- **修改 config.yaml 后报错**：检查是否改动了"可调范围"之外的值（如 optimizer 改成了未实现的类型）
