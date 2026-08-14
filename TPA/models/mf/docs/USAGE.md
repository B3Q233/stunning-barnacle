# MF（矩阵分解）模型 使用文档

## 1. 项目结构

```
models/mf/
├── model.py          # MatrixFactorization（TrainableModel，BPR 训练）
├── dataset.py        # MFDataset / MFDataLoader（与 LightGCN 同构）
├── config.yaml       # 超参数配置
├── main.py           # 训练入口（支持 --resume）
├── train.py          # 训练循环（全量排序评估回调）
├── scripts/preprocess.py  # 数据预处理（raw -> meta.pkl）
├── data/processed/   # 预处理产物（{dataset}/meta.pkl）
├── outputs/          # checkpoints + history.json + eval_log.csv
└── docs/             # 本文档 + IMPLEMENTATION_DOCS.md
```

## 2. 环境准备

使用仓库根虚拟环境（如 Windows 下 `G:\Idea\.venv`，Linux 下 `.venv/bin/python`）。

## 3. 数据准备

把数据集放到 `<仓库>/TPA/data/raw/{dataset}/train.txt / test.txt`，然后：

```powershell
python TPA/models/mf/scripts/preprocess.py --dataset ml100k
```

支持 `ml100k / gowalla / yelp2018 / amazon-book`。

## 4. 训练

```powershell
python TPA/models/mf/main.py
```

断点续训：加 `--resume`。训练完成后查看 `outputs/eval_log.csv` 与 `outputs/history.json`。

## 5. 配置说明（config.yaml）

| 参数 | 默认 | 说明 |
|------|------|------|
| data.dataset | ml100k | 数据集名 |
| model.emb_dim | 64 | 嵌入维度 |
| model.init_method | normal | 嵌入初始化（N(0,0.1)）|
| training.lr | 0.001 | 学习率 |
| training.epochs | 100 | 训练轮数 |
| training.batch_size | 256 | batch 大小 |
| training.weight_decay | 0.0001 | L2 正则系数 |
| evaluation.k | 20 | 评估 Top-K |

## 6. 常见问题

- **Q1：换数据集后训练报 FileNotFoundError？**
  先运行对应的 preprocess 生成 `models/mf/data/processed/{dataset}/meta.pkl`。
- **Q2：为什么 MF 不做图卷积？**
  MF 本身就是矩阵分解：`M ≈ U V^T`，直接查嵌入表即可；这与 LightGCN 的区别
  正是"有没有 LGC 传播"，接口完全一致。
