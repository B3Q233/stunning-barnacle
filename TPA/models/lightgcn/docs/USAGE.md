# LightGCN 使用文档

## 1. 项目结构

```
TPA/
├── data/raw/                          ← 原始数据集（需手动下载）
│   ├── gowalla/  (train.txt, test.txt, user_list.txt, item_list.txt)
│   ├── yelp2018/ (同上)
│   └── amazon-book/ (同上)
├── training/framework.py              ← 训练骨架（TrainableModel / Trainer / Callback）
├── evaluation/metrics.py              ← recall@K / ndcg@K（all-ranking 协议）
├── docs/
│   ├── IMPLEMENTATION_DOCS.md         ← 六步实现文档
│   └── USAGE.md                       ← 本文件
└── models/lightgcn/
    ├── main.py                        ← 训练入口
    ├── train.py                       ← 组装车间（DataLoader→Model→Trainer）
    ├── config.yaml                    ← 超参数配置
    ├── model.py                       ← LightGCN 模型
    ├── dataset.py                     ← 数据载入器
    ├── data/processed/                ← 预处理后的数据
    │   └── {dataset}/
    │       ├── train_pairs.txt
    │       ├── test_pairs.txt
    │       └── meta.pkl
    ├── scripts/preprocess.py          ← 预处理脚本
    └── outputs/
        ├── checkpoints/               ← 模型权重 + 训练状态
        ├── history.json               ← 训练指标历史
        ├── eval_log.csv               ← 定期评估记录
        ├── training_curve.png         ← 训练曲线图
        └── comparison_table.md        ← 复现值 vs 论文值对比表
```

## 2. 环境准备

项目使用仓库根共享虚拟环境（如 Windows 下 `G:\Idea\.venv`，Linux 下 `.venv/bin/python`）。首次使用：

```bash
# 激活环境（Linux: source .venv/bin/activate）
python models/lightgcn/main.py
```

## 3. 数据集准备

数据集来自 NGCF 论文作者。请将数据放入 `data/raw/{dataset}/`：

```
data/raw/gowalla/
├── train.txt       ← 每行: user_id item1 item2 ...（NGCF 格式）
├── test.txt
├── user_list.txt   ← org_id remap_id
└── item_list.txt   ← org_id remap_id
```

## 4. 复现流程

```
① 数据预处理（已生成 → data/processed/）
   python models/lightgcn/scripts/preprocess.py --dataset gowalla
   python models/lightgcn/scripts/preprocess.py --dataset yelp2018
   python models/lightgcn/scripts/preprocess.py --dataset amazon-book

② 修改 config.yaml 中的 dataset 字段选择数据集

③ 开始训练
   cd TPA
   python models/lightgcn/main.py

④ 断点续训（从上次 checkpoint 恢复）
   python models/lightgcn/main.py --resume

⑤ 训练完成后生成结果
   python models/lightgcn/outputs/plot_results.py gowalla
```

## 5. 配置文件说明 (config.yaml)

### data 模块
| 参数 | 含义 | 默认值 | 可调范围 |
|------|------|--------|---------|
| dataset | 数据集名 | gowalla | gowalla / yelp2018 / amazon-book |

### model 模块
| 参数 | 含义 | 默认值 | 可调范围 | 来源 |
|------|------|--------|---------|------|
| emb_dim | 嵌入维度 | 64 | 8/16/32/64/128 | [paper] |
| n_layers | LGC 层数 | 3 | 1~4 | [paper] |

### training 模块
| 参数 | 含义 | 默认值 | 可调范围 | 来源 |
|------|------|--------|---------|------|
| lr | 学习率 | 0.001 | >0 | [paper] Adam 默认 |
| epochs | 训练轮数 | 1000 | 正整数 | [paper] |
| batch_size | 批大小 | 1024 | 正整数, Amazon-Book 用 2048 | [paper] |
| weight_decay | L2 正则系数 λ | 1e-4 | {1e-6,...,1e-2} | [paper] |
| device | 设备 | cuda | cuda / cuda:0..N / cpu | [unreported] |
| num_workers | DataLoader 子进程数 | 4 | 0~8 | [ai] |
| persistent_workers | 跨 epoch 复用 worker（需 num_workers>0） | true | true/false | [ai] |

### evaluation 模块
| 参数 | 含义 | 默认值 | 来源 |
|------|------|--------|------|
| k | Top-K | 20 | [paper] |
| eval_every | 全量评估间隔(epoch) | 1 | [ai] |
| metrics | 评估指标 | [recall@20, ndcg@20] | [paper] |

## 6. 结果解读

训练完成后，在 `models/lightgcn/outputs/` 下查看：

- `history.json` — 每 epoch 的 loss/val_loss + 评估指标（recall@20/ndcg@20）
- `eval_log.csv` — 每 eval_every epoch 的 recall@20/ndcg@20（表格格式，可直接粘贴到论文）
- `training_curve.png` — 训练/验证 loss 曲线
- `comparison_table.md` — 复现值 vs 论文值对比表，偏差 ±2% 内为对齐

## 7. 常见问题

- **显存不足**: 调小 `batch_size`（如 512），或换 `device: cpu`（慢）
- **loss 不变 (0.6931)**: 检查 weight_decay 是否太大（>1e-2 会压制学习）
- **指标为 0**: 检查 `evaluation/metrics.py` 中 train_user_items 和 test_user_items 是否正确传入
- **断点续训**: `python models/lightgcn/main.py --resume`，会从 `outputs/checkpoints/latest.pt` 恢复模型、优化器、epoch 计数
- **配置文件中文乱码**: 用 UTF-8 编码保存，用 Python yaml.safe_load 读取
