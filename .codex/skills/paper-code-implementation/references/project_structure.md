# 项目目录结构规范

新建复现项目时，在用户指定的项目根目录下直接执行（以根目录为 `<ROOT>` 代表，例如 `test/model`）：

```bash
cd <ROOT>
mkdir -p data/implicit/raw data/explicit/raw data/processed
mkdir -p datasets models evaluation training
mkdir -p outputs/checkpoints
mkdir -p docs
python -m venv .venv
```

完整结构：

```
<ROOT>/
├── main.py
├── config.yaml
├── requirements.txt
├── .venv/
├── data/
│   ├── raw/
│   └── processed/
├── datasets/
│   ├── __init__.py
│   └── paper_dataset.py
├── models/
│   ├── __init__.py
│   └── paper_model.py
├── evaluation/
│   ├── __init__.py
│   └── metrics.py
├── training/
│   ├── __init__.py
│   └── framework.py
├── outputs/
│   ├── checkpoints/
│   ├── training_curve.png
│   └── comparison_table.md
└── docs/
    ├── IMPLEMENTATION_DOCS.md
    └── USAGE.md
```

## 规则

1. 根目录只允许 `main.py` 和 `config.yaml` 两个文件，其余都进对应文件夹。
2. 每个文件夹只放该阶段的代码，不跨阶段混放逻辑。
3. `training/framework.py` = `tool_template.py` 的落地版本：固定框架部分（`TrainableModel`/`DatasetProtocol`/`TrainingConfig`/`ConfigBuilder`/`Experiment`/`Trainer`/`Callback` 体系）原样保留，`PaperModel`/`PaperDataset`/`PaperDataLoader` 分别拆到各自文件夹的对应文件中，`framework.py` 中通过 import 引用。
4. 实验隔离 run_tag（必须执行）：每次实验输出到 `outputs/{tag}/`（checkpoints / history /
   eval_log / config.yaml 快照），默认 tag = 当前时间（`%Y-%m-%d-%H:%M`，路径中 `:` 替换为
   `-`）；训练结束后复制最新 checkpoint 到稳定指针 `outputs/checkpoints/latest.pt`，并在
   `outputs/latest.json` 记录本次 run_tag。统一工具见 `assets/run_tag.py`。
4. `data/{implicit|explicit}/raw/` 不要由代码自动下载填充（除非用户明确提供了可程序化下载的链接并同意），优先提示用户手动下载并核对版本后放入对应类型目录（隐式交互 / 显式评分）；若数据集明确提供了官方下载脚本/API，可以使用，但要在 `docs/USAGE.md` 中写清楚来源链接；项目若使用 `training/paths.py`，新数据集须在 `DATASET_INTERACTION` 登记。
5. `.venv` 一定建在 `<ROOT>` 内部，不使用用户的全局 Python 环境，也不建在 `<ROOT>` 的上层目录或系统默认位置。
