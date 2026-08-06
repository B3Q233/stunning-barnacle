# 项目目录结构规范

新建复现项目时，在用户指定的项目根目录下直接执行（以根目录为 `<ROOT>` 代表，例如 `test/model`）：

```bash
cd <ROOT>
mkdir -p data/raw data/processed
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
4. `data/raw/` 不要由代码自动下载填充（除非用户明确提供了可程序化下载的链接并同意），优先提示用户手动下载并核对版本后放入此目录；若数据集明确提供了官方下载脚本/API，可以使用，但要在 `docs/USAGE.md` 中写清楚来源链接。
5. `.venv` 一定建在 `<ROOT>` 内部，不使用用户的全局 Python 环境，也不建在 `<ROOT>` 的上层目录或系统默认位置。
