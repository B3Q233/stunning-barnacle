# 原始数据目录按隐式/显式交互分层——设计文档

> 日期：2026-09-02
> 状态：待人工审阅（brainstorming 已确认方案 A）
> 关联规范：根 `AGENTS.md` §4/§6；`TPA/` 复现模板约定

## 1. 背景与目标

仓库现有原始数据统一放在 `TPA/data/raw/{dataset}/`，未区分反馈类型。后续要接入
显式评分数据集与评分型模型/攻击（如 Leg-UP），需要先把原始数据目录按
**隐式交互（implicit）** 与 **显式交互（explicit）** 分层，并同步所有读取
raw 的消费点与对应 skill 模板。

已确认的决策（brainstorming 结论）：

1. 只移动/新建**原始数据集**目录；`models/*/data/processed/`、dataset.py、
   各攻击模块（只读 processed）一律不动。
2. 目录层级采用 `data/{implicit|explicit}/raw/{dataset}/`。
3. 当前四个数据集（gowalla、amazon-book、yelp2018、ml100k）在仓库内均为
   隐式交互格式，全部归入 `implicit`，不做内容转换。
4. 显式目录仅新建空占位（`data/explicit/raw/README.md`），评分数据后续接入。
5. 采用**公共解析器（方案 A）**统一 raw 路径，避免各消费点重复维护映射。
6. 改动完成后自动同步仓库内**对应 skill**（`.codex/skills/` 下引用旧
   `data/raw` 约定的模板与文档）。

## 2. 目标目录结构

```
TPA/data/
├── implicit/raw/
│   ├── gowalla/       # 由 data/raw/gowalla git mv 而来
│   ├── amazon-book/   # 由 data/raw/amazon-book git mv 而来
│   ├── yelp2018/      # 由 data/raw/yelp2018 git mv 而来
│   └── ml100k/        # 由 data/raw/ml100k git mv 而来
└── explicit/raw/
    └── README.md      # 占位说明：显式评分数据集（user item rating）后续放此
```

`data/raw/` 迁移后删除（git 不跟踪空目录）。所有迁移用 `git mv`，保留历史。

## 3. 公共解析器设计（方案 A）

在 `TPA/training/paths.py`（现有跨平台路径解析模块）追加：

```python
IMPLICIT_RAW_DIR = PROJECT_ROOT / "data" / "implicit" / "raw"
EXPLICIT_RAW_DIR = PROJECT_ROOT / "data" / "explicit" / "raw"

# dataset -> interaction_type；显式数据集接入时在此登记
DATASET_INTERACTION = {
    "gowalla": "implicit",
    "amazon-book": "implicit",
    "yelp2018": "implicit",
    "ml100k": "implicit",
}

def raw_data_dir(dataset: str) -> Path:
    """返回 {dataset} 的原始数据目录（= raw_data_root(类型)/{dataset}）；
    未知数据集报错并列出 DATASET_INTERACTION 可用项。"""

def raw_data_root(interaction: str) -> Path:
    """返回 implicit/explicit 的 raw 根目录；未知类型报错。"""
```

约束：

- 所有 raw 消费点默认路径必须来自该解析器，不再各自写死根目录。
- `--raw_dir` 之类 CLI 覆盖能力保留（显式数据集可显式传
  `data/explicit/raw/{dataset}` 或绝对路径）。
- 解析器基于 `PROJECT_ROOT` 相对解析，禁止盘符硬编码（延续
  `test_portable_paths.py` 的既有约定）。

## 4. 改动清单

### 4.1 数据目录（git mv）

- `TPA/data/raw/gowalla` → `TPA/data/implicit/raw/gowalla`
- `TPA/data/raw/amazon-book` → `TPA/data/implicit/raw/amazon-book`
- `TPA/data/raw/yelp2018` → `TPA/data/implicit/raw/yelp2018`
- `TPA/data/raw/ml100k` → `TPA/data/implicit/raw/ml100k`
- 新增 `TPA/data/explicit/raw/README.md`（内容：说明该目录用于显式评分
  数据集；格式示例 `user_id item_id rating`；0=未评分语义；接入时需同时登记
  `training/paths.py` 映射并按根 `AGENTS.md` §6.6 预留槽位补齐配置）

### 4.2 代码

| 文件 | 改动 |
|---|---|
| `TPA/training/paths.py` | 增加 implicit/explicit 常量、`DATASET_INTERACTION`、`raw_data_dir()` / `raw_data_root()` |
| `TPA/models/lightgcn/scripts/preprocess.py` | `DEFAULT_RAW_DIR` 改用公共解析（implicit 根）；docstring/注释同步；`--raw_dir` 保留 |
| `TPA/models/mf/scripts/preprocess.py` | 同上 |
| `TPA/models/wmf/scripts/preprocess.py` | 同上 |
| `TPA/models/wmf/config.yaml` | `raw_data_path: data/implicit/raw/ml100k`（注释同步） |
| `TPA/visualization/item_freq/plot_item_freq.py` | `RAW_ROOT` 改用公共解析（implicit 根）；docstring 同步 |

### 4.3 测试

- `TPA/tests/test_portable_paths.py`：`DEFAULT_RAW_DIR` 断言更新为新 implicit
  根；新增对 `raw_data_dir()` / `raw_data_root()` 的用例（含未知数据集报错）。
- 全量回归命令：`<repo>\.venv\Scripts\python.exe -m unittest tests.test_* -v`

### 4.4 文档（仓库内）

- 根 `AGENTS.md` §4："原始数据 `data/raw/`" → "原始数据
  `data/{implicit|explicit}/raw/`"。
- `.gitignore` 数据集入库注释同步（不影响任何 ignore 规则本身）。
- `TPA/models/{lightgcn,mf,wmf}/docs/USAGE.md` 与
  `IMPLEMENTATION_DOCS.md` 中的 `data/raw/{dataset}` 表述。
- `TPA/visualization/item_freq/README.md` 中 raw 路径表述。

### 4.5 对应 skill 自动更新

范围：`.codex/skills/paper-code-implementation`（仓库内唯一引用旧 raw 约定的
skill；`.claude` 目录无命中）。改动仅为路径约定表述，不动方法论。

| 文件 | 改动要点 |
|---|---|
| `SKILL.md`（约 L307–312） | `data/raw/` → `data/{implicit|explicit}/raw/`，说明按反馈类型选择目录并在 `training/paths.py` 登记 |
| `references/config_template.yaml` | `raw_data_path` 示例改为 `data/{implicit|explicit}/raw/dataset_name`，注释说明先归类再填路径 |
| `references/project_structure.md` | `mkdir -p data/raw data/processed` → `mkdir -p data/{implicit,explicit}/raw data/processed`；§55 描述同步 |
| `references/usage_doc_template.md` | 目录表、下载放置路径、示例结构中的 raw 表述同步 |

说明：skill 中关于 `data/processed/` 的通用模板表述不属于本次"原始数据
分层"范围，除与 raw 路径同句出现的必要修改外不做大改。

## 5. 验证方式

1. `unittest` 全量回归通过。
2. 用临时 `--out_dir`（如 `tmp/`）对 ml100k 跑一次 wmf/lightgcn preprocess
   冒烟，确认能从新 implicit 根读到 raw（不重写仓库内 processed 文件）。
3. `git status` 确认旧 `data/raw` 无残留、改动文件与清单一致。

## 6. 明确不做（Out of scope）

- 不移动/重建 `models/*/data/processed/`；不修改各 dataset.py 与攻击模块
  （random/bandwagon/pgd/tpa/advinject/batch）——它们不读 raw。
- 不把 ml100k 转成显式评分格式；不新增任何评分数据集文件。
- 不修改 `.codex/skills/` 之外的其它 skill；不引入新第三方依赖。
- 不处理历史 output 快照中的旧路径字符串。

## 7. 提交

单次 Conventional Commits：`refactor(data): 原始数据按隐式/显式交互分层目录`，
仅包含 §4 列出的文件（数据 git mv、代码、测试、文档、skill 模板）。

## 8. 风险与备注

- 存在运行中的脚本/配置硬编码旧 raw 路径（含未跟踪副本），迁移后需重跑
  preprocess 或改传 `--raw_dir`；文档统一更新以降低此类风险。
- `explicit/raw` 目前为空占位，接入显式数据集时必须同步登记
  `DATASET_INTERACTION` 并按根 `AGENTS.md` §6.6 补齐配置与模型需求。
