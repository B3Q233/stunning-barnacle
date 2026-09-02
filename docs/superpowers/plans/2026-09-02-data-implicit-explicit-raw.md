# 原始数据按隐式/显式交互分层——实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 TPA 原始数据目录改为 `data/{implicit|explicit}/raw/{dataset}/`，并同步所有 raw 消费点、测试、仓库文档与对应 skill 模板。

**Architecture:** 在 `training/paths.py` 增加唯一数据集→交互类型映射与 raw 路径解析函数；lightgcn/mf/wmf 预处理脚本、wmf config、可视化脚本统一从该解析器取默认 raw 根（保留 `--raw_dir`/`--raw-root` 覆盖）；数据文件用 `git mv` 迁移，processed/dataset.py/攻击模块不动。

**Tech Stack:** Python 3.12（仓库 `.venv`）、unittest（仅标准库）、git。

## Global Constraints

- 只移动/新建原始数据目录；`models/*/data/processed/`、dataset.py、攻击模块不改。
- 目录层级固定：`TPA/data/{implicit|explicit}/raw/{dataset}/`。
- 当前数据集 gowalla / amazon-book / yelp2018 / ml100k 全部归 implicit；explicit 只建占位 README。
- 路径一律基于 TPA 项目根相对解析，禁止盘符硬编码；`--raw_dir`/`--raw-root` 覆盖能力保留。
- 全量回归命令：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_* -v`（测试不新增第三方依赖）。
- 提交信息用 Conventional Commits 中文描述；只 `git add` 明确路径。
- 对应 skill 更新范围仅 `.codex/skills/paper-code-implementation`（.claude 无命中）。

---

### Task 1: 公共 raw 路径解析器（TDD）

**Files:**
- Modify: `TPA/training/paths.py`
- Modify: `TPA/tests/test_portable_paths.py`（追加 RawDataResolverTest；更新 PreprocessDefaultPathTest）

**Interfaces:**
- Produces: `training.paths.IMPLICIT_RAW_DIR`、`EXPLICIT_RAW_DIR`（Path）；`DATASET_INTERACTION`（dict）；`raw_data_root(interaction: str) -> Path`；`raw_data_dir(dataset: str) -> Path`

- [ ] **Step 1: 写失败测试**（追加到 `TPA/tests/test_portable_paths.py`）

```python
class RawDataResolverTest(unittest.TestCase):
    """raw 根目录与数据集解析（implicit/explicit 分层）。"""

    def test_raw_roots(self):
        from training.paths import (
            EXPLICIT_RAW_DIR, IMPLICIT_RAW_DIR, raw_data_root,
        )
        self.assertEqual(IMPLICIT_RAW_DIR,
                         PROJECT_ROOT / "data" / "implicit" / "raw")
        self.assertEqual(EXPLICIT_RAW_DIR,
                         PROJECT_ROOT / "data" / "explicit" / "raw")
        self.assertEqual(raw_data_root("implicit"), IMPLICIT_RAW_DIR)
        self.assertEqual(raw_data_root("explicit"), EXPLICIT_RAW_DIR)

    def test_raw_data_dir_for_known_datasets(self):
        from training.paths import raw_data_dir
        for ds in ("gowalla", "amazon-book", "yelp2018", "ml100k"):
            self.assertEqual(
                raw_data_dir(ds),
                PROJECT_ROOT / "data" / "implicit" / "raw" / ds,
            )

    def test_raw_data_dir_unknown_raises(self):
        from training.paths import raw_data_dir
        with self.assertRaises(ValueError):
            raw_data_dir("filmtrust")

    def test_raw_data_root_unknown_raises(self):
        from training.paths import raw_data_root
        with self.assertRaises(ValueError):
            raw_data_root("unknown")
```

- [ ] **Step 2: 运行确认失败**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_portable_paths.RawDataResolverTest -v`
Expected: FAIL（`ImportError: cannot import name 'raw_data_root'`）

- [ ] **Step 3: 最小实现**（追加到 `TPA/training/paths.py`）

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMPLICIT_RAW_DIR = PROJECT_ROOT / "data" / "implicit" / "raw"
EXPLICIT_RAW_DIR = PROJECT_ROOT / "data" / "explicit" / "raw"

DATASET_INTERACTION = {
    "gowalla": "implicit",
    "amazon-book": "implicit",
    "yelp2018": "implicit",
    "ml100k": "implicit",
}


def raw_data_root(interaction: str) -> Path:
    """返回 implicit/explicit 的 raw 根目录；未知类型报错。"""
    if interaction == "implicit":
        return IMPLICIT_RAW_DIR
    if interaction == "explicit":
        return EXPLICIT_RAW_DIR
    raise ValueError(
        f"未知交互类型 {interaction!r}，可选: implicit | explicit"
    )


def raw_data_dir(dataset: str) -> Path:
    """返回 {dataset} 的原始数据目录；未知数据集报错并列出可用项。"""
    if dataset not in DATASET_INTERACTION:
        raise ValueError(
            f"未知数据集 {dataset!r}，已登记: {sorted(DATASET_INTERACTION)}"
        )
    return raw_data_root(DATASET_INTERACTION[dataset]) / dataset
```

- [ ] **Step 4: 运行确认通过**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_portable_paths.RawDataResolverTest -v`
Expected: PASS（4 个用例）

- [ ] **Step 5: 提交**

```bash
git add TPA/training/paths.py TPA/tests/test_portable_paths.py
git commit -m "feat(paths): raw 数据按隐式/显式类型解析目录"
```

---

### Task 2: 迁移原始数据目录 + 显式占位

**Files:**
- Move: `TPA/data/raw/{gowalla, amazon-book, yelp2018, ml100k}` → `TPA/data/implicit/raw/`
- Create: `TPA/data/explicit/raw/README.md`

**Interfaces:**
- Consumes: Task 1 的 `IMPLICIT_RAW_DIR` / `EXPLICIT_RAW_DIR` 常量路径

- [ ] **Step 1: git mv 四个数据集**

```bash
git mv TPA/data/raw/gowalla TPA/data/implicit/raw/gowalla
git mv TPA/data/raw/amazon-book TPA/data/implicit/raw/amazon-book
git mv TPA/data/raw/yelp2018 TPA/data/implicit/raw/yelp2018
git mv TPA/data/raw/ml100k TPA/data/implicit/raw/ml100k
```

- [ ] **Step 2: 创建显式占位 README**

`TPA/data/explicit/raw/README.md` 内容：

```markdown
# explicit/raw：显式评分原始数据

此目录存放显式评分（rating）数据集，如 MovieLens（1–5 星）、FilmTrust 等。

约定：
- 文件格式示例：`user_id item_id rating`（每行一条评分，tab 或空格分隔）；
- 0 = 未评分；评分上界由各数据集/配置的 `rating_scale` 声明；
- 接入新数据集时：① 文件放入 `{dataset}/`；② 在
  `TPA/training/paths.py` 的 `DATASET_INTERACTION` 登记为 `explicit`；
  ③ 按根 `AGENTS.md` §6.6 补齐显式评分配置与模型需求。
```

- [ ] **Step 3: 确认旧目录无残留**

Run: `Get-ChildItem TPA/data/raw -ErrorAction SilentlyContinue`
Expected: 无输出（目录已空/不存在）

- [ ] **Step 4: 提交**

```bash
git add -A TPA/data
git commit -m "refactor(data): 原始数据迁入 implicit/raw 并预留 explicit/raw"
```

---

### Task 3: 更新 raw 消费点（预处理脚本 / wmf config / 可视化 / 路径测试）

**Files:**
- Modify: `TPA/models/lightgcn/scripts/preprocess.py`
- Modify: `TPA/models/mf/scripts/preprocess.py`
- Modify: `TPA/models/wmf/scripts/preprocess.py`
- Modify: `TPA/models/wmf/config.yaml`
- Modify: `TPA/visualization/item_freq/plot_item_freq.py`
- Modify: `TPA/tests/test_portable_paths.py`（PreprocessDefaultPathTest 期望值）

**Interfaces:**
- Consumes: `training.paths.IMPLICIT_RAW_DIR`

- [ ] **Step 1: 三个 preprocess.py 使用公共常量**

每个文件在 `import os/argparse` 段补 `import sys`，并把
`PROJECT_ROOT` 定义之后改为：

```python
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from training.paths import IMPLICIT_RAW_DIR

DEFAULT_RAW_DIR = IMPLICIT_RAW_DIR
```

同步把文件顶部 docstring 中 `data/raw/{dataset}` 改为
`data/implicit/raw/{dataset}`。

- [ ] **Step 2: wmf config.yaml 指向新路径**

`TPA/models/wmf/config.yaml`：

```yaml
  raw_data_path: data/implicit/raw/ml100k   # [unreported] 原始成对数据目录（train.txt / test.txt）
```

上一行注释 `本地 TPA/data/raw/ml100k` 同步改为 `本地 TPA/data/implicit/raw/ml100k`。

- [ ] **Step 3: visualization RAW_ROOT 改用公共常量**

`TPA/visualization/item_freq/plot_item_freq.py` 顶部：

```python
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from training.paths import IMPLICIT_RAW_DIR

RAW_ROOT = IMPLICIT_RAW_DIR
```

并同步 docstring 中 `TPA/data/raw/{dataset}/train.txt` →
`TPA/data/implicit/raw/{dataset}/train.txt`。

- [ ] **Step 4: 更新路径回归测试期望**

`TPA/tests/test_portable_paths.py` 中：

```python
self.assertEqual(module.DEFAULT_RAW_DIR, PROJECT_ROOT / "data" / "implicit" / "raw")
```

- [ ] **Step 5: 运行相关测试**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_portable_paths -v`
Expected: PASS

- [ ] **Step 6: 冒烟：临时目录跑 wmf preprocess**

```powershell
$env:TMPOUT = Join-Path $env:TEMP ("legup-smoke-" + [guid]::NewGuid())
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\models\wmf\scripts\preprocess.py --dataset ml100k --out_dir $env:TMPOUT
```
Expected: 打印 `=== ml100k 预处理 ===` 且 `预处理完成 -> {TMPOUT}/ml100k/`。

- [ ] **Step 7: 提交**

```bash
git add TPA/models/lightgcn/scripts/preprocess.py TPA/models/mf/scripts/preprocess.py TPA/models/wmf/scripts/preprocess.py TPA/models/wmf/config.yaml TPA/visualization/item_freq/plot_item_freq.py TPA/tests/test_portable_paths.py
git commit -m "refactor(data): raw 消费点改用 implicit/raw 公共路径"
```

---

### Task 4: 仓库文档同步

**Files:**
- Modify: `AGENTS.md`（§4 数据入库行）
- Modify: `.gitignore`（数据集入库注释）
- Modify: `TPA/models/lightgcn/docs/USAGE.md`、`IMPLEMENTATION_DOCS.md`
- Modify: `TPA/models/mf/docs/USAGE.md`
- Modify: `TPA/models/wmf/docs/USAGE.md`
- Modify: `TPA/visualization/item_freq/README.md`

**Interfaces:**
- Consumes: 最终目录 `TPA/data/{implicit|explicit}/raw/`

- [ ] **Step 1: 逐文件替换 raw 表述**

规则：把各文件中的 `data/raw/` 目录示例改为 `data/implicit/raw/`；凡提到
“显式评分/rating 数据集放这里”的新增说明统一指向 `data/explicit/raw/`。

必改位置：

```text
AGENTS.md:43           原始数据 `data/raw/` → `data/{implicit|explicit}/raw/`
.gitignore:38          注释同 AGENTS.md
TPA/models/lightgcn/docs/USAGE.md:7,47,50
TPA/models/lightgcn/docs/IMPLEMENTATION_DOCS.md:16-17
TPA/models/mf/docs/USAGE.md:24
TPA/models/wmf/docs/USAGE.md:37,39
TPA/visualization/item_freq/README.md:15
```

- [ ] **Step 2: 自查无旧路径残留**

Run: `git grep -n "data/raw" -- AGENTS.md .gitignore TPA/models TPA/visualization`
Expected: 无命中（已排除 skill 目录与历史快照）

- [ ] **Step 3: 提交**

```bash
git add AGENTS.md .gitignore TPA/models/lightgcn/docs TPA/models/mf/docs TPA/models/wmf/docs TPA/visualization/item_freq/README.md
git commit -m "docs(data): 同步隐式/显式 raw 目录约定"
```

---

### Task 5: 同步对应 skill 模板

**Files:**
- Modify: `.codex/skills/paper-code-implementation/SKILL.md`
- Modify: `.codex/skills/paper-code-implementation/references/config_template.yaml`
- Modify: `.codex/skills/paper-code-implementation/references/project_structure.md`
- Modify: `.codex/skills/paper-code-implementation/references/usage_doc_template.md`

**Interfaces:**
- Consumes: 最终目录 `data/{implicit|explicit}/raw/` 约定

- [ ] **Step 1: SKILL.md 数据集准备段**

`SKILL.md` 约 L307–312 改为：

```markdown
3. **数据集准备**：数据集去哪里下载（官方链接）、下载后按反馈类型放入
   `data/{implicit|explicit}/raw/{dataset}/` 下的具体哪个子路径、目录结构示例
   （隐式交互=user item 交互；显式评分=user item rating）

① 下载数据集 → 按类型放入 data/implicit/raw/ 或 data/explicit/raw/
② 在 config.yaml 的 data 模块中配置 raw_data_path（并在
   training/paths.py 的 DATASET_INTERACTION 登记类型）
③ 运行预处理脚本，生成 data/processed/
```

- [ ] **Step 2: config_template.yaml**

```yaml
  raw_data_path: "data/{implicit|explicit}/raw/dataset_name"   # [unreported] 先按隐式/显式类型归类，再填路径
```

- [ ] **Step 3: project_structure.md**

```text
mkdir -p data/implicit/raw data/explicit/raw data/processed
```

§55 描述同步为：`data/{implicit|explicit}/raw/` 不要由代码自动下载填充；
隐式交互与显式评分数据集分目录放置，路径须在 `training/paths.py` 登记。

- [ ] **Step 4: usage_doc_template.md**

把目录表、下载放置路径、示例结构中所有 `data/raw/` 改为
`data/{implicit|explicit}/raw/`，并在下载步骤加一句“先判断数据是隐式交互
还是显式评分，再放入对应类型目录”。

- [ ] **Step 5: 自查无旧路径残留（skill 范围）**

Run: `rg -n "data/raw" G:\Idea\.codex\skills\paper-code-implementation`
Expected: 无命中（`data/processed` 相关行保留）

- [ ] **Step 6: 提交**

```bash
git add .codex/skills/paper-code-implementation
git commit -m "docs(skill): 数据目录模板同步隐式/显式 raw 分层"
```

---

### Task 6: 全量回归与交付

**Files:** 无新增（验证 + 收尾）

- [ ] **Step 1: 全量测试**

Run: `cd G:\Idea\TPA; G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_* -v`
Expected: 全部 PASS

- [ ] **Step 2: 仓库级自查**

Run:

```powershell
git status --short
git grep -n "data/raw" -- AGENTS.md .gitignore TPA .codex/skills/paper-code-implementation
```

Expected: 改动文件与 Task 2–5 清单一致；上述 `git grep` 无命中（历史快照
`*/outputs/*` 不在跟踪范围）。

- [ ] **Step 3: 向用户交付变更摘要**

列出：目录迁移结果、公共解析器接口、消费点/测试/文档/skill 改动、测试结论、
提交列表。

---

## Self-Review 结论

- Spec 覆盖：目录结构（Task 2）、公共解析器（Task 1）、消费点（Task 3）、
  仓库文档（Task 4）、skill（Task 5）、验证（Task 6）全部有对应任务。
- 无占位符；所有代码步骤含可直接复制内容。
- 类型一致：`raw_data_root(interaction)` / `raw_data_dir(dataset)` 在 Task 1
  定义并在后续任务仅作为常量来源使用，无命名漂移。
