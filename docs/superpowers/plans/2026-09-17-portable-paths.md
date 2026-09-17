# 入库文件可迁移性（禁止绝对路径）——实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 spec（`docs/superpowers/specs/2026-09-17-portable-paths-design.md`）把"禁止绝对路径 / 只引用可迁移位置"写进 AGENTS.md，并用守护测试 + 机械替换清掉仓库中 80 个跟踪文本文件的 418 处命中。

**Architecture:** 先建立可证伪的守护测试（RED），再按唯一映射表做机械替换（只匹配"盘符 + 冒号 + 分隔符"且左侧非字母数字），最后手工修复三类特殊命中（本机程序安装目录、外部盘符示例、既有测试里的字面断言）。

**Tech Stack:** Python 3.12、stdlib `unittest`、正则；不新增第三方依赖。

## Global Constraints

- 只处理 git 跟踪文件；不动数据文件、历史产物、`.gitignore` 跟踪策略。
- 不改业务逻辑、指标语义、配置键；机械替换必须保持换行符与编码不变（二进制读写）。
- 占位符约定：`<repo>`（仓库根）、`<papers>`、`<MinerU-Skill>`、`%TEMP%`、`%ProgramFiles%`。
- 命令示例统一 `python -m ...` 或 `<repo>\.venv\Scripts\python.exe -m ...`。
- 允许清单默认为空；守护测试必须含一条运行时拼接的反向用例。
- 测试命令（工作目录 `TPA`）：`..\.venv\Scripts\python.exe -m unittest <模块> -v`；全量：`..\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v`。
- 提交信息 Conventional Commits 中文；只 `git add` 明确路径。

## 唯一映射表（§3 of spec）

| 命中的仓库根前缀 | 替换为 |
|---|---|
| 根 + `.venv` | `<repo>\.venv` |
| 根 + `TPA` | `TPA` |
| 根 + `.codex` / `.claude` / `tmp` | `.codex` / `.claude` / `tmp` |
| 根 + `AGENTS.md` / `.gitignore` / `requirements.txt` | 同名相对路径 |
| 根 + `papers` | `<papers>` |
| 根 + `MinerU-Skill` | `<MinerU-Skill>` |
| 裸根 | `<repo>` |
| 外部盘符示例 | `outputs/<tag>/...` |
| 本机程序安装目录 | 环境变量拼接 + `shutil.which` |

---

## P1 守护测试（RED）

**Files:**
- Create: `TPA/tests/test_no_absolute_paths.py`

**Interfaces:**
- Produces: `REPO_ROOT`、`ABS_RE`、`POSIX_RE`、`UNC_RE`、`iter_tracked_text_files() -> list[Path]`、`find_violations(text) -> list[str]`

- [ ] **Step 1: 写测试**

```python
# -*- coding: utf-8 -*-
"""入库文本文件不得含绝对路径（AGENTS.md「路径与可迁移性」守护测试）。

扫描范围：``git ls-files`` 的跟踪文件（与"入库"定义一致），跳过二进制扩展名与
非 UTF-8 文本。四类命中：盘符路径、POSIX 用户目录、UNC 路径、本机程序目录
（后者的具体形态由盘符/UNC 两类规则覆盖）。

反向用例用运行时拼接构造样例，保证测试文件自身不含字面绝对路径。
"""
from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "TPA") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "TPA"))

ABS_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
POSIX_RE = re.compile(r"(?<![A-Za-z0-9.])/(?:home|Users|root|mnt|usr)/")
UNC_RE = re.compile(r"\\\\[A-Za-z0-9._-]+\\[A-Za-z0-9._$-]+")
SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".woff", ".woff2",
    ".ttf", ".map", ".js", ".pkl", ".pt", ".pth", ".npy", ".npz",
}


def iter_tracked_text_files() -> list[Path]:
    """git 跟踪的文本候选文件（排除二进制扩展名）。"""
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True)
    files: list[Path] = []
    for line in out.stdout.splitlines():
        path = REPO_ROOT / line
        if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
            continue
        files.append(path)
    return files


def find_violations(text: str) -> list[str]:
    """返回命中的绝对路径片段（按出现顺序去重）。"""
    found: list[str] = []
    for pattern in (ABS_RE, POSIX_RE, UNC_RE):
        for match in pattern.finditer(text):
            value = text[match.start():match.start() + 60].split()[0]
            if value not in found:
                found.append(value)
    return found


class NoAbsolutePathTest(unittest.TestCase):

    def test_tracked_text_files_have_no_absolute_paths(self):
        offenders: list[str] = []
        for path in iter_tracked_text_files():
            if path == Path(__file__).resolve():
                continue                     # 本文件含检测用的正则，非路径样例
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            violations = find_violations(text)
            if violations:
                rel = path.relative_to(REPO_ROOT).as_posix()
                offenders += [f"{rel}:{lineno}: {value}"
                              for lineno, value in violations[:3]]
        self.assertEqual([], offenders,
                         "以下入库文件含绝对路径，违反 AGENTS.md「路径与可迁移性」")

    def test_detector_catches_synthetic_absolute_path(self):
        """反向用例：运行时拼接样例，证明检测器不是恒真。"""
        sample = "C" + ":" + chr(92) + "Users" + chr(92) + "someone"
        self.assertTrue(find_violations(sample))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `..\.venv\Scripts\python.exe -m unittest tests.test_no_absolute_paths -v`
Expected: FAIL，offenders 列出约 80 个文件（含 `AGENTS.md`、`TPA/**`、`docs/superpowers/**`、`.codex/skills/**`）

---

## P2 AGENTS.md 新增规则章节

**Files:**
- Modify: `AGENTS.md`（新增第 9 节；同步修正第 3、4 节里自带的绝对路径示例）

- [ ] **Step 1: 新增「9. 路径与可迁移性（必须）」**

```markdown
## 9. 路径与可迁移性（必须）

- 入库文件（代码 / 注释 / 配置 / 文档 / spec / plan / USAGE / DESIGN / 测试 / 技能）
  **禁止出现绝对路径**：Windows 盘符路径（盘符 + 冒号 + 分隔符，含反斜杠、正斜杠、
  大小写、双反斜杠转义）、UNC 路径、POSIX 绝对路径（`/home`、`/Users`、`/root`、
  `/mnt`、`/usr`）以及本机程序安装目录/用户名目录。
- **禁止引用仓库根之外的本地文件**。只允许两类写法：① 仓库内相对路径（相对仓库根）；
  ② 占位符或环境变量：`<repo>`、`<papers>`、`<MinerU-Skill>`、`%TEMP%`、
  `%ProgramFiles%`、`$TPA_DATA_ROOT` 等。
- 文档与计划里的 `.gitignore` 资产（`papers/`、`MinerU-Skill/`、`.claude/`）
  一律按未入库资产写占位符，不得写成真实绝对路径。
- 命令示例：`python -m unittest ...`，或 `<repo>\.venv\Scripts\python.exe -m unittest ...`
  （POSIX：`<repo>/.venv/bin/python`）；工作目录标注与命令必须自洽。
- 代码禁止硬编码盘符或用户目录：一律 `Path(__file__)` / `PROJECT_ROOT` 推导；
  程序安装路径走 `%ProgramFiles%` 拼接或 `shutil.which`。
- 注释里的位置标记（本机仓库根字样）改为"仓库 TPA 根"这类无绝对路径的表述。
- 强制校验：`TPA/tests/test_no_absolute_paths.py` 扫描全部跟踪文本文件，命中即失败；
  允许清单必须逐条写明原因，且反向用例保证检测器可证伪。
```

- [ ] **Step 2: 修正 AGENTS.md 自身的示例**

第 1/3/4 节里三处解释器路径改为 `<repo>\.venv\Scripts\python.exe`，
数据集说明里的本机根字样改为"仓库根"。

---

## P3 机械替换（80 个文件）

**Files:**
- Modify: 全部命中的跟踪文本文件（不含 P4 手工处理的两个文件）

**Interfaces:**
- Consumes: §唯一映射表
- Produces: 替换后的文本；`git diff --stat` 可核对

- [ ] **Step 1: dry-run 统计**

```python
# 一次性脚本（放 tmp/，不入库）：只匹配"盘符 + 冒号 + 分隔符"，左侧非字母数字
import re, subprocess
from pathlib import Path

REPO_ROOT = Path.cwd()
REPO_NAME = "Idea"
SKIP = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".woff", ".woff2",
        ".ttf", ".map", ".js", ".pkl", ".pt", ".pth", ".npy", ".npz"}
SEGMENT_MAP = {
    ".venv": "<repo>/.venv", "TPA": "TPA", ".codex": ".codex",
    ".claude": ".claude", "tmp": "tmp", "papers": "<papers>",
    "MinerU-Skill": "<MinerU-Skill>", "AGENTS.md": "AGENTS.md",
    ".gitignore": ".gitignore", "requirements.txt": "requirements.txt",
}
PREFIX_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])[A-Za-z]:(?:\\\\|\\|/)+" + REPO_NAME
    + r"(?:(?:\\\\|\\|/)([A-Za-z0-9._-]+))?")
```

dry-run 输出 `文件 → 命中数`，人工抽查 10 条，确认无误伤（URL、数学文本不受影响，
因为匹配要求冒号紧跟单个盘符字母）。

- [ ] **Step 2: 应用替换（二进制读写，保持换行符与编码）**

```python
def replace(text: str) -> tuple[str, int]:
    count = 0

    def sub(match: re.Match) -> str:
        nonlocal count
        count += 1
        segment = match.group(1)
        if segment is None:
            return "<repo>"
        repl = SEGMENT_MAP.get(segment)
        if repl is None:                      # 未登记的子路径（如 .codex 下的深层文件）
            repl = segment
        return repl

    return PREFIX_RE.sub(sub, text), count
```

注意：分隔符由原文本保留（`\` 形态与 `/` 形态各自保留），避免在代码块里破坏转义。
两个双反斜杠转义文件（`TPA/tests/test_portable_paths.py`、
`TPA/tests/test_metric_k_unification.py`）由 P4 手工处理。

- [ ] **Step 3: 复扫**

Run: `..\.venv\Scripts\python.exe -m unittest tests.test_no_absolute_paths -v`
Expected: 仅剩 P4 的三类特殊命中（程序安装目录、外部盘符示例、测试字面断言）

---

## P4 特殊命中手工修复

**Files:**
- Modify: `.codex/skills/knowledge-point-visualization/scripts/verify_outputs.py`
- Modify: `TPA/tests/test_portable_paths.py`
- Modify: `TPA/tests/test_metric_k_unification.py`（docstring 命令示例）
- Modify: `docs/superpowers/plans/2026-09-16-pre-scaleup.md`（外部盘符示例）

- [ ] **Step 1: `verify_outputs.py` 浏览器探测去绝对路径**

```python
def _program_files_candidates() -> list[str]:
    """从环境变量推导浏览器安装位置（禁止硬编码盘符）。"""
    out: list[str] = []
    for var in ("ProgramFiles(x86)", "ProgramFiles", "ProgramW6432"):
        base = os.environ.get(var)
        if not base:
            continue
        out += [
            os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
            os.path.join(base, "Google", "Chrome", "Application", "chrome.exe"),
        ]
    return out


DEFAULT_BROWSERS = [
    os.environ.get("KPV_BROWSER", ""),
    *_program_files_candidates(),
    *[p for p in (shutil.which(n) for n in
                  ("google-chrome", "chromium", "msedge", "chrome")) if p],
]
```

（新增 `import shutil`；保持原有"显式路径优先、否则按顺序回退"语义）

- [ ] **Step 2: `test_portable_paths.py` 断言去字面盘符**

```python
ABS_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
...
self.assertNotRegex(str(p).lower(), ABS_RE)
```

（文件头 docstring 里的历史事故描述改为"本机仓库根前缀"这类无绝对路径表述）

- [ ] **Step 3: 外部盘符示例改为仓库内相对示例**

`docs/superpowers/plans/2026-09-16-pre-scaleup.md` 里"外部盘符占位"形式的测试样例
改为 `outputs/<tag>/raw/...` 形式；断言相应改为 `as_posix()` 相对路径比较。

- [ ] **Step 4: 命令示例统一**

`TPA/tests/test_metric_k_unification.py` 的 docstring 改为
`..\.venv\Scripts\python.exe -m unittest ...`（工作目录 `TPA`）。

---

## P5 复扫与全量回归

- [ ] **Step 1: 守护测试**

Run: `..\.venv\Scripts\python.exe -m unittest tests.test_no_absolute_paths -v`
Expected: PASS

- [ ] **Step 2: `git grep` 复查零命中**

```powershell
git grep -n -E "(^|[^A-Za-z0-9])[A-Za-z]:[\\/]" -- . ":(exclude)*.js" ":(exclude)*.pdf"
```

Expected: 无输出

- [ ] **Step 3: 全量回归**

Run: `..\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v`
Run: `..\.venv\Scripts\python.exe -m unittest discover -s pre/tests -t . -v`
Expected: 全绿

- [ ] **Step 4: 提交**

```bash
git add AGENTS.md TPA/tests/test_no_absolute_paths.py TPA/tests/test_portable_paths.py TPA/tests/test_metric_k_unification.py .codex/skills/knowledge-point-visualization/scripts/verify_outputs.py docs/superpowers/specs/2026-09-17-portable-paths-design.md docs/superpowers/plans/2026-09-17-portable-paths.md
# 其余机械替换文件按 git diff --name-only 精确补齐
git commit -m "chore(tpa): 入库文件去绝对路径并新增可迁移性守护测试"
```

---

## 计划自查

**1. Spec 覆盖**

| spec 条款 | 落到任务 |
|---|---|
| R1 零绝对路径 | P1 守护测试 + P3 机械替换 + P4 手工修复 |
| R2 只引用可迁移位置 | P2 规则 + P3 映射表 + P4 占位符 |
| R3 命令可跨机复制 | P2 规则 + P4 Step 4 |
| R4 校验可证伪 | P1 反向用例 |
| §6 验收 1–5 | P5 |

**2. 占位符扫描**：无 TBD/TODO；脚本与断言给出可直接粘贴实现。

**3. 命名一致性**：`find_violations`、`iter_tracked_text_files`、`_program_files_candidates`
在测试与实现中同名同签名。

**4. 已知边界**：不改 `.gitignore` 白名单策略；跳过二进制与未跟踪文件；
两个含双反斜杠转义的文件手工处理以免破坏转义。
