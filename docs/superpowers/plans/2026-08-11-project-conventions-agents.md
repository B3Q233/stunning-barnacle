# 项目默认规范（AGENTS.md）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在仓库根目录创建中文 `AGENTS.md`，以精简硬性规范 + 链接的形式固化
仓库既有约定，供所有 agent 开工时自动读取。

**Architecture:** 单文件交付，无代码改动。文件内容由 spec 第 3 章逐节落地：
项目概述 → 工作流门禁 → 代码与工程规范 → 数据与产物卫生 → Git 与提交规范 →
建议条目 → 参考链接。每条"必须"均以仓库实证为准，"建议"明确标注待确认。

**Tech Stack:** Markdown；验证用 git、PowerShell 只读命令，不新增任何依赖。

## Global Constraints

- 文件位置：仓库根 `G:\Idea\AGENTS.md`（agent 可自动识别的约定文件名）。
- 语言：中文；目标 ≤ 100 行。
- 详略：精简硬性规范 + 链接；有仓库依据的写"必须"，无依据的写"建议（待确认）"。
- 不改动 .gitignore、现有代码、现有文档；不创建 CLAUDE.md / GEMINI.md 副本。
- 提交信息遵循 Conventional Commits（`docs: ...`）。

---

### Task 1: 创建并校验 `G:\Idea\AGENTS.md`

**Files:**
- Create: `G:\Idea\AGENTS.md`
- Test: 无新测试文件；验证步骤见下（git 抽查、目录/文档抽查、行数检查）

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-08-11-project-conventions-agents-design.md`
  （第 3 章结构、第 4 章实证映射、第 7 章范围外）
- Produces: `G:\Idea\AGENTS.md`（后续所有 agent 的入口规范）

- [ ] **Step 1: 创建 AGENTS.md 全文**

按以下内容创建 `G:\Idea\AGENTS.md`：

````markdown
# 项目默认规范（AGENTS.md）

本文件是 G:\Idea 仓库的默认规范，所有 agent 与协作者开工前必读。
仓库文档与提交信息默认使用中文。

## 1. 项目概述

推荐算法论文复现仓库。主代码位于 `TPA/`（attacks / models / training /
evaluation / tests）；论文资料见 `papers/`；流程文档见
`docs/superpowers/`（specs + plans）；技能见 `.codex/skills/` 与
`.claude/skills/`。

## 2. 工作流门禁（必须）

- 论文复现固定链路：paper-pipeline（PDF→Markdown）→ paper-understanding
  （结构化理解文档）→ paper-code-implementation（按模板实现，六步顺序 +
  每步最小验证，不允许跳步）。
- 新功能/变更：brainstorming → 设计文档 → 实施计划 → 实施。设计文档与计划
  先于代码提交；路径固定为
  `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md` 与
  `docs/superpowers/plans/YYYY-MM-DD-<topic>.md`。
- 质量门禁：遇 bug 先 systematic-debugging；写代码前 test-driven-development；
  声称完成前 verification-before-completion；使用任何技能前先读完对应
  SKILL.md，且只在任务匹配时使用。

## 3. 代码与工程规范（必须）

- 环境：使用仓库根 `.venv`（`G:\Idea\.venv\Scripts\python.exe`）；依赖锁定
  在 `requirements.txt`（Python 3.12 + PyTorch 2.5；测试不新增第三方依赖）。
- 目录：`TPA/{attacks, models, training, evaluation, tests}`。每个攻击/模型
  目录配齐 `config.yaml`（唯一配置入口）、`registry.py`、
  `classify.py / generate.py / fit.py / evaluate.py / run.py`、`docs/`
  （USAGE.md + DESIGN.md）。
- 实验隔离：使用 run_tag 机制，数据与输出按 `{dataset}/{model}/{tag}/`
  分层，随实验保存 config.yaml 快照。
- 测试：stdlib unittest，测试文件放 `TPA/tests/test_*.py`；运行命令
  `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_* -v`；改动必须
  运行相关测试，交付前全量回归通过。
- 只改与任务相关的文件，保留他人的改动。

## 4. 数据与产物卫生（必须）

- 以下内容一律不入库（.gitignore 已定义，禁止 `git add -f` 绕过）：
  `data/`、`outputs/`、`checkpoints/`、`*.pkl / *.pt / *.pth / *.png /
  *.log`、`.venv/`、`tmp/`、`papers/`、`MinerU-Skill/`、`.claude/`、
  `.codex/`。
- 中间过程文件放 `tmp/` 或 `.superpowers/sdd/` 会话目录，不入库。

## 5. Git 与提交规范（必须）

- 提交信息：Conventional Commits，`type(scope): 中文描述`；type ∈
  feat / fix / docs / chore / refactor / impl；scope 如 attacks / eval /
  models / tpa / skill / docs。
- 提交粒度：一个逻辑变更一个提交，不混入无关改动；提交前用
  `git status` 与 `git diff --stat` 自查。
- 索引卫生：只用 `git add` 加明确路径；禁止 `git add -f`，禁止不检查就
  `git add -A`。
- 历史安全：禁止对已推送的共享分支 force-push 或改写历史；不提交未验证的
  改动。
- 文档同步：改实现必须同步更新对应 USAGE.md / DESIGN.md；每个任务结束
  跑测试。

## 6. 建议条目（待确认，尚未升级为硬规范）

- 分支与隔离：特性开发用独立分支或 git worktree（using-git-worktrees），
  大改动合并前走 requesting-code-review。
- 文档与提交信息保持中文（现状已基本一致，是否升级为硬规范待确认）。
- 攻击/模型实现遵循 paper-code-implementation 模板的
  TrainableModel / DatasetProtocol / Experiment / Trainer 松耦合结构。
- 引入 CI / 一键验证脚本（当前仓库无 CI）。

## 7. 参考链接

- [README.md](README.md)
- [TPA 目录](TPA/)
- [specs](docs/superpowers/specs/) / [plans](docs/superpowers/plans/)
- 各模块用法：`TPA/attacks/*/docs/USAGE.md`、`TPA/models/*/docs/USAGE.md`
````

- [ ] **Step 2: 事实校验 A（git 提交规范）**

Run:
```powershell
git -C G:\Idea log --pretty=format:'%s' | Select-Object -First 20
```

Expected: 提交信息均为 `type(scope): 中文描述` 格式，type ∈
feat/fix/docs/chore/refactor/impl，scope ∈ attacks/eval/models/tpa/skill/docs
等。若出现格式不符的历史提交，只调整 AGENTS.md 的措辞（如列出常见 type），
不改动历史。

- [ ] **Step 3: 事实校验 B（目录/测试/卫生清单）**

Run:
```powershell
Get-ChildItem G:\Idea\TPA\attacks\pgd -Name
Get-Content G:\Idea\TPA\tests\test_attack_eval.py -TotalCount 6
Get-Content G:\Idea\.gitignore -Raw
Get-Content G:\Idea\TPA\attacks\pgd\docs\USAGE.md -TotalCount 40
```

Expected:
- `attacks/pgd` 下存在 config.yaml、registry.py、classify/generate/fit/
  evaluate/run.py、docs/（六件套 + docs）。
- `tests/test_attack_eval.py` 头部为 `import unittest`（stdlib unittest）。
- `.gitignore` 包含 data/、outputs/、checkpoints/、*.pkl、*.pt、*.png、
  *.log、.venv/、tmp/、papers/、MinerU-Skill/、.claude/、.codex/。
- USAGE.md 包含 run_tag 分层说明（`{dataset}/{model}/{tag}/`）。

- [ ] **Step 4: 行数与格式校验**

Run:
```powershell
(Get-Content G:\Idea\AGENTS.md).Count
```

Expected: 行数 ≤ 100；"必须"与"建议（待确认）"标注齐全；第 7 节链接目标
（README.md、TPA/、docs/superpowers/）均可解析。

- [ ] **Step 5: 提交**

```powershell
git -C G:\Idea add -- AGENTS.md
git -C G:\Idea commit -m "docs: 新增项目默认规范 AGENTS.md"
```
