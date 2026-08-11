# MinerU 默认输出路径调整实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 mineru 提取默认输出路径改为 `papers/<源文件名>/`（md 与原文件同目录），并把当前 PGD 提取产物按新约定归位。

**Architecture:** 只改两处技能副本中的一条默认输出规则 + 移动两个文件，不改代码、不改历史文档、不清理旧目录。规则文本与 paper-pipeline 的 `{paper_name}/` 约定对齐。

**Tech Stack:** Markdown（技能文件）；PowerShell（目录与文件操作）；验证用 rg / Get-ChildItem。

## Global Constraints

- 只改两处文件中的默认输出规则一条：`G:\Idea\.codex\skills\mineru-document-extractor\SKILL.md` 与 `G:\Idea\.claude\skills\mineru-document-extractor-0.1.29\SKILL.md`。
- 技能文件位于 .codex/、.claude/（gitignored），不提交；提交的只有 spec/plan 文档。
- 不更新历史 KPV spec/plan 文档；不清理 `MinerU-Skill/PGD_07b060/` 其余文件。
- 文件移动限定在 `G:\Idea\papers\` 与 `G:\Idea\MinerU-Skill\` 内，操作前验证路径存在。

---

### Task 1: 更新两处技能的默认输出规则

**Files:**
- Modify: `G:\Idea\.codex\skills\mineru-document-extractor\SKILL.md`
- Modify: `G:\Idea\.claude\skills\mineru-document-extractor-0.1.29\SKILL.md`

**Interfaces:**
- Consumes: spec 第 2.1 节的旧/新规则文本
- Produces: 两处 Agent rules 中一致的默认输出规则（`papers/<name>/`，md + 原文件同目录）

- [ ] **Step 1: 替换 .codex 副本的旧规则**

将 `G:\Idea\.codex\skills\mineru-document-extractor\SKILL.md` Agent rules 中的

```text
- When user does NOT specify `-o`, generate output directory: `~/MinerU-Skill/<name>_<hash>/` where `<hash>` = first 6 chars of MD5 of the source path
```

整行替换为：

```text
- When user does NOT specify `-o`, default to the project `papers/` directory (e.g. `g:/Idea/papers/`): create a folder named after the source file (without extension), and save the extracted Markdown together with the original source file (e.g. the PDF) in that folder — matching the paper-pipeline `{paper_name}/` convention. Run `mkdir -p papers/<name>/` before extraction, then pass `-o papers/<name>/`.
```

- [ ] **Step 2: 替换 .claude 副本的旧规则**

对 `G:\Idea\.claude\skills\mineru-document-extractor-0.1.29\SKILL.md` 执行与
Step 1 完全相同的替换（旧规则文本一致）。

- [ ] **Step 3: 验证旧规则已消失、新规则已生效**

Run:
```powershell
rg -n "MinerU-Skill|<name>_<hash>" G:\Idea\.codex\skills\mineru-document-extractor\SKILL.md G:\Idea\.claude\skills\mineru-document-extractor-0.1.29\SKILL.md
rg -n "default to the project" G:\Idea\.codex\skills\mineru-document-extractor\SKILL.md G:\Idea\.claude\skills\mineru-document-extractor-0.1.29\SKILL.md
```

Expected: 第一条无输出（旧规则已清除）；第二条两处各命中 1 行。

- [ ] **Step 4: 提交（仅文档）**

技能文件 gitignored，无需提交；此任务无 git 操作。

---

### Task 2: PGD 产物归位

**Files:**
- Create: `G:\Idea\papers\PGD\`
- Move: `G:\Idea\papers\PGD.pdf` → `G:\Idea\papers\PGD\PGD.pdf`
- Move: `G:\Idea\MinerU-Skill\PGD_07b060\PGD.md` → `G:\Idea\papers\PGD\PGD.md`

**Interfaces:**
- Consumes: spec 第 2.2 节
- Produces: `papers/PGD/` 下 PGD.md + PGD.pdf 齐全；旧目录其余文件保留

- [ ] **Step 1: 前置检查（移动目标验证）**

Run:
```powershell
Test-Path -LiteralPath 'G:\Idea\papers\PGD.pdf'
Test-Path -LiteralPath 'G:\Idea\MinerU-Skill\PGD_07b060\PGD.md'
```

Expected: 均为 True。

- [ ] **Step 2: 创建目录并移动文件**

```powershell
New-Item -ItemType Directory -Path 'G:\Idea\papers\PGD' -Force | Out-Null
Move-Item -LiteralPath 'G:\Idea\papers\PGD.pdf' -Destination 'G:\Idea\papers\PGD\PGD.pdf'
Move-Item -LiteralPath 'G:\Idea\MinerU-Skill\PGD_07b060\PGD.md' -Destination 'G:\Idea\papers\PGD\PGD.md'
```

- [ ] **Step 3: 验证目录结构**

Run:
```powershell
Get-ChildItem -LiteralPath 'G:\Idea\papers\PGD' -Force | Select-Object Name, Length
Get-ChildItem -LiteralPath 'G:\Idea\MinerU-Skill\PGD_07b060' -Force | Select-Object -First 10 Name
```

Expected: `papers/PGD/` 下只有 PGD.md 与 PGD.pdf；旧目录中 PGD.md 已不在，其余文件（all.json、html、mathjax 等）仍在。

---

### Task 3: 收尾验证

**Files:**
- 无（只读验证 + 提交 plan 文档）

- [ ] **Step 1: 技能规则理解探针（writing-skills GREEN 验证）**

基线失败已在本会话观察到（输出落在 `MinerU-Skill/PGD_07b060/`，即旧规则生效）。
绿色验证：派一个全新子代理，只读 `G:\Idea\.codex\skills\mineru-document-extractor\SKILL.md`，
回答"用户要求把 `g:/Idea/papers/PGD.pdf` 提取为 Markdown 且未指定输出路径，输出与
原 PDF 应分别落在哪里"，预期回答 `G:\Idea\papers\PGD\PGD.md` 与
`G:\Idea\papers\PGD\PGD.pdf`（原文件随约定移入同目录）。

- [ ] **Step 2: 最终检查**

Run:
```powershell
rg -n "default to the project" G:\Idea\.codex\skills\mineru-document-extractor\SKILL.md G:\Idea\.claude\skills\mineru-document-extractor-0.1.29\SKILL.md
Get-ChildItem -LiteralPath 'G:\Idea\papers\PGD' -Force | Select-Object Name
git -C G:\Idea status --short
```

Expected: 两处规则命中；papers/PGD/ 含 PGD.md + PGD.pdf；git 工作区除既有
`TPA/attacks/bandwagon/config.yaml` 修改外无其他变更。

- [ ] **Step 3: 提交 plan 文档**

```powershell
git -C G:\Idea add -- docs/superpowers/plans/2026-08-11-mineru-default-output-path.md
git -C G:\Idea commit -m "docs: MinerU 默认输出路径调整实施计划"
```
