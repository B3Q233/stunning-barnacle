# 项目默认规范（AGENTS.md）设计文档

- 日期：2026-08-11
- 状态：设计已确认，待实现
- 范围：在仓库根目录 `G:\Idea` 新增 `AGENTS.md`
- 流程技能：using-superpowers → brainstorming → writing-plans

## 1. 背景与目标

`G:\Idea` 仓库目前没有一份统一的"项目默认规范"文件（不存在
AGENTS.md / CLAUDE.md / GEMINI.md，`README.md` 仅为占位）。agent 每次开工
只能靠浏览代码和文档自行推断约定，容易产生不一致。

目标：在仓库根目录新增一份中文 `AGENTS.md`，以"精简硬性规范 + 链接"的形式
固化仓库既有约定，供 Codex / Claude 等 agent 在开工时自动读取。

## 2. 决策记录

| 决策点 | 结论 |
|---|---|
| 覆盖范围 | 整个仓库（根目录 `AGENTS.md`），不只 TPA |
| 语言 | 中文 |
| 详略 | 精简硬性规范 + 链接；有仓库依据的写"必须"，无依据的写"建议（待确认）" |
| 规范来源 | 归纳现状优先：git 历史、docs、代码结构、.gitignore 均有实证 |

## 3. 文件结构（AGENTS.md 章节）

### 3.1 项目概述

- 仓库定位：推荐算法论文复现仓库，主代码在 `TPA/`。
- 链接：`README.md`、`TPA/` 结构、`docs/superpowers/`。

### 3.2 工作流门禁（必须）

- 论文复现固定链路：paper-pipeline（PDF→Markdown）→ paper-understanding
  （结构化理解文档）→ paper-code-implementation（按模板实现，六步顺序 +
  每步最小验证）。
- 新功能/变更：brainstorming → spec → plan → 实施；spec/plan 先于代码提交，
  路径命名 `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md` 与
  `docs/superpowers/plans/YYYY-MM-DD-<topic>.md`。
- 质量门禁：bug 先 systematic-debugging；写代码前 test-driven-development；
  声称完成前 verification-before-completion；使用任何技能前先读对应
  `SKILL.md`。

### 3.3 代码与工程规范（必须）

- 环境：仓库根 `.venv`（`G:\Idea\.venv\Scripts\python.exe`），依赖锁定在
  `requirements.txt`（Python 3.12 + PyTorch 2.5；测试不新增第三方依赖）。
- 目录：`TPA/{attacks, models, training, evaluation, tests}`；每个攻击/模型
  目录配齐 `config.yaml`（唯一配置入口）、`registry.py`、
  `classify/generate/fit/evaluate/run.py`、`docs/`（USAGE.md + DESIGN.md）。
- 实验隔离：run_tag 机制，数据与输出按 `{dataset}/{model}/{tag}/` 分层，
  随实验保存 config.yaml 快照。
- 测试：stdlib unittest；运行命令
  `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_* -v`；改动必须跑
  相关测试，交付前全量回归通过。

### 3.4 数据与产物卫生（必须）

- 不入库清单（.gitignore 已定义，禁止用 `git add -f` 绕过）：
  `data/`、`outputs/`、`checkpoints/`、`*.pkl/*.pt/*.pth/*.png/*.log`、
  `.venv/`、`tmp/`、`papers/`、`MinerU-Skill/`、`.claude/`、`.codex/`。
- 中间过程文件放 `tmp/` 或 `.superpowers/sdd/` 会话目录，不入库。

### 3.5 提交与文档规范（必须）

- Conventional Commits：`type(scope): 中文描述`；type ∈
  feat/fix/docs/chore/refactor/impl；scope 如 attacks/eval/models/tpa/skill/
  docs。
- 改实现必须同步更新 USAGE.md / DESIGN.md；提交按任务拆分，每个任务结束
  跑测试。

### 3.6 建议条目（软性，待确认）

- 攻击/模型实现遵循 paper-code-implementation 模板的
  TrainableModel / DatasetProtocol / Experiment / Trainer 结构（现状 attacks
  使用 classify/data/model 三阶段 + 六件套，是否升级为硬规范待确认）。
- 文档与提交信息保持中文（现状已基本一致，是否升级为硬规范待确认）。
- 是否需要 CI / 一键验证脚本（当前仓库无 CI）。

## 4. 规范依据（实证映射）

| AGENTS.md 条目 | 仓库实证 |
|---|---|
| Conventional Commits | git log：feat/fix/docs/chore/impl + scope（attacks/eval/tpa/lightgcn/kpv 等） |
| unittest 测试 | `TPA/tests/*.py` 使用 unittest；spec/plan 文档写明 stdlib unittest 与运行命令 |
| 六件套结构 | `TPA/attacks/{pgd,tpa,random,bandwagon}` 目录与各自 USAGE.md 项目结构节 |
| run_tag 实验隔离 | `TPA/attacks/pgd/docs/USAGE.md` 第 1.5 节 |
| 不入库清单 | `.gitignore` |
| spec→plan 流程 | `docs/superpowers/specs`、`plans` 现存 3 组文档 |
| .venv / requirements | 仓库根 `.venv`、`requirements.txt`（torch==2.5.1+cu121 等锁定） |
| 论文复现技能链路 | `.codex/skills/paper-pipeline`、`paper-understanding`、`paper-code-implementation` |

## 5. 成功标准

- `G:\Idea\AGENTS.md` 存在，中文，精简（目标 100 行内）。
- 所有"必须"条目均可在仓库中找到实证（见第 4 节映射表）。
- "建议"条目明确标注待确认，不与现状冲突。
- 文件使用 agent 可自动识别的根目录约定文件名（AGENTS.md）。

## 6. 验收方式

- 人工审阅 AGENTS.md，与第 4 节映射表逐条对照。
- 抽查 git log / tests / 目录结构，确认规范描述与现状一致。
- 不要求改动任何现有代码。

## 7. 范围外（不做）

- 不改动 .gitignore、现有代码、现有文档。
- 不引入 CI、lint、格式化工具（仅作为建议条目提出，由用户决定）。
- 不创建 CLAUDE.md / GEMINI.md 副本（用户要求"一个特殊的文件"）。
- 不对仓库执行大规模重命名或重构。
