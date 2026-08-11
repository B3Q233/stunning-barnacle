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
