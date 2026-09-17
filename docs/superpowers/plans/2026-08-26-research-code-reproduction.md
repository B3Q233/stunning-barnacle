# research-code-reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 创建一个只读分析科研代码仓库并输出完整复现指南的模块化 Skill。

**Architecture:** `SKILL.md` 负责触发、总流程和安全边界；`workflow/` 提供阶段分析方法；`templates/` 提供固定输出骨架；`rules/` 提供证据、命令和不确定性约束；`agents/openai.yaml` 提供 UI 元数据。

**Tech Stack:** Markdown、YAML、Codex Skill 目录规范、`quick_validate.py`。

## Global Constraints

- 只读分析目标科研代码仓库，不修改目标仓库任何文件。
- 不执行训练、推理、批量评估、数据集下载、大文件下载、依赖安装或环境创建。
- 证据优先级固定为：源代码事实 > 配置文件 > 可执行命令验证 > README > 论文 > 推断。
- 无法确认的信息必须使用 `⚠️ 未从源码确认`、`⚠️ 需要用户补充` 或 `推断：` 标记。
- 文档与提交信息默认使用中文。
- Skill 名称只使用小写字母、数字和连字符。

---

### Task 1: 创建 Skill 主入口

**Files:**
- Create: `.codex\skills\research-code-reproduction\SKILL.md`

**Interfaces:**
- Consumes: 用户提供的本地仓库路径、GitHub URL 或压缩包路径。
- Produces: 触发条件、只读约束、分析顺序、模块加载指引和固定输出章节。

- [ ] 写入仅含 `name` 与 `description` 的 YAML frontmatter。
- [ ] 在正文中定义输入、输出、证据优先级、禁止操作和十阶段工作流。
- [ ] 明确按需读取 `workflow/`、`templates/`、`rules/` 文件。
- [ ] 运行 `quick_validate.py`，确认 frontmatter 与命名有效。

### Task 2: 创建工作流参考文件

**Files:**
- Create: `workflow/repository.md`
- Create: `workflow/environment.md`
- Create: `workflow/dataset.md`
- Create: `workflow/weights.md`
- Create: `workflow/config.md`
- Create: `workflow/training.md`
- Create: `workflow/inference.md`
- Create: `workflow/troubleshooting.md`

**Interfaces:**
- Consumes: `SKILL.md` 定义的分析顺序与证据约束。
- Produces: 各阶段的只读检查清单、证据记录格式和输出要点。

- [ ] 每个文件只描述一个分析阶段。
- [ ] 每个阶段包含检查目标、优先检查位置、允许验证方式和常见误判。
- [ ] 训练、推理和排错章节明确禁止高风险执行。
- [ ] 检查所有文件名与 `SKILL.md` 引用一致。

### Task 3: 创建输出模板

**Files:**
- Create: `templates/usage-guide.md`
- Create: `templates/environment.md`
- Create: `templates/dataset.md`
- Create: `templates/troubleshooting.md`

**Interfaces:**
- Consumes: `workflow/` 阶段分析结果。
- Produces: 中文固定章节、证据状态、命令状态和复现清单的输出骨架。

- [ ] `usage-guide.md` 包含十四个固定章节。
- [ ] 各专题模板可独立填充并与主模板章节对应。
- [ ] 命令模板包含来源、前置条件、验证状态和限制字段。
- [ ] 模板不虚构项目名称、命令、路径、指标或结果。

### Task 4: 创建规则文件

**Files:**
- Create: `rules/source-of-truth.md`
- Create: `rules/command-validation.md`
- Create: `rules/uncertainty.md`

**Interfaces:**
- Consumes: 分析过程中发现的源码、配置、文档和命令证据。
- Produces: 冲突裁决、命令分级和不确定性标注规范。

- [ ] 定义六类证据状态及引用要求。
- [ ] 将命令分为允许只读验证、禁止执行和需用户明确授权三类。
- [ ] 定义 `⚠️ 未从源码确认`、`⚠️ 需要用户补充`、`推断：` 的使用场景。
- [ ] 添加防止 README 覆盖源码事实的反例规则。

### Task 5: 创建 Skill 元数据

**Files:**
- Create: `agents/openai.yaml`

**Interfaces:**
- Consumes: `SKILL.md` 的名称、触发条件和输出目标。
- Produces: Skill 列表显示名称、简短描述和默认提示词。

- [ ] 使用 `skill-creator` 的元数据格式。
- [ ] 描述突出科研代码仓库、环境、数据、权重、训练、评估、推理和复现指南。
- [ ] 通过 `generate_openai_yaml.py` 或等价规则检查字段完整性。

### Task 6: 完成验证与只读试运行

**Files:**
- Modify: 无目标仓库文件

**Interfaces:**
- Consumes: 完整 Skill 目录。
- Produces: 校验通过结果和一个本地仓库的静态分析检查记录。

- [ ] 运行 `quick_validate.py .codex\skills\research-code-reproduction`。
- [ ] 检查 Markdown 内部引用的文件均存在。
- [ ] 对 `TPA` 仅执行目录、文本和配置读取类检查。
- [ ] 记录未执行训练、下载、安装和写入操作。
- [ ] 运行 `git diff --check` 与 `git status --short`。