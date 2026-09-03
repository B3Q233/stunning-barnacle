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
- 配置规范：复现/新增模型与攻击时，config.yaml 必须参照
  `TPA/docs/config-template.unified.yaml` 的 canonical 键组织；禁止新增
  同义键；历史别名仅由 `training/config_utils.py` 的兼容层接受，新文件
  不得使用。新复现模型/攻击如需新增配置项，必须同步更新该模板
  （canonical 键与别名映射），模板未覆盖的新键不允许合入。
- 注释完整性（复现所有代码的强制要求）：配置文件与代码必须提供完整注释，
  注释需涵盖：① 为什么这样做（设计动机/取舍）；② 功能是什么（该配置项或
  代码段的作用）；③ 参考的公式是什么（如有，需给出公式编号或原文/代码出处）；
  ④ 使用举例（典型取值或调用方式）。当代码涉及复杂逻辑与矩阵运算时，除上述
  四点外，必须给出矩阵的变换过程（输入形状 → 中间张量形状 → 输出形状、
  行列含义）与对应逻辑说明，禁止只写无解释的矩阵操作。注释默认中文，来源
  标注沿用仓库约定（[paper]/[ai]/[unreported]/[官方代码]）。
- 输出与产物规范：每个训练/生成 epoch 必须输出耗时（控制台
  `[Epoch i/N结束 耗时X分Y秒]`，history 记 `epoch_seconds`），`[eval]` 行、
  history.json、eval_log.csv 输出该轮全部评测指标且不硬编码指标名单；
  新增指标须在 `evaluation/metrics_registry.py` 注册并由配置
  `evaluation.metrics` 启用后自动同步；history.json 统一为
  `{history: [...], best: {...}}`。新复现模型/攻击必须遵守。
- 测试：stdlib unittest，测试文件放 `TPA/tests/test_*.py`；运行命令
  `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_* -v`；改动必须
  运行相关测试，交付前全量回归通过。
- 只改与任务相关的文件，保留他人的改动。

## 4. 数据与产物卫生（必须）

- 数据集允许入库：原始数据 `data/{implicit|explicit}/raw/`（隐式交互 /
  显式评分分目录）与预处理产物 `models/*/data/processed/` 随代码提交，
  保证克隆后可复现；新增/更新数据集时同步提交。
- 以下内容一律不入库（.gitignore 已定义，禁止 `git add -f` 绕过）：
  `attacks/*/data/`（poisoned / rec_freq 等实验产物）、`outputs/`、`checkpoints/`、
  `*.pt / *.pth / *.png / *.log`、`.venv/`、`tmp/`、`papers/`、`MinerU-Skill/`、
  `.claude/`、`.codex/`。
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

## 6. 攻击模块与批量攻击复现硬性模板（2026-09-02 新增，必须）

适用于：在 `TPA/attacks/` 下新增/复现任何投毒攻击（含经典基线、白盒/黑盒
优化攻击、代理/surrogate 攻击与显式评分攻击）。现状基线：random / bandwagon
（纯数据层画像）、pgd（投影梯度，白盒+代理引擎）、tpa（路径注入，白盒/代理
双模式）、advinject（surrogate 优化，尚未接入批量）；下述规则对所有新增模块
一视同仁，不允许为某个攻击另起一套目录/配置/接口结构。

### 6.1 目录与入口约定

- 每个攻击一个目录 `attacks/{name}/`，至少包含：`config.yaml`（唯一配置入口）、
  `registry.py`、`classify.py`、`generate.py`、`fit.py`、`evaluate.py`、
  `run.py`、`docs/USAGE.md` 与 `docs/DESIGN.md`；额外阶段文件（如
  `path_builder.py`、`train_surrogate.py`）按需追加，但必须保留上述固定入口。
- `run.py` 统一 CLI：`--config <路径>`、`--mode <classify|data|model|both|all>`、
  `--tag <run_tag>`；攻击若有自定义阶段（如 tpa 的 `paths`）须在 `USAGE.md`
  中说明，且 `all` 语义必须可串联完整攻击闭环。
- `classify.py / generate.py / fit.py` 必须各自暴露 `main(config) -> dict`，
  供单攻击 run.py 与批量调度器统一调用；`registry.py` 只做薄壳，模型解析
  一律转发 `models/registry.py` 公有注册表。

### 6.2 config.yaml 统一结构

- 顶层固定键：`dataset / mode / seed / k / run_tag / model.name /
  classification / attack / warm_start / training / evaluation / output`；
  攻击特有超参数一律放进 `attack.*`（禁止在顶层散落）。
- 目标选择统一为 `attack.target_items.strategy ∈ specified | category |
  coldest | random`；`ids` 用于固定目标，`category` 依赖 classify 产物。
- 评估 K 由顶层 `k` 绑定；`evaluation.metrics` 必须含 `target_ndcg@{k}: upper`
  / `target_hr@{k}: upper` 等 target_* 攻击选优指标，`recall@{k} / ndcg@{k}`
  仅作投毒代价参考。
- run_tag 优先级：`--tag` > `config.run_tag` > 当前时间；数据与输出按
  `{dataset}/{model}/{tag}/` 分层，每个实验目录保存本次 config 快照。
- config 结构以 `TPA/docs/config-template.unified.yaml` 为唯一参考模板
  （canonical 键与别名映射见模板文末映射表）。

### 6.3 数据契约与产物卫生

- 输入 meta 契约：`meta.pkl` 至少含 `num_users / num_items / train_pairs /
  test_pairs / user_items`；显式评分攻击另见 6.6。
- 攻击内部产物目录（一律不入库）：`data/rec_freq/{dataset}/{model}_top{k}.json`
  （可选分类缓存）、`data/poisoned/{dataset}/{model}/{tag}/`、`data/paths/`（如
  有）。
- `generate` 阶段统一产出 `meta.pkl + profiles.json + stats.json`；`fit` 只读
  generate 产物，不自行改数据。
- 仅依赖数据的攻击（random/bandwagon 类）`generate` 不得 import 模型代码；
  需要权重的攻击（pgd/tpa/advinject 类）必须显式声明读取干净模型或代理
  checkpoint，并在 DESIGN/USAGE 中注明白盒/黑盒假设。

### 6.4 评估协议

- 攻击效果：对“训练集未交互目标物品”的用户报告目标物品 HR@K、NDCG@K、
  命中人数、平均排名；clean 与 poisoned 统一用干净训练集过滤，口径一致。
- warm-start：用户嵌入行不变、物品嵌入行整体偏移 n_fake、假用户行随机初始化；
  无 `embedding` 属性的模型（如纯 ALS）不得强行 warm-start，须降级并告警。
- checkpoint 选优：按 `target_ndcg@K` 主指标 / `target_hr@K` 副指标选优，
  `checkpoint_mode` 支持 per_metric；每 epoch 指标写入 `history.json`。

### 6.5 批量攻击适配（接入 attacks/batch 的硬条件）

- 两层架构：原子攻击（classify→generate→fit 闭环）与 Batch（配置生成 +
  调度 + 汇总，不实现算法）职责分离。
- 注册：在 `attacks/batch/registry.py` 调用
  `register(name, config_path, classify=…main, generate=…main, fit=…main)`；
  未注册的攻击不得声称“支持批量跑”。
- 配置四层继承（高→低）：Generator 运行时字段（target_items / run_tag /
  output.dir）> `override` > Batch 配置 > 攻击默认 `config.yaml`；Batch 配置
  只写与攻击默认不同的差异项。
- 产物归位：fit 产物先入 staging，再整理到
  `output/{batch_tag}/runs/{攻击}_{数据集}_{模型}_top{k}/{层}/item{id}/`；
  汇总产物固定 `results.csv / summary.md / meta.json / tier_stats.json`。
- 新增攻击接入前自查：① config 可被 Generator 展开成单目标原子配置；
  ② classify/generate/fit 均为 `main(config)`；③ generate 产物路径可被调度器
  归位；④ fit 输出指标与 aggregate 解析兼容（不满足则先在 batch DESIGN 里
  登记缺口，禁止静默跳过）。

### 6.6 显式评分攻击/模型的预留槽位（接入 Leg-UP 等评分型攻击时启用）

- 触发条件：攻击生成离散显式评分（如 1–5 星）构造假档案，或 victim /
  surrogate 以显式评分（而非隐式 0/1 交互）训练。接入此类攻击前必须完成下述
  预留项，否则视为未完成适配。
- 数据要求：数据管线须提供评分三元组 `(user_id, item_id, rating)`（或
  `meta.pkl` 增加 `ratings` 字段：稀疏矩阵/三元组），并在配置中声明
  `rating_scale`（如 5）与缺失语义（0 = 未评分）。
- 配置预留块（键名示例，具体值按论文实现时填写，禁止另起结构）：
  ```yaml
  attack:
    explicit_rating:
      scale: 5                 # 评分上界
      target_rating: 5         # push 时目标物品固定评分（可配 learn）
      profile: {}              # 档案级字段：如 filler 数量/取值模式/每用户阈值
  surrogate:
    enabled: true              # 是否需要可微 surrogate 重训
    name: <模型注册名>
    epochs: <int>              # surrogate 每轮重训轮数
    lr: <float>
    l2: <float>                # 或 weight_decay
    batch_size: <int>
    hidden_dims: []            # AE 类 surrogate 隐层（如 [64]）
    unroll_steps: 1            # 有限 unroll；0=直接优化
  ```
- 模型需求：victim / surrogate 在 `models/registry.py` 登记外，还必须满足
  显式评分接口：对 `(user, item)` 输出评分预测的接口、按评分计算损失的训练
  方式（RMSE / 交叉熵 / 加权 MSE）、以及显式评分下的攻击评估（target
  HR@K/NDCG@K 或 RMSE 报告）；无 `embedding` 的模型注明不可 warm-start。
- 注入契约：`profiles.json` 每个假用户记录 `(item, rating)` 列表而非只有
  物品集合；`stats.json` 记录评分分布统计；fake uid 仍从真实用户数之后连续
  编号。
- 模板占位：任何显式评分攻击实现前，须先在对应 config.yaml 与 DESIGN.md
  中按上述键位列出字段来源等级（论文明确/A 级代码/推断），并逐条核对
  6.1–6.5 约定（目录、注册、批量、评估），不允许显式评分成为脱离统一
  模板的理由。

## 7. 建议条目（待确认，尚未升级为硬规范）

- 分支与隔离：特性开发用独立分支或 git worktree（using-git-worktrees），
  大改动合并前走 requesting-code-review。
- 文档与提交信息保持中文（现状已基本一致，是否升级为硬规范待确认）。
- 攻击/模型实现遵循 paper-code-implementation 模板的
  TrainableModel / DatasetProtocol / Experiment / Trainer 松耦合结构。
- 引入 CI / 一键验证脚本（当前仓库无 CI）。

## 8. 参考链接

- [README.md](README.md)
- [TPA 目录](TPA/)
- [specs](docs/superpowers/specs/) / [plans](docs/superpowers/plans/)
- 各模块用法：`TPA/attacks/*/docs/USAGE.md`、`TPA/models/*/docs/USAGE.md`
