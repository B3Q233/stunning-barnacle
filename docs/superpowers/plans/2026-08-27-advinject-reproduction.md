# AdvInject 攻击复现实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在现有 TPA 规范内复现 AdvInject 的代理模型投毒流程，并支持对已注册 victim model 进行迁移评估。

**Architecture:** 新增独立 `TPA/attacks/advinject/` 模块，复用现有 `models/registry.py`、数据预处理和训练框架。攻击流程拆分为目标选择、代理训练/伪用户优化、投毒数据保存、victim 重训评估四个边界；不修改目标仓库代码，不创建新虚拟环境。

**Tech Stack:** Python 3.12、PyTorch 2.5、stdlib unittest、YAML、现有 TPA 训练与 run_tag 工具。

## Global Constraints

- 使用仓库根 `.venv`：`<repo>\.venv\Scripts\python.exe`。
- 只修改 AdvInject 相关代码、注册表、测试和同步文档。
- 所有攻击产物写入 `TPA/attacks/advinject/data/` 或 `outputs/`，不入库。
- 通过 `TPA/models/registry.py` 解析 victim/surrogate model，不重复实现 victim model。
- 先写失败测试，再实现代码；每个阶段运行最小验证。
- 区分 surrogate 优化损失与 victim 最终攻击指标。

---

### Task 1: 建立攻击模块契约

**Files:**
- Create: `TPA/attacks/advinject/__init__.py`
- Create: `TPA/attacks/advinject/config.yaml`
- Create: `TPA/attacks/advinject/registry.py`
- Create: `TPA/attacks/advinject/run.py`
- Create: `TPA/tests/test_advinject_contract.py`

- [ ] 定义配置、模式、run_tag、模型解析和 CLI 契约。
- [ ] 为未知模型、非法目标物品和非法伪用户数量添加失败测试。
- [ ] 实现最小公共入口并运行契约测试。

### Task 2: 实现目标选择与伪用户数据生成

**Files:**
- Create: `TPA/attacks/advinject/generate.py`
- Create: `TPA/attacks/advinject/data.py`
- Modify: `TPA/tests/test_advinject_contract.py`

- [ ] 从 clean `meta.pkl` 读取连续 ID 交互。
- [ ] 支持指定目标和按流行度类别选择目标。
- [ ] 生成可梯度优化的伪用户交互矩阵，并保存 `meta.pkl`、`fake_data.npz`、目标记录和配置快照。
- [ ] 运行数据阶段最小验证。

### Task 3: 实现 surrogate 投毒优化

**Files:**
- Create: `TPA/attacks/advinject/surrogate.py`
- Create: `TPA/attacks/advinject/optimize.py`
- Modify: `TPA/tests/test_advinject_contract.py`

- [ ] 支持 `wmf` 的可微 SGD 内层训练和 ItemAE surrogate。
- [ ] 使用有限 unroll 计算正常用户目标物品的多项式交叉熵。
- [ ] 对 fake tensor 求梯度并执行约束更新。
- [ ] 记录 surrogate loss、伪数据变化和每轮历史。
- [ ] 运行单步梯度和 finite-loss 验证。

### Task 4: victim 重训与迁移评估

**Files:**
- Create: `TPA/attacks/advinject/evaluate.py`
- Modify: `TPA/attacks/advinject/run.py`
- Create: `TPA/tests/test_advinject_evaluate.py`

- [ ] 将 fake interactions 拼接到 clean train pairs，按公共 registry 构造 victim。
- [ ] 支持 WMF/MF/LightGCN 及阶段一新增模型的评估入口。
- [ ] 输出 TargetHR@K、TargetNDCG@K、TargetAvgRank 及 clean utility 对比。
- [ ] 区分 clean、surrogate、attacked 三类产物。
- [ ] 运行小型 fixture 的端到端验证。

### Task 5: 文档与回归验证

**Files:**
- Create: `TPA/attacks/advinject/docs/USAGE.md`
- Create: `TPA/attacks/advinject/docs/DESIGN.md`
- Create: `TPA/attacks/advinject/docs/IMPLEMENTATION_DOCS.md`
- Modify: `TPA/attacks/registry.py` if required

- [ ] 记录与 `revisit_adv_rec` 的对应关系和无法确认的实现差异。
- [ ] 说明命令、配置、数据格式、指标解释和故障排查。
- [ ] 运行编译、攻击相关测试和全量测试。
