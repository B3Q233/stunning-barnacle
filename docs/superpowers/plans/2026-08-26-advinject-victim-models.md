# AdvInject 阶段一 Victim Models Implementation Plan

> **For agentic workers:** Use the paper-code-implementation attack/model rules and execute each task with a fresh verification checkpoint.

**Goal:** 在 TPA 现有模型架构中实现并注册 AdvInject 对应的 victim models，并改进 WMF 的可微训练能力。

**Architecture:** 每个 victim model 是 `TPA/models/{name}/` 下的独立模块；公共 registry 只保存导入路径和模型配置。共享数据与指标协议通过现有训练框架和最小公共工具复用，攻击模块本阶段不创建。

**Tech Stack:** Python 3.12、PyTorch 2.5、NumPy、SciPy、stdlib unittest、仓库根 `.venv`。

## Global Constraints

- 不创建新虚拟环境，不修改 requirements.txt，不修改无关 attack。
- 只使用 `TPA/models/{name}/`、`TPA/models/registry.py`、相关 `TPA/tests/test_*.py`、设计/计划文档。
- 每个模型必须有 `config.yaml`、`dataset.py`、`model.py`、`train.py`、`main.py`、`docs/USAGE.md`、`docs/DESIGN.md`。
- 所有模型构造兼容 `(cfg, num_users, num_items, edge_index=None)`。
- `eval_step` 和训练指标必须是标量；动态用户索引必须有边界检查。

### Task 1: 冻结公共数据与评估契约

**Files:**
- Create: `TPA/models/revisit_common.py`
- Create: `TPA/tests/test_revisit_common.py`

- [ ] 实现 meta.pkl 读取、train/test pair dataset、用户历史构建、全量 Top-K mask 与统一指标辅助函数。
- [ ] 用内存 meta fixture 验证 shape、边界和指标手算值。

### Task 2: 实现 ItemCF victim

**Files:**
- Create: `TPA/models/itemcf/config.yaml`
- Create: `TPA/models/itemcf/dataset.py`
- Create: `TPA/models/itemcf/model.py`
- Create: `TPA/models/itemcf/train.py`
- Create: `TPA/models/itemcf/main.py`
- Create: `TPA/models/itemcf/docs/USAGE.md`
- Create: `TPA/models/itemcf/docs/DESIGN.md`
- Modify: `TPA/models/registry.py`
- Test: `TPA/tests/test_itemcf_model.py`

- [ ] 先验证 item-item 相似度、评分和 Top-K 排名。
- [ ] 再验证模型构造、注册和小数据 fit/validate。

### Task 3: 实现 ItemAE victim

**Files:**
- Create: `TPA/models/itemae/...`（同 ItemCF 文件集合）
- Modify: `TPA/models/registry.py`
- Test: `TPA/tests/test_itemae_model.py`

- [ ] 先验证输入/输出 shape 和重构 loss。
- [ ] 再验证单 batch 梯度更新、eval 标量和全量评分。

### Task 4: 实现 NCF victim

**Files:**
- Create: `TPA/models/ncf/...`
- Modify: `TPA/models/registry.py`
- Test: `TPA/tests/test_ncf_model.py`

- [ ] 验证 GMF/MLP 前向 shape、BPR loss、参数更新和动态用户边界。

### Task 5: 实现 MultVAE victim

**Files:**
- Create: `TPA/models/multvae/...`
- Modify: `TPA/models/registry.py`
- Test: `TPA/tests/test_multvae_model.py`

- [ ] 验证重构 logits、KL 标量、训练更新和 deterministic eval。

### Task 6: 实现 CML victim

**Files:**
- Create: `TPA/models/cml/...`
- Modify: `TPA/models/registry.py`
- Test: `TPA/tests/test_cml_model.py`

- [ ] 验证距离评分、margin loss、训练更新和全量评分。

### Task 7: 修正 WMF

**Files:**
- Modify: `TPA/models/wmf/config.yaml`
- Modify: `TPA/models/wmf/model.py`
- Modify: `TPA/models/wmf/dataset.py`
- Modify: `TPA/models/wmf/train.py`
- Modify: `TPA/models/wmf/docs/USAGE.md`
- Modify: `TPA/models/wmf/docs/DESIGN.md`
- Modify: `TPA/tests/test_wmf_model.py`

- [ ] 保留 ALS 闭式路径。
- [ ] 新增 SGD 可微路径并验证梯度、动态用户边界和 full ranking。
- [ ] 运行现有 WMF 测试后再进入整合。

### Task 8: 注册表、文档与全量验证

**Files:**
- Modify: `TPA/models/registry.py`
- Modify: `TPA/tests/test_model_registry.py`
- Create/Modify: 各模型 docs

- [ ] 注册七个模型并验证配置加载。
- [ ] 运行相关测试与全量 unittest。
- [ ] 检查 git diff --check、git status、禁止产物未跟踪。