---
name: paper-code-implementation
description: 基于固定的松耦合训练框架模板（TrainableModel / DatasetProtocol / Experiment / Trainer），将已确认的论文理解文档转化为可运行代码，并产出完整的工程化项目（独立虚拟环境、分文件夹结构、配置文件、依赖管理、使用文档）。用于论文复现工作流的代码编写阶段，当用户提到"写复现代码"、"实现这个模型"、"按框架填代码"、"论文转代码"、"骨架填充"，或已有 paper-understanding skill 产出的理解文档并要求继续实现时，必须使用此 skill。核心约束：严格按照"数据处理→数据导入→模型结构→模型评估→模型训练→结果展示"的固定顺序逐段实现，每段完成后必须运行对应的最小验证才能进入下一段，不允许跳过顺序或一次性生成全部代码。即使用户要求"直接给我完整代码"，也要先按此顺序产出并验证每一段，最后再组装、配环境、写文档、整体交付。当论文包含攻击/投毒方法（用户提到"实现攻击基线"、"投毒攻击"、"攻击模板"、"复现论文中的攻击方法"等）时，使用本 skill 内置的 attack-imp-direct-poison 攻击模板：该模板为 no-subgoal，不适用六步固定顺序，走 classify（推荐频次分类）→ data（目标指定+假用户注入）→ model（投毒训练+HR@K/NDCG@K 对比评估）三阶段。
---

 

  # 论文代码实现 Skill

  把"理解文档"转化为可运行代码，核心原则是**用固定框架约束自由度 + 按依赖顺序逐段验证**，而不是一次性生成整份代码。一次性生成的代码即使能跑通，错误也会混在一起无法定位；逐段验证则把每一类错误锁定在最小范围内。

  最终交付的不是一堆散落的脚本，而是一个**可以独立运行、有独立环境、结构清晰、有完整使用文档**的工程项目。代码注释全部使用中文。

  ## 前置条件检查

  1. **理解文档已冻结确认**：如果用户还没有经过 paper-understanding skill 产出并人工确认理解文档，先提示用户完成这一步，不要在没有事实来源的情况下直接写代码。
  2. **确定项目根目录**：向用户确认本次复现的项目根目录路径（例如 `test/model`），后续所有产出——虚拟环境、依赖文件、代码、配置、文档——**全部写入这个目录内部**，不污染用户当前工作目录或系统默认环境。如果用户没有提供，主动询问，不要自己假设一个路径。
  3. **框架模板已加载**：使用 `assets/tool_template.py` 中的框架代码作为骨架基础。这是固定契约，**禁止修改**以下既有结构：
     - `DatasetProtocol` 的五个抽象方法签名（`train_loader` / `val_loader` / `test_loader` / `get_init_params` / `get_dataset`）
     - `TrainableModel` 的三个抽象方法签名（`train_step` / `eval_step` / `build_dataloader`）
     - `TrainingConfig` / `ConfigBuilder` / `Experiment` / `Trainer` / `Callback` 体系的对外接口

     只允许在模板标注 `TODO` 的区域内新增代码，新增类必须实现对应抽象基类。

  ## 项目目录结构（固定规范，必须遵循）

  根目录下**只允许两个文件**：`main.py` 和 `config.yaml`。其余所有代码按功能分文件夹存放。完整结构如下（以根目录为 `test/model` 为例）：

  ```
  test/model/                      ← 项目根目录（用户指定）
  ├── main.py                      ← 唯一入口脚本，支持 --config / --model 参数
  ├── config.yaml                  ← 唯一的多级配置文件
  ├── requirements.txt             ← 依赖清单
  ├── .venv/                       ← 独立虚拟环境（自动创建，依赖自动安装到这里）
  ├── data/
  │   ├── raw/                     ← 原始数据集存放处（提示用户手动下载到此处）
  │   └── processed/                ← ① 数据处理阶段产出的处理后数据
  ├── datasets/
  │   ├── __init__.py
  │   └── paper_dataset.py         ← ② 数据导入：PaperDataset / PaperDataLoader
  ├── models/
  │   ├── __init__.py
  │   └── paper_model.py           ← ③ 模型结构：PaperModel
  ├── evaluation/
  │   ├── __init__.py
  │   └── metrics.py                ← ④ 模型评估：compute_metrics 等
  ├── training/
  │   ├── __init__.py
  │   └── framework.py              ← 框架核心代码（即 tool_template.py 的落地版本，
  │                                     包含 TrainableModel/DatasetProtocol/Experiment/
  │                                     Trainer/Callback 体系，⑤ 模型训练在此接入）
  ├── outputs/
  │   ├── checkpoints/              ← 模型权重
  │   ├── training_curve.png        ← ⑥ 结果展示：训练曲线
  │   └── comparison_table.md       ← ⑥ 结果展示：复现值 vs 论文值对比表
  └── docs/
      ├── IMPLEMENTATION_DOCS.md    ← 六步实现文档
      └── USAGE.md                  ← 使用文档（项目结构/数据集放置/运行方式/配置说明）
  ```

  每个子文件夹只放该阶段对应的代码，不要把不同阶段的逻辑混进同一个文件——这本身也是"最小爆炸半径"原则的延伸：出问题时能通过文件夹定位到是哪个阶段的逻辑出了错。

  ## 虚拟环境与依赖管理（必须执行，不可省略）

  **不要**把依赖安装到用户的系统默认环境或全局 Python 环境。具体步骤：

  1. 在项目根目录内创建独立虚拟环境：
     ```bash
     cd <项目根目录>
     python -m venv .venv
     ```
  2. 后续所有依赖安装、所有验证脚本的运行，都必须在激活这个虚拟环境后执行（Windows: `.venv\Scripts\activate`；Linux/Mac: `source .venv/bin/activate`）。
  3. 实现过程中每用到一个新的第三方库，就追加进根目录的 `requirements.txt`（不要等到最后一次性回忆所有用过的库，容易遗漏）。
  4. 全部六段实现完成后，在虚拟环境中执行一次完整安装验证：
     ```bash
     pip install -r requirements.txt
     ```
     确认无报错，这是交付前的最后一道检查。
     
     ##### 注意：
     
     安装torch时，需要确认设备的cuda版本
     
     

  ## 固定实现顺序（不可跳过、不可并行）

  ```
  ① 数据处理 → ② 数据导入 → ③ 模型结构 → ④ 模型评估 → ⑤ 模型训练 → ⑥ 结果展示
  ```

  每段完成后做对应验证，验证通过才进入下一段；验证失败只排查当前段，不回头怀疑已验证通过的段。每段产出物写入 `IMPLEMENTATION_DOCS.md` 对应章节（模板见 `references/step_doc_template.md`）。

  ### ① 数据处理 → 产出物：`data/processed/`、`docs/IMPLEMENTATION_DOCS.md` 步骤①章节

  - **依据**：理解文档「模块一·1.4 预处理方式」。这一过程只生成实际训练使用的数据，不要和②的数据导入逻辑混在一起
  - **实现内容**：原始数据 → 处理后数据的转换逻辑（归一化、编码、划分等），可作为独立脚本（如 `scripts/preprocess.py`，运行一次产出 `data/processed/` 下的文件）
  - **必须产出的文档字段**：原始数据格式、处理方法表（方法名/参数/输入→输出示例）、输出规格（各 split 的规模格式）、关键决策记录（AI 推断/需人工确认的选择及依据）
  - **验证方式**：抽样打印处理前后对比，人工核对。标注【AI推断补全】的步骤重点核查
  - **数据格式审计（必须执行，高频 bug 来源）**：预处理脚本运行前，**必须**先 `head -5` 原始数据文件，确认真实格式：
    - 每行是 `user_id item1 item2 ...`（一用户多物品，NGCF 格式）还是 `user_id item_id`（成对格式）？
    - **严禁**假设格式——理解文档中记录的格式描述必须与文件实际内容一致
    - 预处理代码的解析逻辑必须与真实格式匹配：如果是多物品格式，必须遍历 `parts[1:]` 而不是只取 `parts[1]`
  - **交互数门禁（必须执行）**：预处理完成后，**必须**打印 `len(train_pairs)` 并与论文报告值（如 Gowalla 应为 ~1,027,370）对比：
    - 偏差 > 5% 则必须报警，排查是数据源问题还是解析问题
    - 这个门禁不过，后续所有步骤都是白跑的——图里没有足够的边，GCN 传播毫无意义

  ### ② 数据导入 → 产出物：`datasets/paper_dataset.py`

  - **依据**：理解文档「模块一·1.1/1.3」+ `DatasetProtocol`。只做导入，不重新做数据处理
  - **实现内容**：`PaperDataset(Dataset)` + `PaperDataLoader(DatasetProtocol)`，实现全部五个抽象方法
  - **必须产出的文档字段**：实现类清单（类名/父类/职责）、batch 格式表（字段/shape/dtype，对照理解文档 2.1）、DataLoader 参数及依据
  - **验证方式（硬性 gate）**：

  ```python
  loader = PaperDataLoader(config).train_loader()
  batch = next(iter(loader))
  assert batch[0].shape == (config.batch_size, ...)  # 对照理解文档 2.1
  print("dtype:", batch[0].dtype)
  ```

  ### ③ 模型结构 → 产出物：`models/paper_model.py`

  - **依据**：理解文档「模块二·2.2/2.3」
  - **实现内容**：`PaperModel(TrainableModel)`，只写 `forward` 和网络层定义，先不接损失/优化
  - **必须产出的文档字段**：逐层结构表（类型/shape/参数量/初始化方式）、架构示意图（文本缩进描述数据流）、关键维度对照表（不同超参下的维度变化）
  - **验证方式**：用②的真实 batch 跑一次 forward，断言输出 shape 与理解文档 2.2 一致，逐层打印 shape 核对
  - **初始化自检（必须执行，常见 bug 来源）**：在验证 forward shape 的同时，**必须**检查各层权重的初始化方式是否与理解文档 2.7 一致：
    - 打印权重的 `mean()`、`std()`、`min()`、`max()`，确认与指定的初始化分布（如 Normal(0, 0.1)）在合理误差内一致
    - 如果文档指定了 Normal(std=0.1) 而你用了 Xavier，均值分布范围可能差 20 倍以上——这会直接影响收敛
    - **严禁**以"标准做法"为由替换论文明确指定的初始化方式

  ### ④ 模型评估 → 产出物：`evaluation/metrics.py`

  - **依据**：理解文档「模块二·2.5」
  - **实现内容**：`compute_metrics()`，严格对齐论文的指标计算方式（负采样策略、过滤规则等协议细节）
  - **必须产出的文档字段**：评估协议表、指标公式（LaTeX+代码逻辑对应）、手工验证用例
  - **验证方式**：用手算好预期结果的小样本验证，不能只用真实数据跑一次"看起来正常"就过关——这类 bug 通常不报错，只会让结果"看起来合理但是错的"

  ### ⑤ 模型训练 → 产出物：`training/framework.py` 中接入的 `train_step`/`eval_step`

  - **依据**：理解文档「模块二·2.4」「模块三」
  - **实现内容**：补全 `train_step`/`eval_step`，接入 `Trainer.run`
  - **必须产出的文档字段**：损失函数公式（含权重系数来源）、优化器配置表、特殊训练策略、1-batch 验证结果
  - **验证方式**：1 个 batch、1 个 epoch，确认 loss 非 NaN/Inf、数值合理、反向传播后参数确实更新
  - **eval_step 标量契约验证（必须执行，高频 bug 来源）**：`eval_step` 的返回值必须全部为标量（`float` / `int` / 0维张量）。**严禁**返回大批量预测张量（如 shape `(B, N)` 的全量排序评分）。训练循环用 `sum(v)/len(v)` 聚合各 batch 指标，如果返回值是张量且各 batch 尺寸不同（尾批），直接 crash：`RuntimeError: The size of tensor a (256) must match the size of tensor b (175)`。如果确实需要全量排序评分（如推荐系统的 HR@K），必须走独立方法（如 `predict_full_ranking()`），在训练循环外单独调用。
    - 验证脚本见 `references/test_snippets.md` ⑤，逐项断言 `isinstance(v, (int, float)) or v.numel()==1`
    - **不等长尾批压测**：用 `drop_last=False` 的 DataLoader 模拟 batch_size 不整除数据集大小的场景，确认所有 batch 的 eval_step 都不 crash
  - **攻击/投毒类论文的动态数据结构验证（必须执行）**：如果论文方法包含注入假用户/假样本等动态数据操作：
    - 从理解文档中确认哪些数据结构是静态的（如 N=物品数）、哪些是动态的（如 M=用户数会递增）
    - 所有按用户/物品索引的容器（如 `_train_matrix[uid]`、Embedding 表）**必须**在访问前做边界检查
    - 步骤⑤验证中必须构造一个 `max(uid) >= original_n_users` 的边界 batch，断言不 crash
  - **公式→API 对照验证（必须执行，高频 bug 来源）**：将理解文档中的每个数学公式与代码实现逐项对照：
    - 🔴 **高频陷阱**：论文写 `λ‖W‖²`（平方 Frobenius 范数），PyTorch 的 `.norm(p=2)` 返回的是 `‖W‖`（未平方）。正确写法是 `.norm(p=2).pow(2)`。检查所有 `.norm()` 调用是否都正确平方。
    - 损失函数的主项和每个正则项分别用一个小数值张量手工验算，确认代码输出与手算一致
    - `F.softplus(x)` = `log(1+exp(x))`，`-F.logsigmoid(x)` = `F.softplus(-x)`——确认语义与论文一致
  - **正则化完整性验证（必须执行）**：对照理解文档 2.6「正则化技术清单」，逐项确认代码中已实现：
    - L2 权重衰减：确认公式正确（尤其注意 `.norm()` vs `.norm().pow(2)`）
    - Message/Graph Dropout：确认在 `forward` 中训练时启用、eval 时自动关闭
    - Early Stopping：确认监控的指标和 patience 值与文档一致
    - 其他论文指定的正则化手段：逐项勾销

  ### ⑥ 结果展示 → 产出物：`outputs/training_curve.png`、`outputs/comparison_table.md`

  - **依据**：理解文档中记录的论文报告指标
  - **实现内容**：训练曲线绘制 + 复现值 vs 论文值对比表
  - **必须产出的文档字段**：输出物清单、论文报告值汇总（注明来源 Figure/Table 编号）、对比结果、缺口分析
  - **验证方式分级**：完全对齐（±2%内）/ 部分对齐（回④⑤排查）/ 未对齐（回②③排查）

  ## 攻击模板（attack-imp-direct-poison，no-subgoal）

  论文包含攻击/投毒方法时，**使用攻击模板，不走上面的六步顺序**。该模板由已验证的
  bandwagon 攻击实现提炼，模板骨架在 `assets/attack-imp-direct-poison/`，使用指南与
  三阶段验证门禁见 `references/attack_imp_direct_poison.md`。

  - **三阶段流程**：classify（加载干净模型 → 全量评分 → 每用户 Top-K → 推荐频次
    分类流行/普通/冷门）→ data（指定目标 + 假用户画像注入）→ model（warm-start
    投毒训练 + HR@K / NDCG@K 对比报告）
  - **前置条件**：理解文档已冻结；受害模型已按六步流程交付
    （`models/{model_name}/` 含 model.py / dataset.py / config.yaml、已训练 checkpoint、
    `data/processed/{dataset}/meta.pkl`）
  - **使用步骤**：复制 `assets/attack-imp-direct-poison/` 到项目
    `attacks/attack_imp_direct_poison/`（要改名则全局替换 `attack_imp_direct_poison`）
    → 编辑 `config.yaml`（dataset / model.name / 目标 ids）→ 按三阶段逐段实现并验证
  - **换攻击语义**（随机/平均/多跳路径画像等）：只改 `generate.py` 的
    `generate_fake_profiles` 一处，其余流程不动
  - **验证门禁**：classify 后核对覆盖率与三档数量；data 后硬断言
    `注入交互数 == 假用户数 × 画像大小`；model 后报告必须含目标物品
    Clean/Poisoned 的 HR@K 与 NDCG@K，并检查投毒代价（模型效用不显著下降）

  ## 配置键名一致性检查（必须执行，常见 bug 来源）

  **已知踩坑模式**：`config.yaml` 中定义的键名（如 `processed_data_path`），如果在多个文件里被独立地用 `config.get(key, default)` 的方式读取，很容易出现某处键名打错或用了旧名字（如 `data_dir`），且因为 `.get()` 自带 fallback，不会报错，只会**默默地落到默认值上**，导致代码在错误的路径里找数据，行为上"看起来正常运行"，但数据是空的或读了别的目录——这类 bug 不报错，难以察觉，往往要等到训练结果不对劲才会被发现，且排查成本很高。

  **根本对策：键名只能有一处定义来源，不允许每个文件各自硬编码 key 字符串。** 具体执行：

  1. 在 `training/framework.py`（即 `TrainingConfig` 所在文件）顶部统一定义一份配置键名常量，例如：
     ```python
     # 配置键名常量：所有读取 config 的地方必须从这里导入，不要在各文件里手写字符串
     KEY_PROCESSED_DATA_PATH = "processed_data_path"
     KEY_RAW_DATA_PATH = "raw_data_path"
     ```
  2. 所有需要读取这些配置的地方（`build_dataloader`、`main.py` 的手动加载逻辑、预处理脚本等）统一 `from training.framework import KEY_PROCESSED_DATA_PATH` 后使用 `config.get(KEY_PROCESSED_DATA_PATH, ...)`，**禁止在多个文件里各自重复写字面字符串 `"processed_data_path"`**——字面字符串重复出现两次以上，就是这类 bug 的温床。
  3. 完成⑤模型训练阶段后，新增一项强制验证：**搜索全项目中所有 `config.get(` 调用，列出用到的 key 字符串，确认它们都来自常量定义，且没有出现"同一概念两个不同拼写"的情况**（如 `data_dir` 和 `processed_data_path` 同时存在）。这一步写入 `IMPLEMENTATION_DOCS.md` 步骤⑤章节的"关键决策记录"中，作为交付前的检查项之一。
  4. 如果模型有多个变体（如本框架的 GMF/MLP/NeuMF 多个 `build_dataloader` 实现），这些变体之间的 config 读取逻辑必须完全一致——多个模型类重复实现同一段读取逻辑本身就是风险点，优先考虑把"从 config 解析出 dataloader 所需参数"抽成一个共享函数，让所有模型变体调用同一份，而不是各自复制粘贴一遍再分别维护。

  ## 参考代码层级与使用规范（必须遵守）

  复现时可能接触到多个版本的代码，但**不是所有代码的可信度都一样**。按可信度分三级：

  | 层级 | 定义 | 可信度 |
  |------|------|--------|
  | **A 级** | 论文作者本人发布的官方代码 | 最高 |
  | **B 级** | 论文明确写出（公式、表格、文字） | 最高 |
  | **C 级** | 第三方实现、社区复现 | 仅参考 |

  **使用规则**：
  1. **A 级和 B 级冲突时，以 B 级（论文原文）为准**，但必须标注差异
  2. **C 级代码不能作为修复基准**。做任何修改前必须先回论文核实
  3. **严禁**在 C 级代码和论文冲突时不加标注地跟 C 级走

  ## 与官方代码的差异处理

  若理解文档记录了与 A 级/B 级代码的差异点，完成⑥后逐项检查当前实现选择的是论文还是官方代码的做法，交付时明确告知用户该选择及理由。

  ## 超参数配置文件 `config.yaml`（多级结构，必须产出）

  在完成步骤⑤之后、步骤⑥之前生成。这是用户唯一需要直接编辑的文件，必须做到**不改代码、只改配置**就能调整复现行为。结构分四个一级模块，每个一级模块下是具体参数（二级），模板见 `references/config_template.yaml`：

  ```yaml
  data:           # 对应①②阶段，数据路径与划分方式
  model:          # 对应③阶段，模型结构超参数
  training:       # 对应⑤阶段，训练相关超参数
  evaluation:     # 对应④阶段，评估协议参数
  ```

  每一项参数必须满足：
  - **来源标注**：行尾注释标记 `# [paper]`（论文明确写出，直接用论文值）/ `# [ai]`（论文未给出但可推断，注释里写清推断依据）/ `# [unreported]`（论文完全未提及，用社区/框架默认值）。不允许把 AI 推断值标成 `[paper]`
  - **取值范围说明**：在 `docs/USAGE.md` 的配置说明章节，每个参数要写清楚"可以改成什么值、改了会影响什么、不能改成什么值"（例如 `model.embedding_dim` 可以改成任意正整数，但改变后需要重新跑①②确认 shape；`training.optimizer` 只能是 `adam`/`sgd`，因为 `train_step` 里只实现了这两种的分支逻辑）

  `main.py` 必须支持：

  ```bash
  python main.py --config config.yaml --model gmf
  ```

  ## 使用文档 `docs/USAGE.md`（必须产出，不可用 IMPLEMENTATION_DOCS.md 代替）

  这份文档面向"第一次拿到这个项目、不了解实现细节"的使用者（包括用户自己半年后再回来看）。必须包含以下章节，写法可以参考 `references/usage_doc_template.md`：

  1. **项目结构说明**：对照上面的目录树，逐文件夹一句话说明用途
  2. **环境准备**：虚拟环境创建与激活命令、依赖安装命令
  3. **数据集准备**：数据集去哪里下载（官方链接）、下载后放在 `data/raw/` 下的具体哪个子路径、目录结构示例
  4. **复现完整流程**（按真实操作顺序，不是按代码实现顺序）：
     ```
     ① 下载数据集 → 放入 data/raw/
     ② 在 config.yaml 的 data 模块中配置 raw_data_path
     ③ 运行预处理脚本，生成 data/processed/
     ④ 在 config.yaml 中配置 processed_data_path
     ⑤ 运行 python main.py --config config.yaml 开始训练
     ⑥ 训练完成后查看 outputs/ 下的曲线图和对比表
     ```
  5. **配置文件详解**：按 `data`/`model`/`training`/`evaluation` 四个模块，逐参数说明含义、默认值、可调范围、调整后的影响、来源标注（[paper]/[ai]/[unreported]）的含义
  6. **常见问题**：结合实现过程中实际遇到的坑写 2-4 条（如"显存不足怎么调 batch_size"“loss 变 NaN 先检查哪里”），不要写空泛的套话，要结合本次复现的具体模型

  ## 交付要求

  ### 逐段交付
  每完成一段，向用户报告：本段实现了什么 / 验证结果 / 是否可进入下一段。验证未通过先停下排查，不要往后走。

  ### 最终交付物清单（全部完成后逐项核对）

  - [ ] 按目录结构规范创建的全部文件夹与代码文件
  - [ ] 根目录 `main.py`（支持 `--config`/`--model`）、`config.yaml`（多级、每项标注来源等级）
  - [ ] `requirements.txt`，且已在项目根目录的 `.venv` 虚拟环境中完整安装验证通过
  - [ ] `.venv/` 虚拟环境已创建在项目根目录内，不是用户系统默认环境
  - [ ] `docs/IMPLEMENTATION_DOCS.md`：六步实现文档，每步含方法说明、输入→输出示例、验证结果、关键决策记录
  - [ ] `docs/USAGE.md`：项目结构、环境准备、数据集准备、复现流程、配置详解、常见问题
  - [ ] `outputs/training_curve.png`、`outputs/comparison_table.md`
  - [ ] 配置键名一致性已检查：全项目搜索过 `config.get(`/`config[`，所有 key 均来自统一常量定义，无重复硬编码、无同概念多拼写
  - [ ] **【新增】数值型超参数全量对照**：对照理解文档，逐一检查 `config.yaml` 中每个数值是否与论文一致（learning_rate、weight_decay、dropout rate、init std、embedding_dim 等）。特别注意：禁止用"框架默认值"或"业界惯例"替代论文指定的值——即使论文值"看起来不常规"也必须先照做，复现不对再排查。
  - [ ] **【新增】公式→API 对照已执行**：所有数学公式（尤其含 `‖·‖²`、`log`、`exp` 的项）已与 PyTorch API 语义逐项核对，确认 `norm()` 已平方、`logsigmoid` 符号正确、正则项公式与论文一致
  - [ ] **【新增】eval_step 标量契约已验证**：`eval_step` 所有返回值已断言为标量（`float/int/0-dim tensor`），已用 `drop_last=False` 的 DataLoader 做过不等长尾批压测
  - [ ] **【新增】动态数据结构已验证**（仅攻击/投毒/对抗类论文需要）：所有按 uid/iid 索引的容器已加边界检查，已构造 `uid >= original_n_users` 的边界 batch 压测通过

  ### 方法文档不可省略

  `IMPLEMENTATION_DOCS.md` 每一步必须写清楚：用了什么方法（方法层面，不是代码逐行翻译）、输入→输出的具体示例、为什么这样做（标注依据来自论文哪个位置）。

  ## 参考文件

  - `assets/tool_template.py` — 固定框架模板，所有代码必须基于此文件扩展，落地时放入 `training/framework.py`
  - `assets/attack-imp-direct-poison/` — 攻击模板骨架（no-subgoal，由已验证的 bandwagon 实现提炼）
  - `references/attack_imp_direct_poison.md` — 攻击模板使用指南：三阶段流程、验证门禁、交付清单
  - `references/test_snippets.md` — 每个阶段的验证代码片段模板
  - `references/step_doc_template.md` — 六步实现文档模板
  - `references/config_template.yaml` — 多级超参数 YAML 模板，含来源标注格式示例
  - `references/usage_doc_template.md` — 使用文档模板
  - `references/project_structure.md` — 目录结构规范的纯文本版本，新建项目时直接对照创建文件夹
  - `references/pytorch_api_pitfalls.md` — PyTorch API 常见陷阱（数学符号 → 代码对照表），步骤⑤「公式→API 对照验证」时使用
  - `references/general_pitfalls.md` — 一般性实现陷阱（数据格式假设、参考代码层级、交互数验证），步骤①②⑤时使用
