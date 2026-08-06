# 攻击模板（attack-imp-direct-poison，no-subgoal）

把已验证的 bandwagon 攻击实现提炼为通用攻击模板。**不走六步模型复现顺序**，
而是独立的 classify → data → model 三阶段；适用于复现论文中的攻击/投毒方法
（shilling attack、后门投毒、间接/多跳攻击基线等）。

## 1. 何时使用

- 论文包含攻击/投毒方法，需要实现攻击模块或攻击基线
- 已有 paper-understanding 产出的理解文档（含攻击定义章节）
- 已有六步流程产出的干净模型与数据处理产物

触发词示例："实现攻击基线"、"投毒攻击"、"攻击模板"、"复现论文的攻击方法"、
"实现 bandwagon / 随机 / 平均攻击"。

## 2. no-subgoal 原则与前置条件

攻击模板**不适用**「数据处理→数据导入→模型结构→模型评估→模型训练→结果展示」的
固定顺序，原因：攻击模块的产物依赖已训练好的干净模型，且自身流程是
分类 → 注入 → 投毒训练，不是从原始数据开始的完整模型复现。

前置条件（不满足先停下补齐）：
1. 理解文档已冻结确认（含攻击参数与评估协议）
2. 受害模型已完成六步复现：`models/{model_name}/`（model.py / dataset.py / config.yaml）
   + `training/framework.py` + 已训练 checkpoint
3. 数据处理产物存在：`models/{model_name}/data/processed/{dataset}/meta.pkl`

## 3. 模板结构与使用流程

复制 `assets/attack-imp-direct-poison/` 到项目 `attacks/attack_imp_direct_poison/`
（要改名就全局替换 `attack_imp_direct_poison`）。文件职责：

| 文件 | 职责 |
|------|------|
| `config.yaml` | 唯一配置入口（dataset / model / classification / attack / training / evaluation） |
| `registry.py` | 受害模型注册表（新增模型在此登记） |
| `classify.py` | 第 1 步：推荐频次分类（流行/普通/冷门） |
| `generate.py` | 第 2 步：目标选择 + 假用户画像注入 |
| `fit.py` | 第 3 步：warm-start 投毒训练 + 对比评估 |
| `evaluate.py` | HR@K / NDCG@K / 模型效用报告 |
| `run.py` | classify / data / model / both / all 编排 |
| `docs/USAGE.md` | 生成项目的使用文档 |
| `docs/DESIGN.md` | 生成项目的设计文档（TODO 按论文填写） |

## 4. 三阶段验证门禁（每阶段通过才进入下一阶段）

### classify 阶段
- 输出 `data/rec_freq/{dataset}/{model}_top{k}.json`
- 检查：`summary.appearing_items > 0`、三档数量打印、流行阈值
  （`min_popular_count`）存在且合理；覆盖率随数据集规模合理（冷门通常占多数）

### data 阶段
- 输出 `data/poisoned/{dataset}/meta.pkl + profiles.json + stats.json`
- 硬断言：`train_pairs_after - train_pairs_before == 假用户数 × 画像大小`
- 校验：目标 ID 在 `[0, num_items)` 内；`stats.json` 含 `category` / `rec_count`
- 指定目标（`strategy: specified`）时 `ids` 不允许为空

### model 阶段
- 输出 `outputs/{dataset}/checkpoints/ + history.json + attack_comparison.md`
- warm-start 时断言迁移的用户/物品行数与干净模型一致
- 报告必须含：目标物品 Clean/Poisoned 的 HR@K 与 NDCG@K
- 投毒代价检查：模型效用（recall/ndcg）不得显著下降；
  若不需要可设 `evaluation.report_model_utility: false`

## 5. 与六步模板的关系

- 六步模板管"模型怎么实现"；攻击模板管"模型怎么被打"。
- 目录约定：`{项目}/models/{model_name}/` 由六步流程交付，`{项目}/attacks/{attack_name}/`
  由攻击模板交付；两者通过 `registry.py` 解耦。
- 攻击模板默认画像 = "流行 filler + 指定目标"（bandwagon 语义）；
  换攻击语义只改 `generate.py` 的 `generate_fake_profiles`，其余流程不动。

## 6. 交付清单

- [ ] `attacks/{attack_name}/` 下 9 个模板文件（代码 + 配置 + 文档）已就位且可运行
- [ ] classify / data / model 三阶段各跑通一次，验证门禁全部通过
- [ ] `docs/DESIGN.md` 的 TODO 已按论文填写（攻击定义、参数来源等级）
- [ ] `docs/USAGE.md` 已按实际攻击名/数据集更新
- [ ] 对比报告 `attack_comparison.md` 含 HR@K / NDCG@K 与投毒代价结论
