# pre 扩展约定：接入新的模型 / 攻击

pre 是编排层。"接一个新模型"不等于改 pre 的核心代码，而是**补一条适配声明**。

---

## 一、接入新模型（两个条件）

### 条件 1：模型本身满足仓库的统一契约

在 `models/registry.py` 的 `AVAILABLE_MODELS` 里登记：

```python
"mymodel": {
    "model_cls": "models.mymodel.model:MyModel",
    "dataset_cls": "models.mymodel.dataset:MyDataset",
    "config_path": "models/mymodel/config.yaml",
    "description": "一句话",
},
```

模型类必须提供 pre 依赖的四个接口：

| 接口 | 说明 |
| --- | --- |
| `__init__(cfg, num_users, num_items, edge_index)` | 统一构造签名（不用图的模型忽略 `edge_index`） |
| `train_step(batch)` | 一步训练，返回含 `loss` 的 dict |
| `get_item_embeddings()` | 全量物品嵌入 `(num_items, d)` |
| `get_user_embeddings()` | 全量用户嵌入 `(num_users, d)` |

（前三个是仓库既有约定；`get_user_embeddings` 只有用到时才需要，但建议一并实现。）

### 条件 2：在 pre 侧声明它属于哪个训练循环家族

```python
from pre.runners.model_adapters import ModelAdapter, register_model

register_model(ModelAdapter(
    name="mymodel",
    dataset="models.mymodel.dataset:MyDataset",
    loop="pairwise",                      # pairwise | full_batch
    defaults={"emb_dim": 64, "lr": 0.001, "weight_decay": 1e-4},
    description="MyModel：BPR 三元组训练"))
```

两个循环家族的数据契约：

| 家族 | batch 格式 | Dataset 构造 | 每 epoch 步数 |
| --- | --- | --- | --- |
| `pairwise` | `(users, pos_items, neg_items)`，neg 形状 `[B, neg_ratio]` | `Dataset(pairs, num_items, user_items, num_users, mode="train", neg_ratio)` | 多步（`batch_size` 控制） |
| `full_batch` | `(users, items, conf, p, user_obs, item_obs)` | `Dataset(pairs, alpha, epsilon, scheme)` | 1 步（闭式全量 sweep） |

`defaults` 里的键会写进 `TrainingConfig`，覆盖顺序：

```
pre.training 通用键  →  adapter.defaults  →  model.overrides（用户显式，最高）
```

例：WMF 需要 `factors=100 / alpha=40 / optimizer=als / weight_decay=0.01`，
这些放在 `defaults` 里，编排代码完全不需要知道 WMF 的存在。

### 自检

```bash
python pre/run.py --mode doctor --models lightgcn,mf,wmf,mymodel
```

输出每个模型的：是否在 `models/registry.py` 登记、pre 是否有适配器、
Dataset 是否可加载、模型类是否有三个必需方法、以及问题清单。
适配器缺失时 `get_adapter()` 会抛带**可执行修复代码**的报错，而不是 KeyError。

---

## 二、接入新攻击（一个声明）

攻击模块必须满足仓库既有约定：把中毒数据写到

```
attacks/<name>/data/poisoned{,_proxy}/<dataset>/<model>/<run_tag>/meta.pkl
```

其中 `meta.pkl` 至少含 `num_users / num_items / train_pairs / test_pairs / user_items`，
且假用户的 uid 从 `原 num_users` 起连续编号（pre 依赖这一点统计注入量）。

登记方式：

```python
from pre.runners.run_attack import register_attack

register_attack(
    name="myattack",
    module="attacks.myattack.generate",
    entry="main",              # 入口函数名（默认 main）
    meta_kwarg="raw_meta",     # 传数据的方式，见下表
)
```

| `meta_kwarg` | 调用方式 | 适用 |
| --- | --- | --- |
| `raw_meta` | `entry(config, raw_meta=Path)` | 支持自定义输入 meta 的攻击（random/bandwagon/pgd/tpa/uba） |
| `data_path` | `entry(config)`，数据路径写在 `config["data_path"]` | 通过 config 取数据的攻击（advinject） |

可选 `pre_stage="attacks.myattack.stage"` 声明前置阶段（每次攻击前先调用一次）。

**pre 能接上的前提**：攻击必须能作用在**任意给定的 meta** 上。
如果攻击把数据路径写死（例如 TPA 的 `path_builder` 固定读干净数据），
pre 无法把它指向退化数据——这类攻击目前只能记录为已知限制，不能混入矩阵。

---

## 三、新增一个受害模型后要重跑什么

`pre` 的缓存键是 `(model, item, ratio, attack)`，新增模型只影响它自己的那一条：

```
1. targets.json 不变（目标集合与模型无关）
2. 跑 python pre/run.py --mode original --models mymodel
3. 跑 python pre/run.py --mode degraded  --models mymodel
4. 跑 python pre/run.py --mode attack    --models mymodel
5. 跑 python pre/run.py --mode analyze   --models mymodel   → CSV 自动多出该模型的行
```

已跑过的模型不会被重训（`pre.cache.reuse=true`）。
