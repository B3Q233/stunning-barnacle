---
name: reproduce-debug
description: 论文代码复现调试——对比两份"相同"代码（跨框架/跨版本/跨实现），系统定位导致结果差异的 root cause
---

# 论文代码复现调试

## 何时使用

当用户遇到以下场景时触发本 skill：
- 自己复现的代码运行结果与论文官方代码不匹配
- 同一份代码在不同环境/框架版本下结果不同
- PyTorch 实现与 TensorFlow/JAX 官方实现结果不一致
- 相同网络、相同数据、相同超参数，但评估指标明显偏差

## 核心原则

1. **Reference 是 ground truth**。论文官方代码/原始实现是参照系，复现代码是待排查对象。如果两边都是复现、没有权威 reference，标注为"实现选择差异"而非 bug
2. **按优先级排查**。越靠前的环节越容易被忽略且影响越大：数据加载 → 模型初始化 → Forward → Loss → Optimizer → Backward → 训练循环
3. **非侵入式诊断**。不修改用户原始代码，生成独立的插桩脚本，用户分别在两边环境运行后收集 dump 文件，回到 skill 对比
4. **分层推进**。默认从 L2（张量级）开始，定位到偏差层后对该层自动升级到 L3（算子级），L4（梯度级）作为用户可选深度
5. **数值比对而非文本比对**。核心是比较运行时 tensor 的数值分布，而不是盯着源代码逐行 diff

---

## 工作流程总览

```
用户提供: 两份代码路径 + 运行日志 + 评估结果差异描述
            │
            ▼
┌─────────────────────────────────────────┐
│ Phase 1: 静态对比 (Static Analysis)      │
│  - 文件结构 & config 差异                │
│  - 关键函数签名 & 逻辑流 diff            │
│  - 算子对标知识库匹配                    │
│  - 输出: 结构化 Diff 报告                │
│  - 耗时: 1-2 轮对话                      │
└──────────────┬──────────────────────────┘
               │ 用户确认继续 / 发现差异已修复
               ▼
┌─────────────────────────────────────────┐
│ Phase 2: 动态验证 (Dynamic Verification) │
│  - 生成非侵入式插桩脚本                  │
│  - 数据加载 → 初始化 → Forward 逐层 hook │
│  - 定位数值偏差的起始层和传播路径         │
│  - 输出: 数值偏差 Checklist + 热力图      │
│  - 耗时: 用户运行脚本 + 1-2 轮对话       │
└──────────────┬──────────────────────────┘
               │ 对偏差层深入 / 用户选择 root cause 分析
               ▼
┌─────────────────────────────────────────┐
│ Phase 3: 根因分析 (Root Cause Analysis)  │
│  - 对问题层开启 L3 算子级追踪            │
│  - L4 梯度对比 (可选)                    │
│  - 交叉验证: 修改后验证修复               │
│  - 输出: Root Cause 报告 + 修复方案      │
│  - 耗时: 用户运行脚本 + 1-2 轮对话       │
└─────────────────────────────────────────┘
```

每个 Phase 结束后询问用户：问题是否已定位？是否继续深入？

---

## Phase 1: 静态对比

### 1.1 收集信息

首先向用户确认以下信息（如果未提供）：

1. **两边的代码路径**（哪个是 reference？哪个是 target？）
2. **两边的框架和版本**（PyTorch x.x / TF x.x / JAX x.x；CUDA 版本；关键依赖版本）
3. **两边的运行日志**（loss curve、最终指标、中间检查点的指标）
4. **超参数配置文件**（YAML / argparse / hydra config）
5. **数据集**（是否完全相同？预处理是否相同？）

如果用户无法提供某些信息，标记为 UNKNOWN 并在后续阶段重点排查。

### 1.2 L0: 文件结构与配置对比

逐项对比以下内容，生成 Diff 表：

| 对比项 | 具体检查点 | 工具方法 |
|--------|-----------|----------|
| 文件结构 | `model.py`, `train.py`, `data_loader.py` 等的对应关系 | `diff -rq` 或 Glob |
| 配置文件 | learning_rate, batch_size, weight_decay, epochs, scheduler 参数 | Read + Diff |
| 依赖版本 | requirements.txt / environment.yml / pyproject.toml | Read + Diff |
| 随机种子 | global seed, Python hash seed, numpy seed, torch seed, cudnn seed, DataLoader worker seed | Grep `seed\|manual_seed\|deterministic` |
| 数据预处理参数 | resize, crop, normalize mean/std, augmentation 参数 | Read data loader |
| 模型超参数 | num_layers, hidden_dim, dropout, activation, norm type | Read model config |

特别注意：
- **BatchNorm momentum**：PyTorch 默认 `momentum=0.1`（EMA: new = 0.1×batch + 0.9×old），TF 默认 `momentum=0.99`（EMA: new = 0.99×old + 0.01×batch）——**语义相反**
- **数据格式**：PyTorch 默认 NCHW，TF 默认 NHWC
- **Conv2d padding**：PyTorch 是四周对称 padding，TF 的 padding 语义不同
- **Scheduler step timing**：PyTorch 中 `scheduler.step()` 在 `optimizer.step()` 之前还是之后？

### 1.3 L1: 关键函数对比

对以下关键函数逐对对比：

```
Data Pipeline:
  - __getitem__ / data loading
  - transform / augmentation 顺序
  - normalization 参数
  - batching / collate_fn

Model:
  - __init__: 所有层的定义顺序和参数
  - forward: 数据流图（手动绘制或分析计算图）
  - 自定义模块的默认参数值

Training:
  - loss function 实现
  - optimizer 配置（betas, eps, weight_decay 的应用方式）
  - scheduler 配置和 step 时机
  - train/eval mode 切换
  - gradient clipping
  - EMA (exponential moving average)
```

对每个函数，输出：
- 签名差异（参数名、默认值）
- 逻辑差异（控制流、操作顺序）
- 引用这个差异可能产生的影响

### 1.4 算子对标知识库匹配

搜索代码中所有框架特定 API 调用，使用以下知识库进行对标检查：

#### 内置知识库：PyTorch ↔ TensorFlow/Keras 常见差异

| 操作 | PyTorch | TensorFlow/Keras | 已知差异 |
|------|---------|------------------|----------|
| Conv2d | `nn.Conv2d(in, out, k, s, p)` | `tf.keras.layers.Conv2D(filters, k, s, padding)` | padding 语义不同；weight 初始化分布不同 |
| BatchNorm | `nn.BatchNorm2d(num_features, momentum=0.1, eps=1e-5)` | `tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)` | momentum 方向相反；eps 默认值不同（1e-5 vs 1e-3） |
| LayerNorm | `nn.LayerNorm(normalized_shape, eps=1e-5)` | `tf.keras.layers.LayerNormalization(epsilon=1e-3)` | eps 默认值不同 |
| Dropout | `nn.Dropout(p=0.5)` 训练时自动启用 | `tf.keras.layers.Dropout(rate=0.5)` | 行为基本一致，但注意 eval mode 切换 |
| MaxPool | `nn.MaxPool2d(k, s, p, ceil_mode=False)` | `tf.keras.layers.MaxPool2D(pool_size, strides, padding)` | `ceil_mode` 默认值不同 |
| AvgPool | `nn.AvgPool2d(k, s, p, count_include_pad=True)` | `tf.keras.layers.AveragePooling2D(pool_size, strides, padding)` | `count_include_pad` 影响 padding 位置的计算 |
| Interpolate/Upsample | `F.interpolate(x, scale_factor, mode, align_corners=False)` | `tf.image.resize(x, size, method)` | `align_corners` 行为不同；TF 默认 half_pixel |
| Linear/Dense | `nn.Linear(in, out, bias=True)` | `tf.keras.layers.Dense(units, use_bias=True)` | 基本一致，注意 weight 初始化 |
| Embedding | `nn.Embedding(num, dim, padding_idx=None)` | `tf.keras.layers.Embedding(input_dim, output_dim)` | PyTorch 支持 `padding_idx` |
| CrossEntropyLoss | `nn.CrossEntropyLoss()` 内置 softmax | `tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)` | PyTorch 默认 input 为 logits；TF 需要显式 `from_logits=True` |
| BCEWithLogitsLoss | `nn.BCEWithLogitsLoss()` | `tf.keras.losses.BinaryCrossentropy(from_logits=True)` | 同上 |
| Adam | `torch.optim.Adam(lr, betas=(0.9,0.999), eps=1e-8, weight_decay=0)` | `tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7)` | eps 默认值不同（1e-8 vs 1e-7）；weight_decay 实现方式不同 |
| AdamW | `torch.optim.AdamW(..., weight_decay=0.01)` | `tf.keras.optimizers.AdamW(..., weight_decay=0.004)` | weight_decay 在 PyTorch 中解耦应用，TF 中可能直接加在 loss 上 |
| SGD + Momentum | `torch.optim.SGD(lr, momentum=0.9, dampening=0)` | `tf.keras.optimizers.SGD(learning_rate, momentum=0.0)` | TF SGD 默认无 momentum |
| Xavier/Glorot init | `nn.init.xavier_uniform_` 默认 gain=1 | `tf.keras.initializers.GlorotUniform` | 基本一致 |
| Kaiming/He init | `nn.init.kaiming_uniform_(a=0, mode='fan_in')` | `tf.keras.initializers.HeUniform` | `a=0` 即 ReLU；`a=1` 即 LeakyReLU |

#### 动态补充规则

对于知识库未覆盖的算子，执行以下步骤：
1. 使用搜索工具搜索 `"[算子名] PyTorch vs TensorFlow difference default parameters"`
2. 打开两边的官方文档页面读取
3. 提取默认参数差异，补充到本次对比报告中
4. 将新发现的差异记录到项目的 `.codex/skills/reproduce-debug/knowledge/custom-mappings.md` 中

### 1.5 Phase 1 输出格式

生成 `REPRODUCE_DEBUG_PHASE1.md`：

```markdown
# 复现调试报告 — Phase 1: 静态对比

## 环境信息
| 维度 | Reference | Target |
|------|-----------|--------|
| 框架版本 | PyTorch 2.1.0 | PyTorch 2.4.0 |
| ... | ... | ... |

## L0: 配置差异
| 参数 | Reference | Target | 风险等级 | 说明 |
|------|-----------|--------|----------|------|
| learning_rate | 1e-3 | 1e-3 | ✅ 一致 | |
| weight_decay | 1e-4 | 0.0 | 🔴 高 | Target 未设置 weight_decay |
| ... | ... | ... | ... | ... |

## L1: 关键函数差异
### Data Pipeline
[逐函数 diff，省略一致部分]

### Model Forward
[逐函数 diff，省略一致部分]

## 算子对标检查
| 算子 | Reference | Target | 差异类型 | 风险 |
|------|-----------|--------|----------|------|
| BatchNorm momentum | 0.9 (TF) | 0.1 (PyTorch) | 语义相反 | 🔴 高 |
| ... | ... | ... | ... | ... |

## 优先级排查建议
基于以上分析，建议按以下顺序排查：
1. [最高风险差异 #1]
2. [最高风险差异 #2]
3. ...
```

---

## Phase 2: 动态验证

### 2.1 生成插桩脚本

**关键原则**：不修改用户原始代码。生成独立的 Python 脚本，通过框架的 hook 机制注册到模型上。

#### 2.1.1 PyTorch 插桩脚本模板

生成以下脚本文件 `dump_instrument_pytorch.py`，用户放在 reference/target 工程中分别运行：

```python
"""
非侵入式模型插桩脚本 — PyTorch 版本
用法: python dump_instrument_pytorch.py --config config.yaml --output dump/
在 reference 和 target 工程中分别运行，生成中间值 dump 文件用于对比。
"""
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
from collections import OrderedDict
import pickle

# ============================================================
# 配置区：用户根据实际情况修改以下导入和加载逻辑
# ============================================================
# TODO: 用户修改 — 导入你的模型和数据加载器
# from your_model import YourModel
# from your_data import get_dataloader
# ============================================================

class TensorDumper:
    """以统一格式记录每层的输入/输出 tensor 统计信息"""
    def __init__(self, output_dir, label="ref"):
        self.output_dir = output_dir
        self.label = label
        self.records = OrderedDict()
        self.step_counter = 0
        os.makedirs(output_dir, exist_ok=True)

    def hook_fn(self, name, module, input, output):
        """注册到 forward hook 的回调函数"""
        key = f"step{self.step_counter}/{name}"
        inp = input[0] if isinstance(input, tuple) else input

        self.records[key] = {
            "input_shape": tuple(inp.shape),
            "input_dtype": str(inp.dtype),
            "input_stats": self._tensor_stats(inp),
            "output_shape": tuple(output.shape) if isinstance(output, torch.Tensor) else None,
            "output_dtype": str(output.dtype) if isinstance(output, torch.Tensor) else None,
            "output_stats": self._tensor_stats(output) if isinstance(output, torch.Tensor) else None,
            "module_type": type(module).__name__,
        }

    @staticmethod
    def _tensor_stats(t):
        """提取 tensor 的统计特征（不存储原始值以避免文件过大）"""
        if not isinstance(t, torch.Tensor) or t.numel() == 0:
            return None
        t_f32 = t.detach().float()  # 统一转为 float32 计算统计量
        return {
            "mean": float(t_f32.mean().item()),
            "std": float(t_f32.std().item()),
            "min": float(t_f32.min().item()),
            "max": float(t_f32.max().item()),
            "norm_L2": float(t_f32.norm().item()),
            "numel": int(t.numel()),
            # 额外检查: NaN/Inf 数量
            "nan_count": int(torch.isnan(t).sum().item()),
            "inf_count": int(torch.isinf(t).sum().item()),
        }

    def dump(self):
        """将所有记录写入文件"""
        filepath = os.path.join(self.output_dir, f"dump_{self.label}_step{self.step_counter}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(self.records, f)
        print(f"[Dumper] {len(self.records)} records written to {filepath}")
        self.records = OrderedDict()  # 清空以节省内存

    def step(self):
        self.step_counter += 1


def register_hooks(model, dumper, prefix=""):
    """递归注册 forward hook 到模型的所有子模块"""
    hooks = []
    for name, module in model.named_modules():
        if prefix:
            full_name = f"{prefix}.{name}" if name else prefix
        else:
            full_name = name if name else "root"

        # 跳过 container 类型的空层
        if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            continue
        # 跳过最顶层（已在 named_modules 中处理）
        if len(list(module.children())) > 0 and not isinstance(module, (nn.Sequential,)):
            continue

        hook = module.register_forward_hook(
            lambda m, inp, out, n=full_name: dumper.hook_fn(n, m, inp, out)
        )
        hooks.append(hook)
    return hooks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="dump_output", help="输出目录")
    parser.add_argument("--label", type=str, default="ref", help="标签 (ref/target)")
    parser.add_argument("--steps", type=int, default=5, help="记录多少个 step")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # ==== 设置随机种子 ====
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ==== TODO: 用户修改 — 初始化模型和数据 ====
    # model = YourModel()
    # model.eval()  # 只在 eval mode 下记录，消除 dropout/bn 的随机性
    # dataloader = get_dataloader(batch_size=1, shuffle=False)
    # 确保使用固定的输入数据（比如取数据集的前 N 个样本）
    # ==========================================

    dumper = TensorDumper(args.output, args.label)
    hooks = register_hooks(model, dumper)

    print(f"[Instrument] {len(hooks)} hooks registered on model")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= args.steps:
                break
            print(f"[Instrument] Step {i}...")
            # ==== TODO: 用户修改 — 前向传播 ====
            # output = model(batch['input'])
            # =================================
            dumper.step()

    dumper.dump()

    # 清理 hooks
    for h in hooks:
        h.remove()
    print("[Instrument] Done. All hooks removed.")


if __name__ == "__main__":
    main()
```

#### 2.1.2 TensorFlow/Keras 插桩脚本模板

```python
"""
非侵入式模型插桩脚本 — TensorFlow/Keras 版本
用法: python dump_instrument_tf.py --output dump/
"""
import tensorflow as tf
import numpy as np
import os
import argparse
import pickle
from collections import OrderedDict


class TensorDumper(tf.keras.callbacks.Callback):
    """Keras Callback + 手动 hook 结合的 dumper"""
    def __init__(self, output_dir, label="ref"):
        super().__init__()
        self.output_dir = output_dir
        self.label = label
        self.records = OrderedDict()
        self.step_counter = 0
        self.layer_outputs = {}
        os.makedirs(output_dir, exist_ok=True)

    def _build_hook_model(self, model):
        """构建一个中间层输出模型，通过 functional API 暴露所有层输出"""
        layer_outputs = {}
        def make_hook(layer_name):
            def hook(layer_input):
                # layer_input 可能是 tensor 或 list of tensors
                inp = layer_input if isinstance(layer_input, tf.Tensor) else layer_input[0]
                key = f"step{self.step_counter}/{layer_name}"
                self.records[key] = {
                    "input_shape": tuple(inp.shape.as_list()),
                    "input_dtype": str(inp.dtype),
                    "input_stats": self._tensor_stats(inp),
                }
                return layer_input  # 透传，不修改计算图
            return hook

        # 这种方法的局限性是 TF 没有像 PyTorch 那么方便的 hook
        # 替代方案：使用 tf.GradientTape 配合手动调用
        return model

    @staticmethod
    def _tensor_stats(t):
        """提取 tensor 统计特征"""
        if t is None:
            return None
        t_np = t.numpy() if hasattr(t, 'numpy') else np.array(t)
        if t_np.size == 0:
            return None
        t_f32 = t_np.astype(np.float32).ravel()
        finite_vals = t_f32[np.isfinite(t_f32)]
        return {
            "mean": float(np.mean(finite_vals)) if len(finite_vals) > 0 else float('nan'),
            "std": float(np.std(finite_vals)) if len(finite_vals) > 0 else float('nan'),
            "min": float(np.min(finite_vals)) if len(finite_vals) > 0 else float('nan'),
            "max": float(np.max(finite_vals)) if len(finite_vals) > 0 else float('nan'),
            "norm_L2": float(np.linalg.norm(finite_vals)) if len(finite_vals) > 0 else float('nan'),
            "numel": int(t_np.size),
            "nan_count": int(np.sum(~np.isfinite(t_np))),
            "inf_count": int(np.sum(np.isinf(t_np))),
        }

    def dump(self):
        filepath = os.path.join(self.output_dir, f"dump_{self.label}_step{self.step_counter}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(self.records, f)
        print(f"[Dumper] {len(self.records)} records written to {filepath}")
        self.records = OrderedDict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="dump_output")
    parser.add_argument("--label", type=str, default="ref")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # ==== TODO: 用户修改 ====
    # model = YourTFModel()
    # model.load_weights('checkpoint.h5')
    # ...
    # =========================

    dumper = TensorDumper(args.output, args.label)

    # TF 的动态捕获方案：逐层手动前向传播并记录中间值
    for i, batch in enumerate(dataloader):
        if i >= args.steps:
            break
        # dumper.step_counter = i
        # ==== TODO: 用户修改 ====
        # output = model(batch['input'], training=False)
        # =========================
        dumper.step_counter += 1

    dumper.dump()


if __name__ == "__main__":
    main()
```

#### 2.1.3 用户操作指引

生成脚本后，引导用户执行：

```bash
# ==== 在 Reference 环境 ====
cd /path/to/reference/code
# 1. 编辑 dump_instrument_pytorch.py 中的 TODO 部分（导入模型、加载数据）
# 2. 运行
python dump_instrument_pytorch.py --output dump_ref --label ref --steps 5 --seed 42

# ==== 在 Target 环境 ====
cd /path/to/target/code
# 1. 同样编辑 TODO 部分
# 2. 运行
python dump_instrument_pytorch.py --output dump_target --label target --steps 5 --seed 42

# ==== 收集 dump 文件 ====
# 将两边的 dump_ref/ 和 dump_target/ 目录拷贝到同一机器，提供给 Codex
```

### 2.2 对比 dump 文件

用户提供两边的 dump 目录后，执行以下分析：

#### 2.2.1 对比方法

```python
# compare_dumps.py —— 对比两个 dump 目录中的记录
# 你也可以直接在 Python 交互环境中执行以下逻辑

import pickle
import glob
import os
import numpy as np

def load_dumps(dump_dir):
    """加载一个 dump 目录中的所有记录"""
    all_records = {}
    for f in sorted(glob.glob(os.path.join(dump_dir, "dump_step*.pkl"))):
        step_name = os.path.basename(f).replace(".pkl", "")
        with open(f, "rb") as fh:
            all_records[step_name] = pickle.load(fh)
    return all_records

def compare_tensor_stats(ref_stats, tgt_stats, layer_name):
    """比较单个 tensor 的统计特征，返回差异度量"""
    if ref_stats is None and tgt_stats is None:
        return {"status": "both_none", "risk": "skip"}
    if ref_stats is None or tgt_stats is None:
        return {"status": "one_none", "risk": "high"}

    diffs = {}
    for key in ["mean", "std", "min", "max", "norm_L2"]:
        rv = ref_stats.get(key, float('nan'))
        tv = tgt_stats.get(key, float('nan'))
        if np.isfinite(rv) and np.isfinite(tv):
            abs_diff = abs(rv - tv)
            rel_diff = abs_diff / (abs(rv) + 1e-8)  # 相对差异
            diffs[key] = {"abs": abs_diff, "rel": rel_diff}
        else:
            diffs[key] = {"abs": float('nan'), "rel": float('nan')}

    # 综合风险判定
    mean_rel = diffs.get("mean", {}).get("rel", 0)
    if mean_rel < 1e-4:
        risk = "negligible"
    elif mean_rel < 1e-2:
        risk = "low"
    elif mean_rel < 0.1:
        risk = "medium"
    else:
        risk = "high"

    return {"status": "ok", "risk": risk, "diffs": diffs}

def compare_all(ref_dir, tgt_dir):
    ref = load_dumps(ref_dir)
    tgt = load_dumps(tgt_dir)

    findings = []
    for step in sorted(ref.keys()):
        if step not in tgt:
            findings.append({"step": step, "error": "missing in target"})
            continue
        ref_records = ref[step]
        tgt_records = tgt[step]

        # 找到共同层
        common_layers = set(ref_records.keys()) & set(tgt_records.keys())
        only_ref = set(ref_records.keys()) - set(tgt_records.keys())
        only_tgt = set(tgt_records.keys()) - set(ref_records.keys())

        for layer in sorted(common_layers):
            rr = ref_records[layer]
            tr = tgt_records[layer]

            input_cmp = compare_tensor_stats(
                rr.get("input_stats"), tr.get("input_stats"), layer
            )
            output_cmp = compare_tensor_stats(
                rr.get("output_stats"), tr.get("output_stats"), layer
            )

            findings.append({
                "step": step,
                "layer": layer,
                "module_type": rr.get("module_type"),
                "input_comparison": input_cmp,
                "output_comparison": output_cmp,
            })

    return findings
```

#### 2.2.2 偏差传播定位

关键算法：找到**第一个输出差异大于阈值的层**（即偏差的 root cause 层）。

```
对于每个 step：
  for layer in topological_order:
    if input_cmp.risk in ["negligible", "low"] and output_cmp.risk in ["medium", "high"]:
      → 该层是可疑的偏差产生点 ⚠️
    elif input_cmp.risk in ["medium", "high"] and output_cmp.risk in ["high"]:
      → 该层放大了上游偏差，但上游才是 root cause
```

### 2.3 Phase 2 输出格式

生成 `REPRODUCE_DEBUG_PHASE2.md`：

```markdown
# 复现调试报告 — Phase 2: 动态验证

## 对比范围
- Reference dump: dump_ref/ (5 steps, 127 layers)
- Target dump: dump_target/ (5 steps, 127 layers)
- 共同层数: 125, 仅 Reference: 2, 仅 Target: 2

## 偏差传播热力图 (Step 0)
```
Input → [Conv1: ✅] → [BN1: ⚠️ shift=0.03] → [ReLU: ✅] → [Conv2: 🔴 shift=0.89] → ...
```
(用 emoji 可视化每层的偏差级别)
  ✅ negligible (< 1e-4)
  ⚠️ low (1e-4 ~ 1e-2)
  🟡 medium (1e-2 ~ 0.1)
  🔴 high (> 0.1)

## 偏差起始点
| Step | 层名 | 模块类型 | 输入偏差 | 输出偏差 | 判断 |
|------|------|----------|----------|----------|------|
| 0 | model.encoder.conv1 | Conv2d | ✅ 0.000001 | 🟡 0.023 | ⚠️ 偏差产生点 |
| 0 | model.encoder.bn1 | BatchNorm2d | 🟡 0.023 | 🔴 0.891 | 🔴 严重放大 |
| ... | ... | ... | ... | ... | ... |

## 下一步建议
- 对 `model.encoder.conv1` 和 `model.encoder.bn1` 开启 Phase 3 算子级追踪
- 优先检查：Conv2d 的 weight 初始化、BN 的 momentum 和 eps
```

---

## Phase 3: 根因分析

### 3.1 L3: 算子级追踪

对 Phase 2 定位到的可疑层，生成更细粒度的插桩脚本。

#### 3.1.1 拆解原子操作

对于 PyTorch，使用 `torch.jit.trace` 或 `functorch` 拆解每个 Module 的原子操作：

```python
# 对可疑层进行算子级拆解
import torch
import torch.fx as fx

def trace_layer(model, sample_input):
    """用 FX 将模型拆解为原子操作图"""
    gm = fx.symbolic_trace(model)
    gm.graph.print_tabular()
    return gm

# 对比每个原子操作的输入/输出
# 例如 Conv2d 内部: unfold → multiply → sum → add_bias
```

对于跨框架对比，将两边的算子降低到**数学原语层面**：
- `Conv2d` → `im2col + matmul + reshape + add_bias`
- `BatchNorm` → `(x - mean) / sqrt(var + eps) * gamma + beta`
- `LayerNorm` → 同上但 axis 不同
- `MultiHeadAttention` → `3×Linear + split_heads + QK^T/√d + softmax + PV + concat_heads + Linear`

### 3.2 L4: 梯度级对比 (用户可选)

```python
# 梯度 hook 模板
def register_gradient_hooks(model, dumper):
    hooks = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Hook 参数梯度
            def grad_hook(grad, n=name):
                dumper.record_grad(n, grad)
                return grad
            param.register_hook(grad_hook)

            # Hook optimizer state (需要在 optimizer.step() 前后记录)
            dumper.record_param(name, param.data.clone())

    return hooks
```

梯度级别的对比项：
| 对比项 | 说明 |
|--------|------|
| grad 的 L2 norm | 检查梯度爆炸/消失 |
| grad 的 sign 分布 | 正负比例是否一致 |
| optimizer state | Adam 的 first_moment/second_moment 初始值 |
| param update 量 | `new_param - old_param` 的 magnitude |

### 3.3 交叉验证

定位到 root cause 后，不直接建议修复，而是先验证：

1. **假设陈述**：`[层名] 的 [参数/操作] 因为 [原因] 导致偏差`
2. **最小复现**：生成一个最小脚本，只用该层 + 固定输入验证
3. **单向修改**：只修改 target 的这一处，重新运行 Phase 2，确认偏差消失
4. **如果偏差未消失**：回退修改，标记该假设为"已排除"，继续排查下一个可疑点

### 3.4 Phase 3 输出格式

生成 `REPRODUCE_DEBUG_PHASE3.md`：

```markdown
# 复现调试报告 — Phase 3: 根因分析

## 根因定位

### Root Cause #1: BatchNorm momentum 语义相反
- **位置**: `model.encoder.bn1` (Target: `nn.BatchNorm2d(momentum=0.1)`)
- **Reference**: TensorFlow `BatchNormalization(momentum=0.99)`
  - TF: `running_mean = 0.99 * old + 0.01 * batch_mean`
- **Target**: PyTorch `BatchNorm2d(momentum=0.1)`
  - PyTorch: `running_mean = 0.9 * old + 0.1 * batch_mean`
- **影响**: 在相同 5 步后，running_mean 相差 ~0.89
- **修复**:
  ```python
  # 将 PyTorch 的 momentum 改为
  nn.BatchNorm2d(momentum=0.01)  # 等价于 TF 的 momentum=0.99
  # 或者在训练时手动设置:
  # bn.momentum = 1.0 - tf_momentum
  ```

### Root Cause #2: [下一个...]

## 修复验证
| 修复 | 修复前偏差 | 修复后偏差 | 状态 |
|------|-----------|-----------|------|
| BN momentum 修正 | 🔴 0.891 | ✅ 0.0003 | ✅ 已修复 |
| ... | ... | ... | ... |

## 遗留差异 (不影响结果)
| 层 | 差异 | 原因 | 风险 |
|----|------|------|------|
| ... | ... | 框架实现细节 | 可忽略 |
```

---

## 排查顺序 Checklist

每次对比时，严格按以下顺序排查。越靠前的环节越隐蔽，对最终结果影响越大。

### 1. 数据加载 🔴 最高优先级
- [ ] 数据集是否完全相同（不是同一数据集的不同版本/划分）？
- [ ] DataLoader 的 shuffle seed 是否固定？
- [ ] `num_workers > 0` 是否引入了随机性？（调试时设为 0）
- [ ] transform/augmentation 的**顺序**是否一致？
- [ ] Normalization 的 mean/std 值是否正确？（是否为数据集的统计值而非 ImageNet 默认值？）
- [ ] BGR vs RGB 通道顺序？
- [ ] 输入 value range：[0, 1] vs [0, 255] vs [-1, 1]？
- [ ] Batch size 是否相同？（影响 BN 统计量）

### 2. 模型初始化
- [ ] 所有权重初始化方法是否一致（kaiming/xavier/uniform/normal）？
- [ ] 预训练权重是否从同一个 checkpoint 加载？
- [ ] Bias 初始化是否为 0？
- [ ] BN 的 running_mean/running_var 初始值（0/1 vs 随机）？
- [ ] Embedding 的 padding_idx 处理是否一致？

### 3. Forward Pass
- [ ] Conv 的 padding 是否等价？
- [ ] Pooling 的 ceil_mode 是否一致？
- [ ] Upsample 的 align_corners 是否一致？
- [ ] Dropout 在 eval mode 下是否正确关闭？
- [ ] BN 的 track_running_stats 是否一致？
- [ ] 激活函数的 inplace 操作是否影响梯度计算？

### 4. Loss Function
- [ ] 是否为 logits 输入（而非 softmax 后）？
- [ ] reduction 方式是否一致（mean/sum/none）？
- [ ] label_smoothing 参数是否一致？
- [ ] 类别权重/样本权重是否一致？
- [ ] 混合精度下的 loss scaling 是否正确？

### 5. Optimizer
- [ ] weight_decay 的实现方式？（解耦 vs 直接 L2 正则化）
- [ ] Adam 的 eps 值？（PyTorch: 1e-8, TF: 1e-7）
- [ ] Adam 的 betas 是否一致？
- [ ] Momentum SGD 的 dampening 参数？
- [ ] 逐层学习率设置是否一致？

### 6. Scheduler
- [ ] Scheduler step 的时机（batch-level vs epoch-level）？
- [ ] Warmup 策略是否一致？
- [ ] min_lr 是否设置？

### 7. Backward & Gradient
- [ ] Gradient clipping 的 max_norm 和 norm_type？
- [ ] 是否对某些层 freeze 了梯度？
- [ ] 自定义 backward 的实现是否正确？

### 8. 训练循环
- [ ] train()/eval() 切换时机是否一致？
- [ ] EMA 的 decay 是否一致？
- [ ] 梯度清零的位置（`optimizer.zero_grad()` 在 forward 前还是 loss.backward() 后）？
- [ ] 是否使用了 gradient accumulation？steps 是否一致？
- [ ] 分布式训练的同步方式（DP vs DDP）？

---

## 输出文件约定

每次运行本 skill，在项目根目录生成以下文件：

```
REPRODUCE_DEBUG_PHASE1.md    # Phase 1: 静态对比报告
REPRODUCE_DEBUG_PHASE2.md    # Phase 2: 动态验证报告
REPRODUCE_DEBUG_PHASE3.md    # Phase 3: 根因分析报告
dump_instrument_{pytorch|tf}.py  # Phase 2: 插桩脚本
compare_dumps.py              # Phase 2: dump 对比脚本
```

---

## 使用流程速查

```bash
# 用户触发
"我复现了论文 X 的代码，PyTorch 实现，指标比官方 TF 代码低 5 个点"

# Phase 1: Codex 分析两边的源码 + config
# → 生成 REPRODUCE_DEBUG_PHASE1.md

# 用户确认继续

# Phase 2: Codex 生成插桩脚本
# → 用户在两边的 GPU 环境分别运行
# → 用户提供 dump 目录，Codex 对比分析
# → 生成 REPRODUCE_DEBUG_PHASE2.md

# 用户确认继续

# Phase 3: Codex 对可疑层深度分析
# → 生成算子级追踪脚本（如需要）
# → 交叉验证修复
# → 生成 REPRODUCE_DEBUG_PHASE3.md
```

---

## 交互规范

- **每个 Phase 结束必须询问用户**：是否继续深入？是否已定位问题？
- **生成脚本前必须告知用户**：脚本会做什么、在哪里运行、需要哪些依赖、预期耗时
- **遇到知识库未覆盖的差异**：先使用搜索工具补充信息，不依赖记忆猜测
- **对比结果用 emoji 可视化**：✅ negligible, ⚠️ low, 🟡 medium, 🔴 high
- **所有数值对比用相对差异**（relative difference），不只看绝对差异——batch size 不同时绝对值不可比
