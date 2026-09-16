# -*- coding: utf-8 -*-
"""模型适配器注册表：让 pre 能接入**任意**受害模型，而不改编排核心。

为什么需要这层（设计动机）：
    pre 的编排逻辑（选目标 → 造退化 → 训练 → 调攻击 → 提嵌入 → 算距离）对所有模型
    都一样，但各模型的"数据契约"和"训练循环"不同：
      * pairwise 家族（LightGCN / MF，BPR）：batch = (users, pos_items, neg_items)，
        Dataset(pairs, num_items, user_items, num_users, mode, neg_ratio)，每 epoch 多步。
      * full_batch 家族（WMF，ALS 全量闭式）：batch = (users, items, conf, p,
        user_obs, item_obs)，Dataset(pairs, alpha, epsilon, scheme)，每 epoch 一步。
    把差异收敛成一张注册表后，新增模型 = 新增一条 adapter，不改 common/pipeline。

扩展约定（新增一个模型只需两步）：
    1) 该模型必须已在 models/registry.py 的 AVAILABLE_MODELS 登记
       （提供 model_cls / dataset_cls / config_path），且实现
       `get_item_embeddings()`、`get_user_embeddings()`、`train_step(batch)`、
       `model_cls(cfg, num_users, num_items, edge_index)` 构造签名。
    2) 在 pre 侧调用 register_model(ModelAdapter(...)) 登记它属于哪个循环家族；
       若家族还能复用，直接在 REGISTRY 里加一行即可。

用法示例：
    from pre.runners.model_adapters import register_model, ModelAdapter, get_adapter
    register_model(ModelAdapter(
        name="multvae", dataset="models.multvae.dataset:MultVAEDataset",
        loop="pairwise", description="MultVAE（若其 batch 为 (u,pos,neg)）"))
    ad = get_adapter("wmf")          # -> ModelAdapter(loop='full_batch', ...)

注意：注册表**不负责实现模型**，只声明如何构造数据与调用训练。
适配器缺失时 pre 会给出可执行的报错（见 get_adapter），而不是抛 KeyError。
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from models.registry import AVAILABLE_MODELS


@dataclass(frozen=True)
class ModelAdapter:
    """一个受害模型在 pre 中的适配声明。

    name        : 模型名（必须与 models/registry.py 的键一致）
    dataset     : Dataset 类的 import 路径 "pkg.module:Class"
    loop        : 训练循环家族：pairwise（BPR 三元组）| full_batch（全量单批）
    defaults    : 该模型的结构性默认超参（会写进 TrainingConfig，用户 overrides 优先）
    description : 一句话说明，便于 available_models() 打印
    """

    name: str
    dataset: str
    loop: str
    defaults: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""


REGISTRY: Dict[str, ModelAdapter] = {}


def register_model(adapter: ModelAdapter, *, overwrite: bool = False) -> None:
    """登记一个模型适配器；重名默认报错，避免静默覆盖。"""
    if adapter.loop not in ("pairwise", "full_batch"):
        raise ValueError(f"未知训练循环家族 {adapter.loop!r}，"
                         f"可选 pairwise | full_batch")
    if adapter.name in REGISTRY and not overwrite:
        raise ValueError(
            f"模型 {adapter.name!r} 已有适配器；如需替换请传 overwrite=True")
    REGISTRY[adapter.name] = adapter


def get_adapter(name: str) -> ModelAdapter:
    """取模型适配器；缺失时给出**可执行**的扩展指引。"""
    if name in REGISTRY:
        return REGISTRY[name]
    registered = ", ".join(sorted(REGISTRY))
    in_repo = name in AVAILABLE_MODELS
    hint = ("该模型已在 models/registry.py 登记，只差 pre 侧的适配器"
            if in_repo else
            "该模型**尚未**在 models/registry.py 登记，请先登记模型再补适配器")
    raise KeyError(
        f"pre 还没有模型 {name!r} 的适配器（{hint}）。\n"
        f"已支持：{registered}\n"
        f"扩展方式见 pre/docs/EXTENDING.md：\n"
        f"  register_model(ModelAdapter(name={name!r}, "
        f"dataset='models.{name}.dataset:<DatasetClass>', loop='pairwise'))")


def available_models() -> List[str]:
    return sorted(REGISTRY)


def load_dataset_cls(adapter: ModelAdapter):
    """按适配器声明加载 Dataset 类（局部导入，避免 import 期循环依赖）。"""
    module_name, _, attr = adapter.dataset.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def check_scaffold(models: List[str]) -> List[Dict[str, Any]]:
    """扩展自检：逐个模型报告"能否被 pre 驱动"。

    检查项：① models/registry.py 是否登记；② pre 是否有适配器；
    ③ 模型类是否具备 pre 依赖的三个接口（构造签名 / 取嵌入 / train_step）。
    返回诊断列表，供 CLI `--mode doctor` 与测试使用。
    """
    report: List[Dict[str, Any]] = []
    for name in models:
        item: Dict[str, Any] = {"model": name, "in_registry": name in AVAILABLE_MODELS,
                                "has_adapter": name in REGISTRY, "issues": []}
        if item["has_adapter"]:
            ad = REGISTRY[name]
            item["loop"] = ad.loop
            try:
                cls = load_dataset_cls(ad)
                item["dataset_cls"] = f"{cls.__module__}.{cls.__name__}"
            except Exception as exc:                      # noqa: BLE001
                item["issues"].append(f"Dataset 加载失败: {exc}")
        try:
            entry = AVAILABLE_MODELS[name]
            module_name, _, attr = entry["model_cls"].partition(":")
            model_cls = getattr(importlib.import_module(module_name), attr)
            for meth in ("get_item_embeddings", "get_user_embeddings", "train_step"):
                if not hasattr(model_cls, meth):
                    item["issues"].append(f"模型类缺少 {meth}()")
        except Exception as exc:                          # noqa: BLE001
            item["issues"].append(f"模型类加载失败: {exc}")
        item["ok"] = not item["issues"] and item["has_adapter"]
        report.append(item)
    return report


# ── 内置适配器（与仓库现有模型一一对应）────────────────────────────────────────
register_model(ModelAdapter(
    name="lightgcn", dataset="models.lightgcn.dataset:LightGCNDataset",
    loop="pairwise",
    defaults={"emb_dim": 64, "n_layers": 3, "init_method": "normal",
              "lr": 0.001, "weight_decay": 0.0001},
    description="LightGCN（SIGIR 2020）：BPR 三元组 + 图传播"))

register_model(ModelAdapter(
    name="mf", dataset="models.mf.dataset:MFDataset",
    loop="pairwise",
    defaults={"emb_dim": 64, "init_method": "normal",
              "lr": 0.001, "weight_decay": 0.0001},
    description="MF：BPR 三元组，无图传播"))

register_model(ModelAdapter(
    name="wmf", dataset="models.wmf.dataset:WMFDataset",
    loop="full_batch",
    defaults={"factors": 100, "alpha": 40.0, "epsilon": 1e-8,
              "confidence_scheme": "minimal", "optimizer": "als",
              "init_method": "normal", "init_std": 0.01,
              "weight_decay": 0.01},
    description="WMF（Hu 2008）：ALS 全量闭式，每 epoch 一次 sweep"))
