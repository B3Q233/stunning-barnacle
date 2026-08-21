"""
松耦合训练架构
结构：
    TrainableModel  TrainingConfig / DatasetProtocol / Experiment / Trainer / CallbackHandler

本文件是论文复现代码生成的固定骨架模板。代码骨架生成 skill 在使用本文件时，
只能在标注 TODO 的位置填空，不能修改既有的类结构、方法签名、契约接口。
"""

from __future__ import annotations

import copy
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, TypeVar

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
import yaml
import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────
# 1. DatasetProtocol  数据载入器接口
# ─────────────────────────────────────────────
T = TypeVar('T', bound=Dataset)
class DatasetProtocol(ABC, Generic[T]):
    """所有数据载入器必须实现的协议接口。
    与模型强相关：输出的 batch 格式须满足对应模型 train_step 的输入要求。
    """

    @abstractmethod
    def train_loader(self) -> TorchDataLoader:
        """返回训练集 DataLoader。"""

    @abstractmethod
    def test_loader(self) -> TorchDataLoader:
        """返回测试集 DataLoader。"""

    @abstractmethod
    def val_loader(self) -> TorchDataLoader:
        """返回验证集 DataLoader。"""

    @abstractmethod
    def get_init_params(self) -> Dict[str, Any]:
        """返回模型初始化所需的参数，如 num_users、num_items 等。"""
    @abstractmethod
    def get_dataset(self, split: str)-> T:
        """返回指定分割的 Dataset，split 为 'train'/'val'/'test'"""


# ─────────────────────────────────────────────
# 2. TrainableModel  模型抽象基类
# ─────────────────────────────────────────────

class TrainableModel(nn.Module,ABC):
    """所有可训练模型的抽象契约。
    Trainer 只通过此接口与模型交互，实现真正的多态松耦合。
    """
    def __init__(self, config):
        super().__init__()

        # 核心逻辑：智能选择设备
        requested_device = config.device.lower()
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            print("⚠️ [警告] 环境不支持 CUDA，已自动切换到 CPU。")
            self._device = torch.device("cpu")
        else:
            self._device = torch.device(requested_device)

        # 然后再执行 .to()
        self.to(self._device)


    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """执行一个训练 step（前向 + 反向 + 参数更新）。
        返回包含训练指标的键值对，如 {"loss": 0.35, "acc": 0.91}。
        """

    @abstractmethod
    def eval_step(self, batch: Any) -> Dict[str, float]:
        """执行一个评估 step（纯推理，无梯度）。

        ⚠️ 契约：返回值必须全部为标量（float/int/0-dim Tensor）。
        Trainer.run() 使用 sum(v)/len(v) 聚合各 batch 的指标，
        如果返回值包含大批量张量（如 shape (B,N) 的预测评分），
        不同 batch 尺寸不同（尾批）会导致 sum() 形状不匹配而 crash。

        返回示例: {"val_loss": 0.28, "val_acc": 0.93}

        如果模型需要输出大批量预测（如推荐系统的全量排序评分），
        请定义独立方法（如 predict_full_ranking()），不要通过 eval_step 返回。
        """

    @abstractmethod
    def build_dataloader(self, config: "TrainingConfig") -> DatasetProtocol:
        """根据 config 构造与本模型匹配的数据载入器。"""

    def save_params(self, path: str) -> None:
        torch.save(self.state_dict(), path)      # 直接用 nn.Module 的方法

    def load_params(self, path: str) -> None:
        self.load_state_dict(torch.load(path))

    def set_train(self) -> None:
        self.train()                             # nn.Module 原生方法

    def set_eval(self) -> None:
        self.eval()


# ─────────────────────────────────────────────
# 3. TrainingConfig  配置参数器
# ─────────────────────────────────────────────

class TrainingConfig:
    """
    访问配置器
    """
    # 默认值（使用类变量作为模板，初始化时深拷贝）
    _DEFAULTS = {
        "lr": 1e-3,
        "epochs": 10,
        "batch_size": 32,
        "weight_decay": 0.0,
        "device": "cuda",
        "checkpoint_dir": "checkpoints",
        "save_every_n_epochs": 1,
        "shuffle":True,
        "val_path":"val_data.pt",
    }

    def __init__(self, overrides: Optional[Dict[str, Any]] = None) -> None:
        self.__dict__["_cfg"] = copy.deepcopy(self._DEFAULTS)
        if overrides:
            self._cfg.update(overrides)


    def __getattr__(self, key: str) -> Any:
        try:
            cfg = object.__getattribute__(self, '_cfg')
            return cfg[key]
        except KeyError:
            raise AttributeError(f"TrainingConfig 没有配置项: '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        if "_cfg" in self.__dict__:
            self._cfg[key] = value
        else:
            super().__setattr__(key, value)

    def __getitem__(self, key: str) -> Any:
        return self._cfg[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._cfg[key] = value

    def update(self, overrides: Dict[str, Any]) -> "TrainingConfig":
        self._cfg.update(overrides)
        return self

    def get(self, key: str, default: Any = None) -> Any:
        return self._cfg.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._cfg)

    def clone(self) -> "TrainingConfig":
        new_overrides = copy.deepcopy(self._cfg)
        return TrainingConfig(overrides=new_overrides)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        path_obj = Path(path).resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"配置文件不存在: {path_obj}")
        with open(path_obj, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(overrides=raw)

    def __repr__(self) -> str:
        return f"TrainingConfig({json.dumps(self._cfg, indent=4)})"

class ConfigBuilder:
    """链式 Builder，支持预设方法和自定义参数。"""

    def __init__(self) -> None:
        self._cfg = TrainingConfig()

    def lr(self, v: float) -> "ConfigBuilder":
        self._cfg.lr = v
        return self

    def epochs(self, v: int) -> "ConfigBuilder":
        self._cfg.epochs = v
        return self

    def batch_size(self, v: int) -> "ConfigBuilder":
        self._cfg.batch_size = v
        return self

    def device(self, v: str) -> "ConfigBuilder":
        self._cfg.device = v
        return self

    def checkpoint_dir(self,v:str)->"ConfigBuilder":
        self._cfg.checkpoint_dir = v
        return self

    def save_every_n_epochs(self,v:int)->"ConfigBuilder":
        self._cfg.save_every_n_epochs = v
        return self

    def set(self, key: str, value: Any) -> "ConfigBuilder":
        """用于设置默认值以外的参数，如 builder.set("n_layers", 3)"""
        self._cfg[key] = value
        return self

    def from_file(self, path: str) -> "ConfigBuilder":
        if path.endswith(".yaml") or path.endswith(".yml"):
            self._cfg.update(TrainingConfig.from_yaml(path).as_dict())
        return self

    def build(self) -> TrainingConfig:
        return self._cfg


# ─────────────────────────────────────────────
# 4. Experiment  实验描述层（中间聚合）
# ─────────────────────────────────────────────

class Experiment:
    """聚合 Config + Model + DataLoader，统一描述一次训练实验。"""

    def __init__(self, config: TrainingConfig, model: TrainableModel) -> None:
        self.config = config
        self.model = model
        self.dataloader: DatasetProtocol = model.build_dataloader(config)

    @classmethod
    def from_builder(cls, builder: ConfigBuilder, model: TrainableModel) -> "Experiment":
        return cls(builder.build(), model)


# ─────────────────────────────────────────────
# 5. CallbackHandler  观察者回调层
# ─────────────────────────────────────────────

class Callback(ABC):
    def on_train_begin(self, config: TrainingConfig) -> None: ...
    def on_epoch_begin(self, epoch: int, total: int) -> None: ...
    def on_batch_end(self, batch_idx: int, metrics: Dict[str, float]) -> None: ...
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None: ...
    def on_error(self, exc: Exception) -> None: ...
    def on_train_end(self, metrics: Dict[str, float]) -> None: ...


class CallbackHandler:
    def __init__(self, callbacks: Optional[List[Callback]] = None) -> None:
        self.callbacks: List[Callback] = callbacks or []

    def add(self, cb: Callback) -> "CallbackHandler":
        self.callbacks.append(cb)
        return self

    def on_train_begin(self, config: TrainingConfig) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(config)

    def on_epoch_begin(self, epoch: int, total: int) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, total)

    def on_batch_end(self, batch_idx: int, metrics: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(batch_idx, metrics)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, metrics)

    def on_error(self, exc: Exception) -> None:
        for cb in self.callbacks:
            cb.on_error(exc)

    def on_train_end(self, metrics: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_train_end(metrics)


class LoggerCallback(Callback):
    def on_train_begin(self, config: TrainingConfig) -> None:
        print(f"\n{'='*50}")
        print(f"  训练开始  lr={config.lr}  epochs={config.epochs}  "
              f"batch_size={config.batch_size}  device={config.device}")
        print(f"{'='*50}")

    def on_epoch_begin(self, epoch: int, total: int) -> None:
        print(f"\n[Epoch {epoch}/{total}]")

    def on_batch_end(self, batch_idx: int, metrics: Dict[str, float]) -> None:
        kv = "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        print(f"  batch {batch_idx:4d} | {kv}", end="\r")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        kv = "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        print(f"\n  epoch 汇总 | {kv}")

    def on_error(self, exc: Exception) -> None:
        print(f"\n[错误] {type(exc).__name__}: {exc}")

    def on_train_end(self, metrics: Dict[str, float]) -> None:
        print(f"\n{'='*50}")
        print("  训练完成")
        print(f"{'='*50}\n")


class CheckpointCallback(Callback):
    def __init__(self, model: TrainableModel, config: TrainingConfig) -> None:
        self._model = model
        self._cfg = config
        self._last_ckpt: Optional[str] = None
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        if epoch % self._cfg.save_every_n_epochs == 0:
            path = f"{self._cfg.checkpoint_dir}/epoch_{epoch:04d}.pt"
            self._model.save_params(path)
            self._last_ckpt = path
            print(f"  [ckpt] 已保存 → {path}")

    def on_error(self, exc: Exception) -> None:
        if self._last_ckpt:
            print(f"  [ckpt] 异常，回滚至 {self._last_ckpt}")
            self._model.load_params(self._last_ckpt)


class MetricAccumulator(Callback):
    def __init__(self) -> None:
        self.history: List[Dict[str, float]] = []

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        self.history.append({"epoch": epoch, **metrics})


# ─────────────────────────────────────────────
# 6. Trainer  训练主体
# ─────────────────────────────────────────────

class Trainer:
    def __init__(self, callbacks: Optional[List[Callback]] = None) -> None:
        self._handler = CallbackHandler(callbacks or [])

    def add_callback(self, cb: Callback) -> "Trainer":
        self._handler.add(cb)
        return self

    def run(self, experiment: Experiment) -> Dict[str, float]:
        cfg = experiment.config
        model = experiment.model

        print("加载数据...")
        train_loader = experiment.dataloader.train_loader()
        val_loader = experiment.dataloader.val_loader()
        experiment.dataloader.test_loader()
        print("数据加载完成")

        last_metrics: Dict[str, float] = {}
        self._handler.on_train_begin(cfg)

        try:
            for epoch in range(1, cfg.epochs + 1):
                self._handler.on_epoch_begin(epoch, cfg.epochs)

                model.set_train()
                epoch_train: Dict[str, List[float]] = {}
                for batch_idx, batch in enumerate(train_loader):
                    metrics = model.train_step(batch)
                    self._handler.on_batch_end(batch_idx, metrics)
                    for k, v in metrics.items():
                        epoch_train.setdefault(k, []).append(v)

                model.set_eval()
                epoch_val: Dict[str, List[float]] = {}
                with torch.no_grad():
                    for batch in val_loader:
                        metrics = model.eval_step(batch)
                        for k, v in metrics.items():
                            epoch_val.setdefault(k, []).append(v)

                last_metrics = {
                    k: sum(v) / len(v)
                    for k, v in {**epoch_train, **epoch_val}.items()
                }
                self._handler.on_epoch_end(epoch, last_metrics)

        except Exception as exc:
            self._handler.on_error(exc)
            raise

        self._handler.on_train_end(last_metrics)
        return last_metrics

    def run_batch(
        self,
        base_config: TrainingConfig,
        model_cls: type,
        param_grid: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        results = []
        for overrides in param_grid:
            cfg = base_config.clone().update(overrides)
            model = model_cls(cfg)
            exp = Experiment(cfg, model)
            print(f"\n>>> 批量训练参数: {overrides}")
            metrics = self.run(exp)
            results.append({"params": overrides, "metrics": metrics})
        return results

# ─────────────────────────────────────────────
# 7. reader 数据的自定义读取
# ─────────────────────────────────────────────
class DatasetReader:
    """
    数据集读取工具类，支持多种文件格式自适应读取和自定义解析器注册。
    内置格式：.csv / .txt / .tsv。输出约定：前两列必须为 user_id, item_id
    """

    _readers: dict[str, Callable[[str], pd.DataFrame]] = {
        '.csv': lambda path: pd.read_csv(path),
        '.txt': lambda path: pd.read_csv(path, sep=r'\s+', header=None),
        '.tsv': lambda path: pd.read_csv(path, sep='\t',   header=None),
    }

    def __init__(self, reader: Optional[Callable[[str], pd.DataFrame]] = None):
        self._custom_reader = reader

    def read(self, path: str) -> pd.DataFrame:
        if self._custom_reader is not None:
            return self._custom_reader(path)
        suffix = Path(path).suffix.lower()
        if suffix not in self._readers:
            raise ValueError(
                f"不支持的文件格式: '{suffix}'，"
                f"已注册格式: {list(self._readers.keys())}，"
                f"可通过 DatasetReader.register() 注册新格式，"
                f"或构造时传入 reader 参数"
            )
        return self._readers[suffix](path)

    @classmethod
    def register(cls, suffix: str, reader: Callable[[str], pd.DataFrame]) -> None:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        cls._readers[suffix.lower()] = reader

    @classmethod
    def supported_formats(cls) -> list[str]:
        return list(cls._readers.keys())


# ═══════════════════════════════════════════════
# ↓↓↓ 以下为待填充区域，按 数据处理→数据导入→模型结构→模型评估→
#     模型训练→结果展示 的顺序逐段实现，每段实现后单独跑测试 ↓↓↓
# ═══════════════════════════════════════════════

# ── TODO[数据处理/数据导入]：实现 Dataset 与 DatasetProtocol ──
#
# class PaperDataset(Dataset):
#     """
#     TODO: 实现论文对应的数据集类
#     输入 shape:  ...
#     输出 shape:  ...
#     """
#     def __init__(self, ...): ...
#     def __len__(self) -> int: ...
#     def __getitem__(self, idx: int): ...
#
# class PaperDataLoader(DatasetProtocol["PaperDataset"]):
#     """TODO: 实现与 PaperModel 配套的数据载入器"""
#     def train_loader(self) -> TorchDataLoader: ...
#     def val_loader(self) -> TorchDataLoader: ...
#     def test_loader(self) -> TorchDataLoader: ...
#     def get_init_params(self) -> Dict[str, Any]: ...
#     def get_dataset(self, split: str): ...


# ── TODO[模型结构]：实现 TrainableModel 子类 ──
#
# class PaperModel(TrainableModel):
#     """
#     TODO: 实现论文对应的模型结构
#     forward 输入 shape: ...
#     forward 输出 shape: ...
#     """
#     def __init__(self, config: TrainingConfig, ...):
#         super().__init__(config=config)
#         # TODO: 定义网络层
#
#     def forward(self, x):
#         # TODO: 前向传播，每层输出 shape 写在注释里
#         ...
#
#     def train_step(self, batch: Any) -> Dict[str, float]:
#         # TODO: 前向 + 损失计算 + 反向 + 参数更新
#         ...
#
#     def eval_step(self, batch: Any) -> Dict[str, float]:
#         # TODO: 纯推理 + 指标计算（对齐论文报告的指标格式）
#         ...
#
#     def build_dataloader(self, config: TrainingConfig) -> DatasetProtocol:
#         return PaperDataLoader(config)


# ── TODO[模型评估]：评估指标计算（与理解文档中的"评估方式"字段严格对齐） ──
#
# def compute_metrics(predictions, targets, k: int = 20) -> Dict[str, float]:
#     """TODO: 按论文指定的协议实现指标计算（如 Recall@K / NDCG@K）"""
#     ...


# ── TODO[结果展示]：训练历史可视化、与论文报告指标的对比 ──
#
# def plot_training_history(history: List[Dict[str, float]]) -> None:
#     """TODO: 绘制 loss/metric 曲线"""
#     ...
#
# def compare_with_paper(reproduced: Dict[str, float], reported: Dict[str, float]) -> None:
#     """TODO: 生成"论文报告值 vs 复现值"对比表"""
#     ...


# ─────────────────────────────────────────────
# 使用示例（TODO 全部填完后，参照此处组装/调整）
# ─────────────────────────────────────────────

if __name__ == "__main__":
    config = (
        ConfigBuilder()
        .build()
    ).from_yaml("tools/test.yml")
    print(config.as_dict())

    # TODO: 替换为 PaperModel / PaperDataLoader
    # model = PaperModel(config)
    # experiment = Experiment(config, model)
    # metric_log = MetricAccumulator()
    #
    # trainer = Trainer(callbacks=[
    #     LoggerCallback(),
    #     CheckpointCallback(model, config),
    #     metric_log,
    # ])
    #
    # trainer.run(experiment)
    # print("训练历史:", metric_log.history)
    # plot_training_history(metric_log.history)
