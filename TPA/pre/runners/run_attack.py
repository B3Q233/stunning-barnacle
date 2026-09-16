# -*- coding: utf-8 -*-
"""Phase 5：调用项目**已有**攻击接口生成假用户，并把产物引用登记进 pre。

硬约束（来自生产计划与 AGENTS §6）：
    pre **只做编排**，不复制、不重写任何攻击算法。所有攻击都通过
    attacks/<name>/generate.py::main(config, raw_meta=...) 调用。

为什么能指向退化数据：
    实测确认 random / bandwagon / pgd / tpa / uba 的 generate.main 都接受
    `raw_meta` 参数（默认才回退到 models/{model}/data/processed/{dataset}/meta.pkl），
    因此 pre 可以把"删过交互的干净 meta"直接喂给攻击；
    advinject 走 common.load_meta，支持 config 的 `data_path` 键，同样可指向退化数据。

已知限制（写进本地记录，不隐藏）：
    TPA 的 paths 阶段（path_builder）把 meta 路径写死为干净数据路径，无法重定向。
    即 TPA 的"共现路径"基于干净图构建，而被投毒的 meta 是退化后的。
    由于本实验只删单个物品的交互，共现图几乎不变，但这一点必须记录。

产物：
    attacks/<name>/data/poisoned{_proxy}/...  ← 攻击模块自己的产物（不入库，路径被引用）
    <exp_dir>/attacks/<model>/item_<id>/<attack>_<ratio>/attack.json  ← 引用与统计
"""
from __future__ import annotations

import copy
import glob
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from pre.runners.common import (TPA_ROOT, now_iso, read_json, save_json)
from pre.runners.degrade import ratio_dirname


# 攻击注册表：name -> 模块与调用方式
ATTACK_SPECS: Dict[str, Dict[str, str]] = {
    # entry：攻击模块暴露的入口函数名（默认 main；advinject 用 generate）
    "random":    {"module": "attacks.random.generate",    "entry": "main",
                  "meta_kwarg": "raw_meta"},
    "bandwagon": {"module": "attacks.bandwagon.generate", "entry": "main",
                  "meta_kwarg": "raw_meta"},
    "pgd":       {"module": "attacks.pgd.generate",       "entry": "main",
                  "meta_kwarg": "raw_meta"},
    "tpa":       {"module": "attacks.tpa.generate",       "entry": "main",
                  "meta_kwarg": "raw_meta",
                  "pre_stage": "attacks.tpa.path_builder"},
    "advinject": {"module": "attacks.advinject.generate", "entry": "generate",
                  "meta_kwarg": "data_path"},
    "uba":       {"module": "attacks.uba.generate",       "entry": "main",
                  "meta_kwarg": "raw_meta"},
}


def available_attacks() -> list[str]:
    """当前 pre 能调度的攻击（以仓库现有实现为准）。"""
    return sorted(ATTACK_SPECS)


def register_attack(name: str, module: str, entry: str = "main",
                    meta_kwarg: str = "raw_meta",
                    pre_stage: str | None = None,
                    *, overwrite: bool = False) -> None:
    """扩展约定：登记一个新的攻击，使其可被 pre 调度（不改编排核心）。

    参数：
        name       攻击名（与 attacks/<name>/ 目录名一致）
        module     入口模块，如 "attacks.myattack.generate"
        entry      入口函数名（默认 main；advinject 用 generate）
        meta_kwarg 如何把数据交给攻击：
                     "raw_meta"  → 调用 entry(config, raw_meta=Path)
                     "data_path" → 写进 config["data_path"] 再调用 entry(config)
        pre_stage  可选的前置阶段模块（如 TPA 的 path_builder），同样取 entry(config)

    约束：攻击必须自行把中毒数据写到
        attacks/<name>/data/poisoned{,_proxy}/<dataset>/<model>/<run_tag>/meta.pkl
    pre 按该约定回读产物；不合约定时记录 status=no_output，不会静默跳过。

    用法示例：
        register_attack("myattack", "attacks.myattack.generate")
    """
    if meta_kwarg not in ("raw_meta", "data_path"):
        raise ValueError("meta_kwarg 只支持 'raw_meta' 或 'data_path'")
    if name in ATTACK_SPECS and not overwrite:
        raise ValueError(f"攻击 {name!r} 已登记；如需替换请传 overwrite=True")
    spec = {"module": module, "entry": entry, "meta_kwarg": meta_kwarg}
    if pre_stage:
        spec["pre_stage"] = pre_stage
    ATTACK_SPECS[name] = spec


def _load_attack_config(attack_name: str) -> Dict[str, Any]:
    """读攻击自身的 config.yaml 作为基线（保证默认超参与攻击模块一致）。"""
    path = TPA_ROOT / "attacks" / attack_name / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _import_callable(qualname: str, entry: str = "main"):
    """把 "pkg.mod.attr" 或 "pkg.mod:attr" 解析成可调用对象。

    修复记录：早期实现 rpartition 后仍 getattr(module, "main")，导致
    "attacks.random.generate" 被解析成 attacks.random.main（不存在）。
    """
    import importlib
    if ":" in qualname:
        module_name, _, attr = qualname.partition(":")
        return getattr(importlib.import_module(module_name), attr)
    # 形如 "attacks.random.generate"：必须导入**完整模块路径**，
    # 只导入父包 attacks.random 不会自动挂上 generate 子模块。
    return getattr(importlib.import_module(qualname), entry)


def build_attack_config(cfg: Dict[str, Any], attack_name: str, model_name: str,
                        item_id: int, tag: str,
                        clean_checkpoint: Optional[str]) -> Dict[str, Any]:
    """在攻击自身默认配置上做最小覆盖：数据集/受害模型/目标/预算/输出 tag。"""
    acfg = _load_attack_config(attack_name)
    pre_attack = cfg.get("pre", {}).get("attack", {})
    acfg["dataset"] = cfg.get("dataset")
    acfg["k"] = int(cfg.get("k", 10))
    acfg["run_tag"] = tag
    acfg.setdefault("model", {})["name"] = model_name
    acfg.setdefault("attack", {})
    acfg["attack"]["num_fake_users"] = int(pre_attack.get("num_fake_users", 18))
    acfg["attack"]["ratio"] = None            # 用绝对假用户数，避免 ratio 与模型用户数耦合
    acfg["attack"]["filler_size"] = int(pre_attack.get("filler_size", 20))
    acfg["attack"].setdefault("target_items", {})
    acfg["attack"]["target_items"].update({
        "strategy": "specified", "count": 1, "ids": [int(item_id)]})
    if clean_checkpoint:
        acfg.setdefault("checkpoint", {})["clean"] = clean_checkpoint
        acfg.setdefault("warm_start", {})["enabled"] = False   # 训练口径由 pre 统一控制
    if attack_name == "uba":
        # 用 uplift 的 path 支路（纯数据层），避免依赖 surrogate estimate 缓存
        acfg["attack"].setdefault("uba", {}).setdefault("treatment", {})[
            "method"] = "path"
    if attack_name == "tpa":
        acfg.setdefault("output", {})
    return acfg


def _find_poisoned_meta(attack_name: str, tag: str) -> Optional[Path]:
    """在攻击模块的 data 目录里找本次 run_tag 对应的中毒 meta。"""
    pattern = str(TPA_ROOT / "attacks" / attack_name / "data" / "**" /
                  tag / "meta.pkl")
    hits = sorted(glob.glob(pattern, recursive=True))
    return Path(hits[0]) if hits else None


def _find_artifact(attack_name: str, tag: str, filename: str) -> Optional[str]:
    pattern = str(TPA_ROOT / "attacks" / attack_name / "data" / "**" /
                  tag / filename)
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None


def run_attack(cfg: Dict[str, Any], exp_dir: Path, attack_name: str,
               model_name: str, item_id: int, ratio: float,
               degraded_meta_path: Path, clean_checkpoint: Optional[str],
               seed: int, reuse: bool = True) -> Dict[str, Any]:
    """对（退化数据, 模型, 目标物品）运行一个攻击，返回产物引用记录。

    失败不会中断整个矩阵：异常被捕获并记录 status=failed + error，
    由汇总阶段统一报告（生产计划要求"扩展全部攻击"时必须能继续）。
    """
    label = f"{attack_name}_{ratio_dirname(ratio)}"
    rec_path = (exp_dir / "attacks" / model_name / f"item_{item_id:05d}"
                / label / "attack.json")
    if reuse and rec_path.exists():
        rec = read_json(rec_path)
        # 只复用成功的记录：失败记录必须允许重试（否则修好接口后仍被缓存挡住）
        if rec.get("status") == "ok":
            rec["cached"] = True
            return rec

    tag = f"pre-{cfg.get('experiment', {}).get('name', 'exp')}" \
          f"-{model_name}-item{item_id}-{ratio_dirname(ratio)}"
    spec = ATTACK_SPECS[attack_name]
    acfg = build_attack_config(cfg, attack_name, model_name, item_id, tag,
                               clean_checkpoint)
    if spec["meta_kwarg"] == "data_path":
        acfg["data_path"] = str(degraded_meta_path)

    rec: Dict[str, Any] = {
        "status": "pending", "attack": attack_name, "model_name": model_name,
        "item_id": int(item_id), "deletion_ratio": float(ratio),
        "run_tag": tag, "degraded_meta": str(degraded_meta_path),
        "clean_checkpoint": clean_checkpoint,
        "attack_config": copy.deepcopy(acfg),
        "created_at": now_iso(),
    }
    try:
        if spec.get("pre_stage"):
            pre_main = _import_callable(spec["pre_stage"])
            pre_main(acfg)                       # TPA：先建共现路径（已知限制见模块头）
        main = _import_callable(spec["module"], spec.get("entry", "main"))
        kwargs = {spec["meta_kwarg"]: (str(degraded_meta_path)
                                       if spec["meta_kwarg"] == "raw_meta" else
                                       str(degraded_meta_path))}
        if spec["meta_kwarg"] == "raw_meta":
            main(acfg, raw_meta=Path(degraded_meta_path))
        else:
            main(acfg)
        poisoned = _find_poisoned_meta(attack_name, tag)
        rec.update({
            "status": "ok" if poisoned else "no_output",
            "poisoned_meta": str(poisoned) if poisoned else None,
            "profiles_path": _find_artifact(attack_name, tag, "profiles.json"),
            "stats_path": _find_artifact(attack_name, tag, "stats.json"),
        })
    except Exception as exc:                     # noqa: BLE001 —— 矩阵级容错
        rec.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-2000:]})
    save_json(rec, rec_path)
    return rec


def poisoned_user_delta(poisoned_meta_path: Path, clean_meta: Dict[str, Any]
                        ) -> int:
    """中毒 meta 相对干净 meta 多出来的用户数（= 假用户数，落盘用）。"""
    import pickle
    with open(poisoned_meta_path, "rb") as f:
        pm = pickle.load(f)
    return int(pm["num_users"]) - int(clean_meta["num_users"])
