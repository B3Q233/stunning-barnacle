"""配置 k 统一解析与指标模板展开。

目标：k 只在一处定义（最外层 `k`，或回退 evaluation/classification/training.k），
各段与指标名自动绑定，避免"改 k 要改多处"。

约定：
- 指标名支持 `{k}` 模板，如 `target_ndcg@{k}` → `target_ndcg@10`；
- `apply_k` 返回深拷贝，不修改入参，可重复调用（幂等）。
"""
from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any, Dict, List


def resolve_k(cfg: Dict[str, Any], default: int = 20) -> int:
    """解析统一 k：顶层 `k` > evaluation.k > classification.k > training.k > default。"""
    if cfg.get("k") is not None:
        return int(cfg["k"])
    for section in ("evaluation", "classification", "training"):
        sec = cfg.get(section)
        if isinstance(sec, dict) and sec.get("k") is not None:
            return int(sec["k"])
    return int(default)


def _expand_name(name: str, k: int) -> str:
    return str(name).replace("{k}", str(k))


def expand_metrics(metrics: Any, k: int) -> Any:
    """把指标列表中的 `{k}` 模板展开为实际 K。"""
    if not metrics:
        return metrics
    out: List[Any] = []
    for item in metrics:
        if isinstance(item, dict):
            out.append({_expand_name(name, k): direction
                        for name, direction in item.items()})
        elif isinstance(item, str):
            parts = item.strip().split()
            name = _expand_name(parts[0], k)
            out.append(" ".join([name] + parts[1:]) if len(parts) > 1 else name)
        else:
            out.append(item)
    return out


def apply_k(cfg: Dict[str, Any], default: int = 20) -> Dict[str, Any]:
    """返回副本：解析统一 k，注入各段 k，展开 evaluation.metrics / 顶层 metrics。"""
    out = copy.deepcopy(cfg)
    k = resolve_k(out, default)
    for section in ("classification", "training", "evaluation"):
        if isinstance(out.get(section), dict):
            out[section]["k"] = k
    ev = out.get("evaluation")
    if isinstance(ev, dict) and "metrics" in ev:
        ev["metrics"] = expand_metrics(ev["metrics"], k)
    if "metrics" in out:
        out["metrics"] = expand_metrics(out["metrics"], k)
    return out


# ── canonicalization pipeline（唯一兼容层；业务代码禁止读取 legacy）──

_MISSING = object()
_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "config-template.unified.yaml"
)


def _warn(message: str) -> None:
    warnings.warn(message, UserWarning, stacklevel=3)


def _get(cfg: dict, keys) -> Any:
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return _MISSING
        cur = cur[key]
    return cur


def _pop(cfg: dict, keys) -> Any:
    if len(keys) == 1:
        return cfg.pop(keys[0], _MISSING)
    cur = cfg
    for key in keys[:-1]:
        if not isinstance(cur, dict) or key not in cur:
            return _MISSING
        cur = cur[key]
    if not isinstance(cur, dict):
        return _MISSING
    return cur.pop(keys[-1], _MISSING)


def _set(cfg: dict, keys, value) -> None:
    cur = cfg
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _move(out: dict, src, dst, conflict: str = "warn") -> None:
    value = _pop(out, src)
    if value is _MISSING:
        return
    existing = _get(out, dst)
    if existing is not _MISSING:
        if conflict == "raise":
            raise ValueError(
                f"[config] 旧键 {'.'.join(src)} 与 canonical 键 "
                f"{'.'.join(dst)} 冲突"
            )
        _warn(
            f"[config] canonical 键 {'.'.join(dst)} 已存在，"
            f"忽略旧键 {'.'.join(src)}"
        )
        return
    _set(out, dst, value)
    _warn(f"[config] 旧键 {'.'.join(src)} 已映射为 {'.'.join(dst)}")


def _prune_empty(out: dict, section: str) -> None:
    if isinstance(out.get(section), dict) and not out[section]:
        out.pop(section, None)


def _dataset_and_seed(out: dict) -> None:
    for section in ("data", "experiment"):
        sec = out.get(section)
        if not isinstance(sec, dict):
            continue
        if "dataset" in sec:
            _move(out, (section, "dataset"), ("dataset",))
        if section == "experiment" and "seed" in sec:
            _move(out, ("experiment", "seed"), ("seed",))
        _prune_empty(out, section)
    if isinstance(out.get("override"), dict):
        out["override"] = canonicalize_config(out["override"])


def _device(out: dict) -> None:
    value = _pop(out, ("use_cuda",))
    if value is _MISSING:
        return
    device = "cuda" if bool(value) else "cpu"
    existing = _get(out, ("training", "device"))
    if existing is not _MISSING:
        _warn("[config] canonical 键 training.device 已存在，忽略旧键 use_cuda")
        return
    training = out.setdefault("training", {})
    training["device"] = device
    _warn("[config] 旧键 use_cuda 已映射为 training.device")


def _eval_every(out: dict) -> None:
    _move(out, ("evaluation", "eval_every"), ("training", "eval_every"))


def _k(out: dict) -> None:
    top = _get(out, ("k",))
    if top is not _MISSING:
        for section in ("evaluation", "classification", "training"):
            if isinstance(out.get(section), dict) and "k" in out[section]:
                _warn(
                    f"[config] canonical 键 k 已存在，"
                    f"忽略旧键 {section}.k"
                )
                out[section].pop("k", None)
        return
    for section in ("evaluation", "classification", "training"):
        value = _pop(out, (section, "k"))
        if value is not _MISSING:
            out["k"] = int(value)
            _warn(f"[config] 旧键 {section}.k 已映射为顶层 k")
            break


def _output_dir(out: dict) -> None:
    _move(out, ("output_dir",), ("output", "dir"))


def _lambda_reg(out: dict) -> None:
    _move(out, ("training", "lambda_reg"), ("training", "weight_decay"))


def _checkpoint(out: dict) -> None:
    clean = _get(out, ("checkpoint", "clean"))
    sources = [("clean_checkpoint",), ("classification", "checkpoint")]
    values = []
    for src in sources:
        value = _get(out, src)
        if value is not _MISSING:
            values.append((src, value))
    if clean is _MISSING and values:
        distinct = {value for _, value in values}
        if len(distinct) > 1:
            raise ValueError(
                "[config] 多个旧 checkpoint 键值冲突: "
                + ", ".join(f"{'.'.join(s)}={v!r}" for s, v in values)
            )
        _set(out, ("checkpoint", "clean"), values[0][1])
        _warn("[config] 旧 checkpoint 键已映射为 checkpoint.clean")
    for src, _ in values:
        _pop(out, src)
        if clean is not _MISSING:
            _warn(
                f"[config] canonical 键 checkpoint.clean 已存在，"
                f"忽略旧键 {'.'.join(src)}"
            )


def _fake_users(out: dict) -> None:
    attack = out.get("attack")
    if not isinstance(attack, dict):
        return
    value = attack.pop("n_fakes", _MISSING)
    if value is _MISSING:
        return
    has = "num_fake_users" in attack or "ratio" in attack
    if has:
        _warn("[config] canonical 假用户数键已存在，忽略旧键 attack.n_fakes")
        return
    if float(value) > 1:
        attack["num_fake_users"] = int(value)
    else:
        attack["ratio"] = float(value)
    _warn("[config] 旧键 attack.n_fakes 已按值分支映射")


def _target_items(out: dict) -> None:
    attack = out.get("attack")
    if not isinstance(attack, dict):
        return
    raw = attack.get("target_items")
    ti = raw if isinstance(raw, dict) else {}
    if isinstance(raw, list):
        ti = {"ids": list(raw)}
        attack.pop("target_items")
        _warn("[config] 旧 attack.target_items 列表已映射为 ids")
    count = attack.pop("n_target_items", _MISSING)
    zone = attack.pop("target_item_popularity", _MISSING)
    if count is not _MISSING:
        if "count" in ti:
            _warn("[config] canonical 键 target_items.count 已存在，"
                  "忽略旧键 attack.n_target_items")
        else:
            ti["count"] = int(count)
            _warn("[config] 旧键 attack.n_target_items 已映射为 "
                  "attack.target_items.count")
    if zone is not _MISSING:
        if "zone" in ti:
            _warn("[config] canonical 键 target_items.zone 已存在，"
                  "忽略旧键 attack.target_item_popularity")
        else:
            ti["zone"] = zone
            _warn("[config] 旧键 attack.target_item_popularity 已映射为 "
                  "attack.target_items.zone")
    if "ids" in ti and "strategy" not in ti:
        ti["strategy"] = "specified"
    if ti:
        attack["target_items"] = ti


def _adv(out: dict) -> None:
    attack = out.get("attack")
    if not isinstance(attack, dict):
        return
    mapping = {
        "adv_epochs": "epochs",
        "adv_lr": "lr",
        "adv_momentum": "momentum",
        "proj_threshold": "proj_threshold",
        "click_targets": "click_targets",
        "attack_type": "attack_type",
    }
    adv = attack.setdefault("adv", {})
    for old, new in mapping.items():
        value = attack.pop(old, _MISSING)
        if value is _MISSING:
            continue
        if new in adv:
            _warn(f"[config] canonical 键 attack.adv.{new} 已存在，"
                  f"忽略旧键 attack.{old}")
        else:
            adv[new] = value
            _warn(f"[config] 旧键 attack.{old} 已映射为 attack.adv.{new}")


def _surrogate(out: dict) -> None:
    sur = out.get("surrogate")
    if not isinstance(sur, dict):
        return
    _move(sur, ("model_name",), ("name",))
    _move(sur, ("l2",), ("training", "weight_decay"))
    training = sur.setdefault("training", {})
    flat = ("epochs", "lr", "batch_size", "hidden_dims",
            "weight_alpha", "unroll_steps", "weight_decay")
    for key in flat:
        value = sur.pop(key, _MISSING)
        if value is _MISSING:
            continue
        if key in training:
            _warn(
                f"[config] canonical 键 surrogate.training.{key} 已存在，"
                f"忽略旧平铺键 surrogate.{key}"
            )
        else:
            training[key] = value
            _warn(
                f"[config] 旧平铺键 surrogate.{key} 已映射为 "
                f"surrogate.training.{key}"
            )
    if "unroll_steps" not in training:
        attack = out.get("attack")
        if isinstance(attack, dict):
            value = attack.pop("unroll_steps", _MISSING)
            if value is not _MISSING:
                training["unroll_steps"] = value
                _warn("[config] 旧键 attack.unroll_steps 已映射为 "
                      "surrogate.training.unroll_steps")


def _percentiles(out: dict) -> None:
    cls = out.get("classification")
    if not isinstance(cls, dict):
        return
    mapping = {
        "popular_percentile": "percentile_head",
        "torso_percentile": "percentile_upper_torso",
        "tail_percentile": "percentile_lower_torso",
    }
    for old, new in mapping.items():
        value = cls.pop(old, _MISSING)
        if value is _MISSING:
            continue
        if new in cls:
            _warn(
                f"[config] canonical 键 classification.{new} 已存在，"
                f"忽略旧键 classification.{old}"
            )
        else:
            cls[new] = value
            _warn(
                f"[config] 旧键 classification.{old} 已映射为 "
                f"classification.{new}"
            )


_PIPELINE = (
    _dataset_and_seed,
    _device,
    _eval_every,
    _k,
    _output_dir,
    _lambda_reg,
    _checkpoint,
    _fake_users,
    _target_items,
    _adv,
    _surrogate,
    _percentiles,
)


def canonicalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """把旧别名配置规范化为 canonical；canonical 优先，不修改入参。"""
    out = copy.deepcopy(cfg)
    for handler in _PIPELINE:
        handler(out)
    return out


def load_config(path) -> Dict[str, Any]:
    """读取 yaml 并 canonicalize；业务代码统一经此加载配置。"""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return canonicalize_config(cfg)


def schema_leaf_paths(template_path=None) -> set:
    """解析统一模板为合法叶子路径集合（唯一 schema 事实源）。"""
    import yaml

    path = Path(template_path) if template_path else _TEMPLATE_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    leaves = set()

    def walk(node, prefix):
        if not isinstance(node, dict):
            leaves.add(prefix)
            return
        if not node:
            leaves.add(prefix)
            return
        for key, value in node.items():
            walk(value, f"{prefix}.{key}" if prefix else str(key))

    walk(data, "")
    return leaves
