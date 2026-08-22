"""攻击模板 —— 第 1 步：按交互数分类（不再依赖模型嵌入）

1. 加载预处理 meta（含训练交互 train_pairs）
2. 统计每个物品的训练集交互次数
3. 按交互数降序排名划分三档（与流行度直方图 Hot/Medium-hot/Tail 一致）：
   - popular:  交互数前 popular_ratio（默认 5%）
   - ordinary: 5% ~ medium_ratio（默认 40%）
   - cold:     其余 40% ~ 100%
4. 缓存到 attacks/tpa/data/rec_freq/{dataset}/{model}_top{k}.json
   （代理模式下文件名追加 _proxy；model/k/_proxy 仅保留在路径与元数据中，
   分类本身与模型无关，victim / surrogate 共用同一份交互数划分）

用法:
  python attacks/tpa/classify.py --config attacks/tpa/config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Counter as CounterType, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # G:\Idea\TPA
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.classify_common import (  # noqa: E402
    classify_by_interaction_counts,
    interaction_counts,
)
from attacks.tpa.generate import (  # noqa: E402
    load_meta,
    load_yaml_config,
    raw_meta_path,
)
from attacks.tpa.registry import active_model_name  # noqa: E402


def rec_freq_dir(config: Dict[str, Any]) -> Path:
    dataset = config["dataset"]
    attack_name = config["attack"]["name"]
    return PROJECT_ROOT / "attacks" / attack_name / "data" / "rec_freq" / dataset


def rec_freq_path(config: Dict[str, Any], model_name: str, k: int) -> Path:
    tag = "_proxy" if config.get("surrogate", {}).get("enabled", False) else ""
    return rec_freq_dir(config) / f"{model_name}_top{k}{tag}.json"


def resolve_clean_checkpoint(config: Dict[str, Any]) -> Path:
    """干净模型 checkpoint：classification.checkpoint > clean_checkpoint >
    warm_start.checkpoint（供 paths/fit 阶段使用；classify 本身已不加载模型）。"""
    cls_cfg = config.get("classification", {})
    candidates = [
        cls_cfg.get("checkpoint"),
        config.get("clean_checkpoint"),
        config.get("warm_start", {}).get("checkpoint"),
    ]
    for c in candidates:
        if c:
            p = Path(c)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                return p
    raise FileNotFoundError(
        "未找到可用的干净模型 checkpoint。请在配置中设置 "
        "classification.checkpoint / clean_checkpoint / warm_start.checkpoint"
    )


def resolve_active_checkpoint(config: Dict[str, Any]) -> Path:
    """paths 阶段使用的 checkpoint：代理模式用 surrogate.checkpoint，
    否则用干净模型（白盒对照）。"""
    sur = config.get("surrogate", {})
    if sur.get("enabled", False):
        ck = sur.get("checkpoint")
        if not ck:
            raise FileNotFoundError(
                "surrogate.enabled=true 但未配置 surrogate.checkpoint"
            )
        p = Path(ck)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"代理模型 checkpoint 不存在: {p}")
        return p
    return resolve_clean_checkpoint(config)


def save_cache(config: Dict[str, Any], model_name: str, k: int,
               counts: CounterType[int],
               categories: Dict[str, List[int]], summary: Dict[str, Any]) -> Path:
    out = rec_freq_path(config, model_name, k)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": config["dataset"],
        "model": model_name,
        "k": k,
        "proxy": config.get("surrogate", {}).get("enabled", False),
        "basis": "interaction_count",
        "popular_ratio": summary["popular_ratio"],
        "medium_ratio": summary["medium_ratio"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "counts": {str(i): c for i, c in counts.items()},
        "categories": categories,
        "summary": summary,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out


def load_cache(config: Dict[str, Any], model_name: str, k: int,
               required: bool = False) -> Dict[str, Any] | None:
    """读取分类缓存；required=True 时缺失直接报错并提示先跑 classify。"""
    path = rec_freq_path(config, model_name, k)
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"交互数分类缓存不存在: {path}\n"
                f"请先运行: python attacks/tpa/run.py --mode classify"
            )
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data["counts"] = {int(i): c for i, c in data["counts"].items()}
    return data


from training.timing import timed  # noqa: E402


@timed("交互数分类")
def main(config: Dict[str, Any]) -> Dict[str, Any]:
    dataset = config["dataset"]
    model_name = active_model_name(config)
    cls_cfg = config.get("classification", {})
    k = cls_cfg.get("k") or config.get("training", {}).get("k") or 20
    popular_ratio = cls_cfg.get("popular_ratio", 0.05)
    medium_ratio = cls_cfg.get("medium_ratio", 0.40)

    meta = load_meta(raw_meta_path(config))
    counts = interaction_counts(meta["train_pairs"])
    categories, summary = classify_by_interaction_counts(
        counts, meta["num_items"], popular_ratio, medium_ratio
    )
    out = save_cache(config, model_name, k, counts, categories, summary)

    mode = "代理(surrogate)" if config.get("surrogate", {}).get("enabled", False) else "白盒(victim)"
    print(f"[classify] 数据集={dataset}（{mode}，路径后缀按代理模式区分），"
          f"{meta['num_users']} 用户 / {meta['num_items']} 物品，按训练集交互数分类")
    print(f"[classify] 有交互物品 {summary['interacting_items']}/{summary['num_items']}，"
          f"划分：popular {summary['popular_count']} / "
          f"ordinary {summary['ordinary_count']} / cold {summary['cold_count']}")
    print(f"[classify] popular 阈值：交互数 ≥ {summary.get('min_popular_count')}；"
          f"cold 最高交互数 {summary.get('max_cold_count')}")
    print(f"[classify] 热门物品示例: {summary['top_interaction_items'][:5]}")
    print(f"[classify] 冷门物品示例: {categories['cold'][:5]}"
          f"（共 {len(categories['cold'])} 个）")
    print(f"[classify] 缓存 → {out}")
    return {"counts": counts, "categories": categories, "summary": summary,
            "cache_path": str(out)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="交互数分类（攻击模板第 1 步）")
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "attacks" / "tpa" / "config.yaml"),
    )
    args = parser.parse_args()
    main(load_yaml_config(Path(args.config)))
