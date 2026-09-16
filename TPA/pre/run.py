# -*- coding: utf-8 -*-
"""pre 实验入口（Phase 编排 CLI）。

用法（在 TPA 目录下执行）：
    # 单条件全链路验收（Phase 0-6：一个 case 从数据生成跑到距离）
    python pre/run.py --config pre/configs/default.yaml --mode all --tag smoke \
        --limit-items 1 --ratios 0.5 --attacks random --models lightgcn --epochs 5

    # 分阶段执行
    python pre/run.py --mode targets      # Phase 1
    python pre/run.py --mode original     # Phase 2
    python pre/run.py --mode degrade      # Phase 3
    python pre/run.py --mode degraded     # Phase 4
    python pre/run.py --mode attack       # Phase 5
    python pre/run.py --mode analyze      # Phase 11/12

    # 全矩阵
    python pre/run.py --mode all --tag full-run

设计约束：pre 只做编排。攻击一律通过 attacks/<name>/generate.py 调用，
模型一律通过 models/registry.py 解析；不修改任何既有模块。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

TPA_ROOT = Path(__file__).resolve().parents[1]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.runners.common import (experiment_dir, git_commit, load_config,
                                now_iso, save_json)  # noqa: E402
from pre.runners.pipeline import (phase_attack, phase_degrade, phase_degraded,
                                  phase_original, phase_targets)  # noqa: E402
from pre.analysis.aggregate import analyze  # noqa: E402


def resolve_tag(cfg: dict, cli_tag: str | None) -> str:
    """run_tag 优先级：CLI --tag > config.run_tag > 实验名-时间戳。"""
    if cli_tag:
        return cli_tag
    if cfg.get("run_tag"):
        return str(cfg["run_tag"])
    name = cfg.get("experiment", {}).get("name", "pre")
    return f"{name}-{time.strftime('%Y-%m-%d-%H%M')}"


def apply_cli_filters(cfg: dict, args) -> dict:
    """把 CLI 过滤项写回 cfg（用于分阶段验收与部分重跑）。"""
    pre = cfg.setdefault("pre", {})
    if args.models:
        pre["models"] = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.attacks:
        pre["attacks"] = [a.strip() for a in args.attacks.split(",") if a.strip()]
    if args.ratios:
        pre["deletion_ratios"] = [float(x) for x in args.ratios.split(",")]
    if args.epochs:
        pre.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.num_fake_users:
        pre.setdefault("attack", {})["num_fake_users"] = int(args.num_fake_users)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="pre：表示恢复前置实验编排")
    ap.add_argument("--config", type=str,
                    default=str(TPA_ROOT / "pre" / "configs" / "default.yaml"))
    ap.add_argument("--mode", type=str, default="all",
                    choices=["targets", "original", "degrade", "degraded",
                             "attack", "analyze", "all", "doctor"])
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--models", type=str, default=None,
                    help="逗号分隔，覆盖 pre.models（如 lightgcn）")
    ap.add_argument("--attacks", type=str, default=None,
                    help="逗号分隔，覆盖 pre.attacks（如 random,pgd）")
    ap.add_argument("--ratios", type=str, default=None,
                    help="逗号分隔，覆盖 pre.deletion_ratios（如 0.5）")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    ap.add_argument("--num-fake-users", type=int, default=None,
                    help="覆盖攻击预算（假用户数）")
    ap.add_argument("--limit-items", type=int, default=None,
                    help="只跑前 N 个目标物品（分阶段验收用）")
    ap.add_argument("--item-ids", type=str, default=None,
                    help="只跑指定的目标物品 id（逗号分隔）。分批推进用，"
                         "避免单次调用过长导致 stdout 被截断")
    args = ap.parse_args()

    cfg = apply_cli_filters(load_config(Path(args.config)), args)
    tag = resolve_tag(cfg, args.tag)
    exp_dir = experiment_dir(cfg, tag)
    save_json({"config": cfg, "tag": tag, "git_commit": git_commit(),
               "created_at": now_iso(), "argv": sys.argv},
              exp_dir / "manifest.json")
    print(f"[pre] tag={tag} mode={args.mode} models={cfg['pre']['models']} "
          f"attacks={cfg['pre']['attacks']} ratios={cfg['pre']['deletion_ratios']}")

    if args.mode == "doctor":
        # 扩展自检：报告目标模型能否被 pre 驱动（registry / adapter / 接口三查）
        from pre.runners.model_adapters import check_scaffold
        import json as _json
        report = check_scaffold(list(cfg["pre"]["models"]))
        print(_json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    targets = phase_targets(cfg, exp_dir)
    if args.limit_items:
        targets = dict(targets)
        targets["items"] = targets["items"][:int(args.limit_items)]
        targets["num_targets"] = len(targets["items"])
    if args.item_ids:
        wanted = {int(x) for x in args.item_ids.split(",") if x.strip()}
        targets = dict(targets)
        targets["items"] = [t for t in targets["items"]
                            if int(t["item_id"]) in wanted]
        targets["num_targets"] = len(targets["items"])
        if not targets["items"]:
            raise SystemExit(f"[pre] --item-ids 没有匹配到任何目标：{sorted(wanted)}")
    print(f"[pre] 目标物品 {targets['num_targets']} 个："
          f"{[t['item_id'] for t in targets['items']]}")

    if args.mode in ("original", "all"):
        phase_original(cfg, exp_dir, targets)
    if args.mode in ("degrade", "degraded", "all"):
        phase_degrade(cfg, exp_dir, targets)
    if args.mode in ("degraded", "all"):
        phase_degraded(cfg, exp_dir, targets)
    if args.mode in ("attack", "all"):
        phase_attack(cfg, exp_dir, targets)
    if args.mode in ("analyze", "all"):
        res = analyze(cfg, exp_dir, targets)
        print(f"[pre] 完成：{res['num_conditions']} 条距离记录 -> "
              f"{res['files']['embedding_distances']}")
    print(f"[pre] 实验目录：{exp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
