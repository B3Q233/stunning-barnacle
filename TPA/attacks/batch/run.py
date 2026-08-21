"""批量投毒攻击编排 CLI。"""
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # G:\Idea\TPA
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    import argparse
    import shutil

    import yaml

    from attacks.batch.aggregate import (
        build_results_rows, compute_clean_baseline, tier_summary,
        write_results_csv, write_summary_md, write_tier_stats_json)
    from attacks.batch.generator import build_atomic_base, load_batch_config
    from attacks.batch.runner import (
        _cleanup_staging, ensure_classify_cache, run_atomic, run_batch,
        staging_dir)
    from attacks.batch.utils import group_name, public_rec_freq_path
    from training.config_utils import resolve_k
    from training.run_tag import resolve_run_tag
    from training.timing import section_enter, section_exit

    parser = argparse.ArgumentParser(description="批量投毒攻击")
    parser.add_argument("--config", type=str,
                        default="attacks/batch/config.yaml")
    parser.add_argument("--mode", choices=["generate", "run", "aggregate", "all"],
                        default="all")
    parser.add_argument("--batch-tag", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-classify", action="store_true")
    parser.add_argument("--max-targets", type=int, default=None)
    args = parser.parse_args()

    cfg = load_batch_config(Path(args.config))
    batch_tag = args.batch_tag or resolve_run_tag(cfg)
    out_root = PROJECT_ROOT / Path(cfg.get("output", {}).get(
        "dir", "attacks/batch/output")) / batch_tag
    configs_dir, runs_root = out_root / "configs", out_root / "runs"
    k = resolve_k(cfg)
    base = build_atomic_base(cfg)
    group = group_name(base)

    if args.mode in ("generate", "all"):
        if args.skip_classify and not public_rec_freq_path(base).exists():
            raise FileNotFoundError(
                f"缓存不存在：{public_rec_freq_path(base)}"
                "（--skip-classify 需要已有缓存）")
        cache = ensure_classify_cache(cfg)
        run_batch(cfg, batch_tag, out_root, cache,
                  max_targets=args.max_targets,
                  dry_run=(args.mode == "generate") or args.dry_run)
    if args.mode == "run":
        for p in sorted(configs_dir.rglob("*.yaml")):
            atomic = yaml.safe_load(p.read_text(encoding="utf-8"))
            print(f"[batch] run {atomic['run_tag']}")
            run_atomic(atomic, "data")
            run_atomic(atomic, "model")
            src = staging_dir(runs_root, atomic)
            dst = runs_root / p.relative_to(configs_dir).with_suffix("")
            if src != dst and src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    shutil.rmtree(str(dst))
                shutil.move(str(src), str(dst))
                _cleanup_staging(runs_root, src)
    if args.mode in ("aggregate", "all") and not args.dry_run:
        _t = section_enter("结果整合")
        rows = build_results_rows(runs_root, group, cfg, k)
        summary = tier_summary(rows, k)
        write_results_csv(rows, k, out_root / "results.csv", summary=summary)
        write_tier_stats_json(batch_tag, summary, k,
                              out_root / "tier_stats.json")
        clean = compute_clean_baseline(cfg, k)
        write_summary_md(batch_tag, summary, clean, k,
                         out_root / "summary.md")
        section_exit("结果整合", _t)
        print(f"[batch] 整合完成：{len(rows)} 个原子实验 -> {out_root}")
