"""批量投毒攻击编排 CLI。"""
if __name__ == "__main__":
    import argparse
    import shutil
    from pathlib import Path

    import yaml

    from attacks.batch.aggregate import (
        build_results_rows, compute_clean_baseline, tier_summary,
        write_results_csv, write_summary_md)
    from attacks.batch.generator import load_batch_config
    from attacks.batch.runner import (
        ensure_classify_cache, run_atomic, run_batch, staging_dir)
    from attacks.batch.utils import group_name, public_rec_freq_path
    from training.run_tag import resolve_run_tag

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
    out_root = Path(cfg.get("output", {}).get(
        "dir", "attacks/batch/output")) / batch_tag
    configs_dir, runs_root = out_root / "configs", out_root / "runs"
    k = cfg.get("evaluation", {}).get("k", 10)
    group = group_name(cfg)

    if args.mode in ("generate", "all"):
        if args.skip_classify and not public_rec_freq_path(cfg).exists():
            raise FileNotFoundError(
                f"缓存不存在：{public_rec_freq_path(cfg)}"
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
                shutil.move(str(src), str(dst))
    if args.mode in ("aggregate", "all") and not args.dry_run:
        rows = build_results_rows(runs_root, group, cfg, k)
        write_results_csv(rows, k, out_root / "results.csv")
        clean = compute_clean_baseline(cfg, k)
        write_summary_md(batch_tag, tier_summary(rows, k), clean, k,
                         out_root / "summary.md")
        print(f"[batch] 整合完成：{len(rows)} 个原子实验 -> {out_root}")
