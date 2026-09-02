"""Stage model: retrain a registered victim on clean and poisoned data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from attacks.advinject.common import canonical_config, load_poisoned_meta, load_meta, save_json
from attacks.advinject.evaluate import evaluate, retrain_victim
from attacks.advinject.generate import attack_paths

def save_checkpoint(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save({"model_state_dict": model.state_dict()}, path)

def fit(config, generated=None):
    config=canonical_config(config)
    if generated is None:
        data_dir, output_dir = attack_paths(config)
        generated = {"data_dir": str(data_dir), "output_dir": str(output_dir),
                     "targets": config.get("attack", {}).get("target_items", {}).get("ids", [])}
    else:
        data_dir, output_dir = Path(generated["data_dir"]), Path(generated["output_dir"])
    poisoned_meta = load_poisoned_meta(data_dir)
    targets = [int(item) for item in generated["targets"]]
    poisoned = evaluate(config, poisoned_meta, targets)
    clean_meta, _ = load_meta(config)
    clean = evaluate(config, clean_meta, targets)
    output_dir.mkdir(parents=True, exist_ok=True)
    poisoned_model, _ = retrain_victim(poisoned_meta, config)
    clean_model, _ = retrain_victim(clean_meta, config)
    save_checkpoint(poisoned_model, output_dir / "checkpoints" / "poisoned.pt")
    save_checkpoint(clean_model, output_dir / "checkpoints" / "clean.pt")
    report = {
        "model": config.get("model", {}).get("name", "wmf"),
        "targets": targets,
        "clean": clean["metrics"],
        "poisoned": poisoned["metrics"],
        "delta": {key: poisoned["metrics"].get(key, float("nan")) - clean["metrics"].get(key, float("nan")) for key in poisoned["metrics"]},
    }
    save_json(output_dir / "attack_comparison.json", report)
    markdown = ["# AdvInject 攻击对比", "", f"- victim model: `{report['model']}`", f"- target items: `{targets}`", "", "| 指标 | Clean | Poisoned | Delta |", "|---|---:|---:|---:|"]
    for key in report["poisoned"]:
        markdown.append(f"| {key} | {report['clean'].get(key, float('nan')):.6f} | {report['poisoned'].get(key, float('nan')):.6f} | {report['delta'].get(key, float('nan')):.6f} |")
    (output_dir / "attack_comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return {"data_dir": str(data_dir), "output_dir": str(output_dir), "clean": clean, "poisoned": poisoned, "comparison": report}

if __name__ == "__main__":
    from training.config_utils import load_config
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="attacks/advinject/config.yaml"); args=parser.parse_args()
    print(fit(load_config(args.config)))
