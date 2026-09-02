"""AdvInject classify -> data -> model pipeline."""
from __future__ import annotations
import argparse
import json
from attacks.advinject.common import canonical_config
from attacks.advinject.classify import classify
from attacks.advinject.generate import generate
from attacks.advinject.fit import fit
from attacks.advinject.generate import attack_paths
from training.run_tag import read_latest_tag, resolve_run_tag

def run(config, mode=None):
    config=canonical_config(config)
    mode=mode or config.get("mode","all")
    result={}
    if mode in {"classify","all"}: result["classify"]=classify(config)
    generated=None
    if mode in {"data","all"}:
        generated=generate(config)
        result["data"]=generated
    if mode == "model":
        data_root, output_root = attack_paths(config)
        tag = resolve_run_tag(config)
        if not config.get("run_tag"):
            tag = read_latest_tag(data_root.parent) or tag
        data_dir = data_root.parent / tag
        target_path = data_dir / "target_items.json"
        if not (data_dir / "meta.pkl").exists() or not target_path.exists():
            raise FileNotFoundError(f"未找到可用于 model 阶段的投毒数据: {data_dir}")
        targets = json.loads(target_path.read_text(encoding="utf-8"))["target_items"]
        generated={"data_dir":str(data_dir),"output_dir":str(output_root.parent / tag),"targets":targets}
    if mode in {"model","all"}: result["model"]=fit(config,generated)
    return result

if __name__=="__main__":
    from training.config_utils import load_config
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="attacks/advinject/config.yaml"); parser.add_argument("--mode",choices=["classify","data","model","all"],default=None); args=parser.parse_args()
    print(run(load_config(args.config),args.mode))
