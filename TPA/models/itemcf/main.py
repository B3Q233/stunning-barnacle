import argparse
import sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from models.itemcf.train import run

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="itemcf ????")
    parser.add_argument("--config", default="models/itemcf/config.yaml")
    args=parser.parse_args()
    with open(args.config, encoding="utf-8") as f: config=yaml.safe_load(f)
    run(config)
