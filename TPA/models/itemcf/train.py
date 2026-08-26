"""ItemCF training entry point."""
import argparse
import yaml
import torch
from models.itemcf.dataset import ItemCFDataLoader
from models.itemcf.model import ItemCFModel
from models.revisit_training import prepare_experiment_dir, ranking_report, save_training_artifacts

def run(config):
    out_dir=prepare_experiment_dir("itemcf",config)
    loader=ItemCFDataLoader(config)
    model=ItemCFModel(config,loader.num_users,loader.num_items)
    model.fit(loader.train_pairs)
    metrics=ranking_report(model,loader.meta,int(config.get("evaluation",{}).get("k",20)),torch.device("cpu"))
    save_training_artifacts(out_dir,model,[{"loss":0.0}],metrics)
    print(metrics)
    return model,metrics

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="models/itemcf/config.yaml")
    args=parser.parse_args()
    with open(args.config,encoding="utf-8") as handle:
        run(yaml.safe_load(handle))
