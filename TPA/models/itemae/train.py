import torch
from models.itemae.dataset import ItemAEModelDataLoader
from models.itemae.model import ItemAEModel
from models.revisit_training import prepare_experiment_dir, ranking_report, save_training_artifacts

def run(config):
    out_dir=prepare_experiment_dir("itemae",config)
    loader=ItemAEModelDataLoader(config)
    model=ItemAEModel(config,loader.num_users,loader.num_items)
    history=[model.fit(loader.interaction_matrix(),int(config.get("training",{}).get("epochs",1)),float(config.get("training",{}).get("lr",1e-3)))]
    metrics=ranking_report(model,loader.meta,int(config.get("evaluation",{}).get("k",20)),torch.device("cpu"))
    save_training_artifacts(out_dir,model,history,metrics)
    print(metrics)
    return model,metrics
