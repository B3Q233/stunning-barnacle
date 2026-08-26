import torch
from models.cml.dataset import CMLModelDataLoader
from models.cml.model import CMLModel
from models.revisit_training import prepare_experiment_dir, pair_loader, train_bpr_epoch, ranking_report, save_training_artifacts

def run(config):
    out_dir=prepare_experiment_dir("cml",config)
    loader=CMLModelDataLoader(config)
    device=torch.device(config.get("training",{}).get("device","cpu"))
    model=CMLModel(config,loader.num_users,loader.num_items).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.get("training",{}).get("lr",1e-3))
    generator=torch.Generator().manual_seed(42)
    history=[]
    for _ in range(int(config.get("training",{}).get("epochs",1))):
        loss=train_bpr_epoch(model,pair_loader(loader.train_pairs,config.get("training",{}).get("batch_size",256)),loader.meta,loader.num_items,optimizer,device,generator)
        history.append({"loss":loss})
    metrics=ranking_report(model,loader.meta,config.get("evaluation",{}).get("k",20),device)
    save_training_artifacts(out_dir,model,history,metrics)
    print(metrics)
    return model,metrics
