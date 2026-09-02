"""Stage model: retrain registered victims on clean plus fake interactions."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from models.registry import get_model_cls
from models.revisit_training import pair_loader
from attacks.advinject.common import canonical_config, load_meta, pairs_to_csr, attack_metrics

def make_config(config):
    from training.framework import TrainingConfig
    flat={}
    flat.update(config.get("model",{})); flat.update(config.get("training",{})); flat["device"]=config.get("training",{}).get("device","cpu")
    return TrainingConfig(overrides=flat)

def score_model(model, user_ids, n_items):
    device = getattr(model, "_device", torch.device("cpu"))
    users = torch.tensor(user_ids, dtype=torch.long, device=device)
    items = torch.arange(n_items, dtype=torch.long, device=device)
    return model.predict_full_ranking(users, items)


def _edge_index(pairs):
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _train_pair_model(model, name, meta, cfg, epochs):
    from models.revisit_training import sample_negative_items
    from training.epoch_log import log_train_line
    from training.timing import section_enter, section_exit
    loader = pair_loader(meta["train_pairs"], int(cfg.get("batch_size", 256)), True)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
    device = getattr(model, "_device", torch.device("cpu"))
    history = []
    for epoch in range(1, epochs + 1):
        _t_epoch = section_enter(f"Epoch {epoch}/{epochs}")
        model.train()
        total = 0.0
        count = 0
        for users, positives in loader:
            users, positives = users.to(device), positives.to(device)
            negatives = sample_negative_items(users, meta["user_items"], int(meta["num_items"]))
            optimizer.zero_grad(set_to_none=True)
            if name == "mf":
                bpr, reg = model._bpr_loss(users, positives, negatives.unsqueeze(1))
                loss = bpr + reg
            elif name == "ncf":
                loss = model.bpr_loss(users, positives, negatives)
            else:
                loss = model.margin_loss(users, positives, negatives)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name} training loss is non-finite")
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(users)
            count += len(users)
        avg = total / max(1, count)
        log_train_line(epoch, epochs, avg)
        entry = {"epoch": epoch, "loss": avg,
                 "epoch_seconds": section_exit(
                     f"Epoch {epoch}/{epochs}", _t_epoch)}
        history.append(entry)
    return history

def retrain_victim(meta,config):
    name=config.get("model",{}).get("name","wmf").lower(); cfg=make_config(config)
    model_cls=get_model_cls(name)
    edge_index = _edge_index(meta["train_pairs"]) if name == "lightgcn" else None
    model=model_cls(cfg,int(meta["num_users"]),int(meta["num_items"]),edge_index)
    epochs=int(config.get("training",{}).get("epochs",1)); device=getattr(model,"_device",torch.device("cpu"))
    if name=="itemcf":
        history=[model.fit(meta["train_pairs"])]
    elif name=="wmf":
        from models.wmf.dataset import WMFDataset
        from training.epoch_log import log_train_line
        from training.timing import section_enter, section_exit
        dataset=WMFDataset(meta["train_pairs"],alpha=float(cfg.get("alpha",40)),epsilon=float(cfg.get("epsilon",1e-8)),scheme=cfg.get("confidence_scheme","minimal"))
        batch=(dataset.users,dataset.items,dataset.conf,dataset.p,dataset.user_obs,dataset.item_obs)
        history=[]
        for epoch in range(1, epochs + 1):
            _t_epoch = section_enter(f"Epoch {epoch}/{epochs}")
            res=model.train_step(batch)
            log_train_line(epoch, epochs, res["loss"])
            entry={"epoch": epoch, "loss": res["loss"],
                   "epoch_seconds": section_exit(
                       f"Epoch {epoch}/{epochs}", _t_epoch)}
            history.append(entry)
    elif name in {"itemae", "multvae"}:
        matrix=pairs_to_csr(meta["train_pairs"],meta["num_users"],meta["num_items"]).toarray()
        history=[model.fit(torch.tensor(matrix,dtype=torch.float32),epochs=epochs,lr=float(cfg.get("lr",1e-3)))]
    elif name == "lightgcn":
        from models.lightgcn.dataset import LightGCNDataset
        from torch.utils.data import DataLoader
        from training.epoch_log import log_train_line
        from training.timing import section_enter, section_exit
        dataset=LightGCNDataset(meta["train_pairs"], int(meta["num_items"]), meta["user_items"], int(meta["num_users"]), neg_ratio=int(cfg.get("neg_ratio", 1)))
        loader=DataLoader(dataset, batch_size=int(cfg.get("batch_size", 256)), shuffle=True)
        history=[]
        for epoch in range(1, epochs + 1):
            _t_epoch = section_enter(f"Epoch {epoch}/{epochs}")
            losses=[float(model.train_step(batch)["loss"]) for batch in loader]
            avg=sum(losses)/max(1,len(losses))
            log_train_line(epoch, epochs, avg)
            history.append({"epoch": epoch, "loss": avg,
                            "epoch_seconds": section_exit(
                                f"Epoch {epoch}/{epochs}", _t_epoch)})
    elif name in {"mf", "ncf", "cml"}:
        history=_train_pair_model(model, name, meta, cfg, epochs)
    else:
        raise ValueError(f"unsupported victim model: {name}")
    return model,history

def evaluate(config, poisoned_meta=None, targets=None):
    config=canonical_config(config)
    meta,_=load_meta(config) if poisoned_meta is None else (poisoned_meta,None)
    targets=[int(x) for x in (targets if targets is not None
                              else config.get("attack",{}).get("target_items",{}).get("ids",[]))]
    model,history=retrain_victim(meta,config); users=sorted({u for u,_ in meta.get("test_pairs",[]) if u<meta["num_users"]})
    scores=score_model(model,users,meta["num_items"]).detach().cpu().numpy() if users else []
    train_items=meta.get("user_items",[set() for _ in range(meta["num_users"])])
    metrics=attack_metrics(scores,targets,[train_items[u] for u in users],int(config.get("evaluation",{}).get("k",50)))
    return {"model":config.get("model",{}).get("name","wmf"),"metrics":metrics,"history":history}

if __name__=="__main__":
    from training.config_utils import load_config
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="attacks/advinject/config.yaml"); parser.add_argument("--meta",required=True); args=parser.parse_args()
    config=load_config(args.config)
    import pickle
    with open(args.meta,"rb") as handle: meta=pickle.load(handle)
    print(evaluate(config,meta))


