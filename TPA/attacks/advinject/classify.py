"""AdvInject classify stage: classify items by training interaction frequency."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from attacks.advinject.common import item_counts, load_meta
from training.run_tag import resolve_run_tag, write_latest_pointer

PROJECT_ROOT=Path(__file__).resolve().parents[2]

def classification_path(config):
    dataset=config.get("dataset","ml100k"); model=config.get("model",{}).get("name","wmf"); tag=resolve_run_tag(config)
    return PROJECT_ROOT/"attacks"/"advinject"/"data"/"rec_freq"/dataset/model/f"{tag}.json"

def classify(config):
    meta,_=load_meta(config); counts=item_counts(meta); values=np.asarray([counts.get(item,0) for item in range(meta["num_items"])],dtype=float)
    cfg=config.get("classification",{}); head=float(cfg.get("popular_percentile",95)); upper=float(cfg.get("torso_percentile",75)); lower=float(cfg.get("tail_percentile",50))
    head_bound=float(np.percentile(values,head)); upper_bound=float(np.percentile(values,upper)); lower_bound=float(np.percentile(values,lower))
    categories={"popular":[],"ordinary":[],"cold":[]}
    for item,count in enumerate(values):
        if count>head_bound: categories["popular"].append(item)
        elif count>lower_bound: categories["ordinary"].append(item)
        else: categories["cold"].append(item)
    payload={"dataset":config.get("dataset","ml100k"),"model":config.get("model",{}).get("name","wmf"),"basis":"training_interaction_count","counts":{str(i):int(v) for i,v in enumerate(values)},"categories":categories,"summary":{"num_users":meta["num_users"],"num_items":meta["num_items"],"appearing_items":int(np.count_nonzero(values)),"popular_count":len(categories["popular"]),"ordinary_count":len(categories["ordinary"]),"cold_count":len(categories["cold"]),"min_popular_count":int(min([values[i] for i in categories["popular"]],default=0))}}
    path=classification_path(config); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8"); write_latest_pointer(path.parent,resolve_run_tag(config)); return payload

if __name__=="__main__":
    import yaml
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="attacks/advinject/config.yaml"); args=parser.parse_args()
    with open(args.config,encoding="utf-8") as handle: print(classify(yaml.safe_load(handle)))
