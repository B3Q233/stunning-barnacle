"""AdvInject attack registry."""
from models.registry import get_model_cls, get_dataset_cls, load_model_config

def resolve_model(name):
    return get_model_cls(name)
