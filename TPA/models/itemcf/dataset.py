"""ItemCF 数据载入器。"""
from models.revisit_common import load_meta

class ItemCFDataLoader:
    def __init__(self, config):
        self.config=config
        self.meta=load_meta(config, "itemcf")
        self.num_users=self.meta["num_users"]
        self.num_items=self.meta["num_items"]
        self.train_pairs=self.meta["train_pairs"]
        self.test_pairs=self.meta["test_pairs"]
        self.user_items=self.meta["user_items"]

    def get_init_params(self):
        return {"num_users": self.num_users, "num_items": self.num_items}