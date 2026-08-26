from models.revisit_common import load_meta

class CMLModelDataLoader:
    def __init__(self, config):
        self.meta=load_meta(config, "cml")
        self.num_users=self.meta["num_users"]
        self.num_items=self.meta["num_items"]
        self.train_pairs=self.meta["train_pairs"]
        self.test_pairs=self.meta["test_pairs"]
        self.user_items=self.meta["user_items"]
    def get_init_params(self):
        return {"num_users":self.num_users,"num_items":self.num_items}
    def interaction_matrix(self):
        import torch
        x=torch.zeros(self.num_users,self.num_items)
        for u,i in self.train_pairs: x[u,i]=1
        return x
