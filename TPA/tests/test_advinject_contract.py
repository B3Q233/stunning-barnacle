import unittest
import tempfile
from pathlib import Path
import torch
from attacks.advinject.data import select_targets, build_fake_tensor, poisoned_meta
from attacks.advinject.surrogate import target_loss
from models.itemae.model import ItemAEModel

class AdvInjectContractTest(unittest.TestCase):
    def setUp(self):
        self.meta={"num_users":3,"num_items":6,"train_pairs":[(0,0),(0,1),(1,1),(1,2),(2,3)],"test_pairs":[],"user_items":{0:{0,1},1:{1,2},2:{3}}}
    def test_attack_config_accepts_null_targets(self):
        from attacks.advinject.common import AttackConfig
        config = AttackConfig.from_dict({"attack": {"target_items": None}})
        self.assertEqual(config.n_target_items, 5)
    def test_target_selection_validates_ids(self):
        self.assertEqual(select_targets(self.meta,{"target_strategy":"specified","target_items":[4]}),[4])
        with self.assertRaises(ValueError): select_targets(self.meta,{"target_strategy":"specified","target_items":[8]})
    def test_fake_data_contains_target(self):
        fake=build_fake_tensor(self.meta,[4],{"num_fake_users":2,"base_items":2},42)
        self.assertEqual(tuple(fake.shape),(2,6))
        self.assertTrue(torch.all(fake[:,4] == 1))
    def test_surrogate_loss_has_gradient(self):
        model=ItemAEModel({"device":"cpu","hidden_dim":4},3,6)
        history=torch.zeros(3,6); history[0,0]=1; history[1,1]=1
        loss=target_loss(model,history,[4]); loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertTrue(any(p.grad is not None for p in model.parameters()))
    def test_poisoned_meta_appends_fake_users(self):
        result,pairs=poisoned_meta(self.meta,torch.tensor([[1,0,0,0,1,0.],[0,0,1,0,1,0.]]))
        self.assertEqual(result["num_users"],5); self.assertEqual(len(pairs),4)

if __name__ == "__main__": unittest.main()
