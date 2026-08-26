import unittest
import torch
from models.itemcf.model import ItemCFModel
from training.framework import TrainingConfig

class ItemCFTest(unittest.TestCase):
    def test_fit_and_rank(self):
        cfg=TrainingConfig(overrides={"device":"cpu","topk":10})
        model=ItemCFModel(cfg, 3, 4)
        pairs=[(0,0),(0,1),(1,1),(1,2),(2,2)]
        model.fit(pairs)
        scores=model.predict_full_ranking(torch.tensor([0,1]))
        self.assertEqual(tuple(scores.shape),(2,4))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertEqual(tuple(model.get_item_embeddings().shape),(4,4))

    def test_dynamic_user_bounds(self):
        cfg=TrainingConfig(overrides={"device":"cpu"})
        model=ItemCFModel(cfg, 2, 3)
        with self.assertRaises(IndexError):
            model.score_users(torch.tensor([2]), torch.tensor([[0,1,0.]]))

if __name__ == "__main__": unittest.main()