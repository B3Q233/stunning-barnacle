import unittest, torch
from models.cml.model import CMLModel
from training.framework import TrainingConfig
class TestCML(unittest.TestCase):
 def test_distance_and_loss(self):
  m=CMLModel(TrainingConfig(overrides={"device":"cpu","emb_dim":4}),3,5)
  u=torch.tensor([0,1]); p=torch.tensor([1,2]); n=torch.tensor([3,4])
  self.assertEqual(tuple(m.forward(u,p).shape),(2,))
  self.assertTrue(torch.isfinite(m.margin_loss(u,p,n)))
if __name__=='__main__': unittest.main()