import unittest, torch
from models.ncf.model import NCFModel
from training.framework import TrainingConfig
class TestNCF(unittest.TestCase):
 def test_forward_loss_and_bounds(self):
  m=NCFModel(TrainingConfig(overrides={"device":"cpu"}),3,4)
  u=torch.tensor([0,1]); p=torch.tensor([1,2]); n=torch.tensor([3,0])
  self.assertEqual(tuple(m.forward(u,p).shape),(2,))
  self.assertTrue(torch.isfinite(m.bpr_loss(u,p,n)))
  with self.assertRaises(IndexError): m.forward(torch.tensor([3]),torch.tensor([0]))
if __name__=='__main__': unittest.main()