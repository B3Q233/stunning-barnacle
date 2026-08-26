import unittest, torch
from models.itemae.model import ItemAEModel
from training.framework import TrainingConfig
class TestItemAE(unittest.TestCase):
 def test_reconstruct_and_update(self):
  m=ItemAEModel(TrainingConfig(overrides={"device":"cpu","hidden_dim":3}),3,4)
  x=torch.zeros(2,4); x[0,1]=1; x[1,2]=1
  out=m.forward(x); self.assertEqual(tuple(out.shape),(2,4))
  loss=m.reconstruction_loss(x); loss.backward(); self.assertTrue(torch.isfinite(loss))
if __name__=='__main__': unittest.main()