import unittest, torch
from models.multvae.model import MultVAEModel
from training.framework import TrainingConfig
class TestMultVAE(unittest.TestCase):
 def test_forward_and_loss(self):
  m=MultVAEModel(TrainingConfig(overrides={"device":"cpu","hidden_dim":3,"latent_dim":2}),3,4)
  x=torch.zeros(2,4); x[0,1]=1
  logits,mu,logvar=m.forward(x); self.assertEqual(tuple(logits.shape),(2,4))
  loss=m.loss(x,logits,mu,logvar); self.assertTrue(torch.isfinite(loss))
if __name__=='__main__': unittest.main()