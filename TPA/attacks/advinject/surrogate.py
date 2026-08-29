"""Compatibility facade for surrogate objectives."""
from attacks.advinject.train_surrogate import compute_adversarial_gradient, mult_ce_loss

def target_loss(model, clean_history, targets, temperature=1.0, fake_history=None):
    import torch
    scale=max(float(temperature),1e-6); logits=model(clean_history); labels=torch.zeros_like(logits); labels[:,targets]=1.0
    loss=-(torch.nn.functional.log_softmax(logits/scale,dim=1)*labels).sum(dim=1).mean()
    if fake_history is not None:
        fake_logits=model(fake_history); fake_labels=torch.zeros_like(fake_logits); fake_labels[:,targets]=1.0
        loss=loss+0.1*-(torch.nn.functional.log_softmax(fake_logits/scale,dim=1)*fake_labels).sum(dim=1).mean()
    return loss
