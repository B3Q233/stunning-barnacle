"""Surrogate trainers matching revisit_adv_rec ItemAE and WeightedMF semantics."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from attacks.advinject.common import canonical_config

@dataclass
class SurrogateResult:
    loss: float
    gradient: np.ndarray
    model_name: str
    epochs: int
    unroll_steps: int

class ItemAESurrogate(nn.Module):
    def __init__(self,input_dim,hidden_dims=(256,128),weight=20.0):
        super().__init__(); q=[input_dim,*hidden_dims]; p=list(reversed(q))
        self.q_layers=nn.ModuleList([nn.Linear(a,b) for a,b in zip(q[:-1],q[1:])])
        self.p_layers=nn.ModuleList([nn.Linear(a,b) for a,b in zip(p[:-1],p[1:])]); self.weight=weight
    def forward(self,data):
        h=data
        for layer in self.q_layers: h=torch.tanh(layer(h))
        for index,layer in enumerate(self.p_layers):
            h=layer(h)
            if index<len(self.p_layers)-1: h=torch.tanh(h)
        return h
    def loss(self,data):
        logits=self.forward(data); weights=torch.where(data>0,self.weight,1.0); return (weights*(data-logits).pow(2)).sum(dim=1).mean()

class WeightedMFSurrogate(nn.Module):
    def __init__(self,n_users,n_items,dim=128):
        super().__init__(); self.P=nn.Parameter(torch.randn(n_users,dim)*0.1); self.Q=nn.Parameter(torch.randn(n_items,dim)*0.1)
    def forward(self,data): return self.P@self.Q.T
    def loss(self,data,weight_alpha=20.0,l2=1e-5):
        weights=1.0+(weight_alpha-1.0)*data; observed=weights*(data-self.forward(data)).pow(2)
        return observed.sum(dim=1).mean()+l2*(self.P.pow(2).mean()+self.Q.pow(2).mean())

def mult_ce_loss(logits,data):
    log_probs=F.log_softmax(logits,dim=-1); instance_data=data.sum(dim=1).clamp_min(1e-12)
    return ((-log_probs*data).sum(dim=1)/instance_data).mean()

def differentiable_sgd(model,data,steps,lr,loss_fn):
    params=tuple(model.parameters())
    for _ in range(int(steps)):
        loss=loss_fn(model,data,params)
        grads=torch.autograd.grad(loss,params,create_graph=True,allow_unused=False)
        params=tuple(param-lr*grad for param,grad in zip(params,grads))
    return params

def functional_itemae(model,data,params):
    state=dict(model.named_parameters()); index=0; h=data
    for layer in model.q_layers:
        weight,bias=params[index],params[index+1]; index+=2; h=torch.tanh(F.linear(h,weight,bias))
    for layer_index,layer in enumerate(model.p_layers):
        weight,bias=params[index],params[index+1]; index+=2; h=F.linear(h,weight,bias)
        if layer_index<len(model.p_layers)-1: h=torch.tanh(h)
    weights=torch.where(data>0,model.weight,1.0); return (weights*(data-h).pow(2)).sum(dim=1).mean()

def functional_wmf(model,data,params,weight_alpha,l2):
    P,Q=params; scores=P@Q.T; weights=1.0+(weight_alpha-1.0)*data
    return (weights*(data-scores).pow(2)).sum(dim=1).mean()+l2*(P.pow(2).mean()+Q.pow(2).mean())

def compute_adversarial_gradient(train_csr, fake, n_items, targets, config):
    """训练 surrogate，并对 fake data 求外层目标梯度。"""
    config = canonical_config(config)
    surrogate_cfg = config.get("surrogate", {})
    sur_tr = surrogate_cfg.get("training", {})
    name = str(surrogate_cfg.get("name", "item_ae")).lower()
    device = torch.device(config.get("training", {}).get("device", "cpu"))
    clean = torch.tensor(train_csr.toarray(), dtype=torch.float32, device=device)
    fake_tensor = torch.tensor(fake, dtype=torch.float32, device=device, requires_grad=True)
    data = torch.cat([clean, fake_tensor], dim=0)
    n_users = clean.shape[0]
    epochs = int(sur_tr.get("epochs", 50))
    unroll_steps = int(sur_tr.get("unroll_steps", 5))
    diff_steps = max(1, min(epochs, unroll_steps or epochs))
    pre_steps = max(0, epochs - diff_steps)
    lr = float(sur_tr.get("lr", 1e-3))
    l2 = float(sur_tr.get("weight_decay", 1e-6))

    if name in {"item_ae", "itemae", "sur-itemae"}:
        model = ItemAESurrogate(data.shape[0], tuple(sur_tr.get("hidden_dims", [256, 128])), float(sur_tr.get("weight_alpha", 20.0))).to(device)
        def ordinary_loss():
            return model.loss(data.T)
        def functional_loss(params):
            return functional_itemae(model, data.T, params)
        def predictions(params):
            return functional_itemae_logits(model, data.T, params).T
    elif name in {"wmf", "weightedmf", "wmf_sgd", "sur-weightedmf-sgd", "sur-weightedmf-als"}:
        dim = int(sur_tr.get("dim", sur_tr.get("hidden_dims", [128])[0]))
        model = WeightedMFSurrogate(data.shape[0], n_items, dim).to(device)
        weight_alpha = float(sur_tr.get("weight_alpha", 20.0))
        def ordinary_loss():
            return model.loss(data, weight_alpha, l2)
        def functional_loss(params):
            return functional_wmf(model, data, params, weight_alpha, l2)
        def predictions(params):
            return functional_wmf_scores(model, data, params)
    else:
        raise ValueError(f"unsupported surrogate: {name}")

    ordinary_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    for _ in range(pre_steps):
        ordinary_optimizer.zero_grad(set_to_none=True)
        loss = ordinary_loss()
        loss.backward()
        ordinary_optimizer.step()

    params = tuple(model.parameters())
    for _ in range(diff_steps):
        inner_loss = functional_loss(params)
        grads = torch.autograd.grad(inner_loss, params, create_graph=True)
        params = tuple(param - lr * grad for param, grad in zip(params, grads))

    score = predictions(params)
    target = torch.zeros_like(score)
    target[:, targets] = 1.0
    outer_loss = mult_ce_loss(score[:n_users], target[:n_users])
    gradient = torch.autograd.grad(outer_loss, fake_tensor)[0]
    record = {
        "loss": float(outer_loss.detach().item()),
        "model_name": name,
        "epochs": epochs,
        "pretrain_steps": pre_steps,
        "unroll_steps": diff_steps,
        "gradient_l2": float(gradient.detach().norm().item()),
    }
    return gradient.detach().cpu().numpy(), record
def functional_itemae_logits(model,data,params):
    index=0; h=data
    for layer in model.q_layers:
        h=torch.tanh(F.linear(h,params[index],params[index+1])); index+=2
    for layer_index,layer in enumerate(model.p_layers):
        h=F.linear(h,params[index],params[index+1]); index+=2
        if layer_index<len(model.p_layers)-1: h=torch.tanh(h)
    return h

def functional_wmf_scores(model,data,params): return params[0]@params[1].T



