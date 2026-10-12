import torch
from torch import nn 
from torch.nn import functional as F

def info_nce(q, k, temp=0.07):
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = q @ k.T / temp
    labels = torch.arange(q.shape[0], device=q.device)
    return F.cross_entropy(logits, labels)

