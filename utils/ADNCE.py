import torch
import torch.nn.functional as F
from torch import nn
import math

__all__ = ['ADNCE', 'adnce']


class ADNCE(nn.Module):

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired', mu=0.2, sigma=1.0):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.mu = mu
        self.sigma = sigma

    def forward(self, query, positive_key, negative_keys=None):
        return adnce(query, positive_key, negative_keys,
                     temperature=self.temperature,
                     reduction=self.reduction,
                     negative_mode=self.negative_mode,
                     mu=self.mu,
                     sigma=self.sigma)


def adnce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired', mu=0.1,
          sigma=1.0):
    # 检查输入维度
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'。")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'paired'。")

    # Check sample number matching
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:

        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        weight = 1. / (sigma * math.sqrt(2 * math.pi)) * torch.exp(
            - (negative_logits - mu) ** 2 / (2 * math.pow(sigma, 2)))
        weight = weight / weight.mean(dim=-1, keepdim=True)
        negative_logits = negative_logits * weight.detach()

        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)

    else:
        logits = query @ transpose(positive_key)

        labels = torch.arange(len(query), device=query.device)

        temp_logits = logits.clone()
        neg_logits = temp_logits.fill_diagonal_(0)

        weight = (1 / (sigma * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * ((neg_logits - mu) / sigma) ** 2)
        weight = weight / weight.mean(dim=-1, keepdim=True)

        adjusted_logits = logits * weight.detach()

        diagonal = torch.diag(logits)
        adjusted_logits = adjusted_logits.clone()
        diag_matrix = torch.diag_embed(diagonal)
        adjusted_logits.fill_diagonal_(0)
        adjusted_logits += diag_matrix

        logits = adjusted_logits

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]