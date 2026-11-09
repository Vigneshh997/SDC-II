"""Asymmetric multi-label classification and symmetric all-pair InfoNCE."""

import torch
import torch.nn.functional as F
from torch import nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, labels):
        if logits.shape != labels.shape or not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("ASL requires binary targets matching logits")
        positive = logits.float().sigmoid()
        negative = (1 - positive + self.clip).clamp(max=1)
        loss = labels * positive.clamp_min(self.eps).log() * (1 - positive).pow(self.gamma_pos)
        loss += (1 - labels) * negative.clamp_min(self.eps).log() * (1 - negative).pow(self.gamma_neg)
        return -loss.sum(-1).mean()


def contrastive_loss(image, text, metadata, temperature=0.07):
    if temperature <= 0 or image.shape[0] < 2:
        raise ValueError("InfoNCE needs a positive temperature and at least two different examinations")
    features = [F.normalize(f.float(), dim=-1) for f in (image, text, metadata)]
    if any(f.shape != features[0].shape for f in features):
        raise ValueError("Contrastive feature shapes must match")
    targets = torch.arange(image.shape[0], device=image.device)
    losses = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        similarity = features[a] @ features[b].T / temperature
        losses.extend([F.cross_entropy(similarity, targets), F.cross_entropy(similarity.T, targets)])
    return torch.stack(losses).mean()
