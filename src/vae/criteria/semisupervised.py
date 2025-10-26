from dataclasses import dataclass
from typing import Mapping, Literal, Optional, Any, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.vae.config import SemiSupervisedConfig

class SemiSupervisedLoss(nn.Module):
    """
    Latent-wise binary cross entropy in mu
    """
    def __init__(self, cfg: SemiSupervisedConfig):
        super().__init__()
        self.cfg = cfg
        self.weights = cfg.weight

    def active(self, labels) -> bool:
        return self.cfg.cap > 0 and labels is not None
    
    def forward(self, mu: Tensor, labels: Tensor | None) -> Tensor:
        if not self.active(labels):
            return mu.new_zeros(())
        labels = labels.float()
        if mu.size(1) != labels.size(1):
            mu = mu[:, :labels.size(1)]
        loss = F.binary_cross_entropy_with_logits(mu, labels, reduction="mean")
        return loss * self.weight

