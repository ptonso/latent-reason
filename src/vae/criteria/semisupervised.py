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
        self.cfg = cfg
    
    def __call__(
        self, 
        mu: torch.Tensor, 
        labels: torch.Tensor, 
        is_labeled: torch.Tensor
        ) -> torch.Tensor:

        mask = torch.tensor(is_labeled, device=mu.device)

        if mask.sum() > 0:
            mu_labeled = mu[mask]
            masked_labels = labels[mask]
            BCE_loss = torch.nn.BCEWithLogitsLoss()
            semisuper_loss = BCE_loss(mu_labeled, masked_labels)
        else:
            semisuper_loss = torch.tensor(0., device=mu.device)

        return semisuper_loss

