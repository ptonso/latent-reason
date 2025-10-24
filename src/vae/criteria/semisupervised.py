from dataclasses import dataclass
#from typing import Mapping, Literal, Optional, Any, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.vae.config import SemiSupervisedConfig

class SemiSupervisedLoss:
    """
    Pixel Reconstruction + KL + SemiSupervision
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled: bool = False
    
    def __call__(self, mu, x, labels, is_labeled):
        mask = torch.tensor(is_labeled, device=x.device)

        if mask.sum() > 0:
            mu_labeled = mu[mask]
            masked_labels = labels[mask]
            BCE_loss = torch.nn.BCEWithLogitsLoss()
            semisuper_loss = BCE_loss(mu_labeled, masked_labels)

        else:
            semisuper_loss = torch.tensor(0., device=x.device)

        return semisuper_loss