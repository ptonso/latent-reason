from dataclasses import dataclass
from typing import Mapping, Literal, Optional, Any, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.vae.types import GenLogits, Context
from src.vae.config import GaussianReconConfig


class GaussianRecon(nn.Module):
    """
    Pixel recon + KL (+ optional perceptual)
    targets: x ∈ [-1,1]
    L_pix(xhat,x)  : {L2, L1, Huber} on [-1,1] images; reduced to scalar
    KL(mu,logvar)  : diagonal gaussian KL with free-nats; per-image sum, mean over batch.
    L_perc(xhat,x) : {L1, L2} on φ(hatx)-φ(x), compares features; reduced to scalar.
    
    total = pix_weight*L_pix + perc_weight*L_perc + beta*KL
    """
    def __init__(self, cfg: GaussianReconConfig):
        super().__init__()
        self.huber_delta = cfg.huber_delta
        self.recon_type  = cfg.recon_type

    def recon_loss(self, hatx: Tensor, x: Tensor) -> Tensor:
        rt = self.recon_type
        if   rt in ("l2","mse"):  per = F.mse_loss(hatx, x, reduction="none")
        elif rt in ("l1","mae"):  per = F.l1_loss (hatx, x, reduction="none")
        else:                     per = F.smooth_l1_loss(hatx, x, reduction="none", beta=self.huber_delta)
        return per.flatten(1).sum(1).mean()

    @torch.no_grad
    def mean_image(self, logits: GenLogits, ctx: Optional[Context] = None):
        """decoder outputs logits; tanh -> [-1,1], which is what LPIPS expects"""
        hatx = torch.tanh(logits.img)
        return hatx
