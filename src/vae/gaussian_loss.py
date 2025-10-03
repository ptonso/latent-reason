from dataclasses import dataclass
from typing import Mapping, Literal, Optional, Any, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.vae.types import GenLogits, GenTargets, Context
from src.vae.base import BaseCriterion, LossDict
from src.vae.config import GaussianCriterionConfig
from src.vae.perceptual import PerceptualLoss


class GaussianCriterionConfig(BaseCriterion):
    """
    Pixel recon + KL (+ optional perceptual)
    targets: x ∈ [-1,1]
    L_pix(xhat,x)  : {L2, L1, Huber} on [-1,1] images; reduced to scalar
    KL(mu,logvar)  : diagonal gaussian KL with free-nats; per-image sum, mean over batch.
    L_perc(xhat,x) : {L1, L2} on φ(hatx)-φ(x), compares features; reduced to scalar.
    
    total = pix_weight*L_pix + perc_weight*L_perc + beta*KL
    """
    def __init__(self, cfg: GaussianCriterionConfig):
        super().__init__()
        self.cfg = cfg
        self.beta = float(cfg.beta)

        self.perc = PerceptualLoss(cfg.perc)


    def _recon(self, hatx: Tensor, x: Tensor) -> Tensor:
        rt = self.cfg.recon_type
        if   rt in ("l2","mse"):  per = F.mse_loss(hatx, x, reduction="none")
        elif rt in ("l1","mae"):  per = F.l1_loss (hatx, x, reduction="none")
        else:                     per = F.smooth_l1_loss(hatx, x, reduction="none", beta=self.cfg.huber_delta)
        pix = per.flatten(1).sum(1).mean()

        perc = self.perc(hatx, x)
        return float(self.cfg.pix_weight) * pix + float(self.cfg.perc.weight) * perc


    def _kld(self, mu: Tensor, logvar: Tensor, free_nats: float) -> Tensor:
        kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
        kl = torch.clamp(kl - free_nats, min=0.0)
        return kl.sum(1).mean()


    def init_perc(
        self,
        *,
        device: Optional[str] = None,
        encoder: Optional[nn.Module] = None,
        features_fn: Optional[Callable[[Tensor], Dict[str, Tensor]]] = None
    ) -> None:
        self.perc.init(device=device, encoder=encoder, features_fn=features_fn)


    def forward(
        self, 
        logits: GenLogits, 
        target: Any, 
        ctx: Optional[Context] = None
    ) -> LossDict:
        
        x    = target.img if hasattr(target, "img") else target
        hatx = torch.tanh(logits.img)
        rec  = self._recon(hatx, x)
        
        if ctx is None or "mu" not in ctx:
            return {"loss": rec, "loss/recon": rec, "loss/kld": torch.zeros((), device=logits.img.device)}
        
        kld   = self._kld(ctx["mu"], ctx["logvar"], ctx.get("free_n", 0.0))
        total = rec + self.beta * kld
        return {"loss": total, "loss/recon": rec, "loss/kld": kld}


    @torch.no_grad
    def predict(self, logits: GenLogits, ctx: Optional[Context] = None):
        hatx = torch.tanh(logits.img)
        return [(hatx + 1.0) * 0.5]