from dataclasses import dataclass
from typing import Mapping, Literal, Optional, Any
import torch
import torch.nn.functional as F
from torch import Tensor

from src.vae.types import GenLogits, GenTargets, Context
from src.vae.base import BaseCriterion, LossDict
from src.vae.config import BetaVAECriterionConfig


class BetaVAECriterion(BaseCriterion):
    """Recon loss + KL."""
    def __init__(self, cfg: BetaVAECriterionConfig):
        super().__init__()
        self.cfg = cfg
        self.beta = cfg.beta

    def _recon(self, recon: Tensor, x: Tensor) -> Tensor:
        rt = self.cfg.recon_type
        if   rt in ("l2","mse"):   per = F.mse_loss(recon, x, reduction="none")
        elif rt in ("l1","mae"):   per = F.l1_loss(recon, x, reduction="none")
        else:                      per = F.smooth_l1_loss(recon, x, reduction="none", beta=self.cfg.huber_delta)
        return per.flatten(1).sum(1).mean()

    def _kld(self, mu: Tensor, logvar: Tensor, free_nats: float) -> Tensor:
        kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
        kl = torch.clamp(kl - free_nats, min=0.0)
        return kl.sum(1).mean()

    def forward(
        self, 
        logits: GenLogits, 
        target: Any, 
        ctx: Optional[Context] = None
    ) -> LossDict:
        
        x   = target.img if isinstance(target, GenTargets) else target
        rec = self._recon(logits.img, x)
        
        if ctx is None or "mu" not in ctx:
            return {"loss": rec, "loss/recon": rec, "loss/kld": torch.zeros((), device=logits.img.device)}
        
        kld   = self._kld(ctx["mu"], ctx["logvar"], ctx.get("free_n", 0.0))
        
        total = rec + float(self.beta) * kld
        
        return {"loss": total, "loss/recon": rec, "loss/kld": kld}


    def predict(self, logits: GenLogits, ctx: Optional[Context] = None):
        return [(logits.img + 1.0) * 0.5]
