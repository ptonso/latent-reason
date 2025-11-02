from dataclasses import dataclass
from typing import Mapping, Literal, Optional, Any, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.vae.types import GenLogits, GenTargets, Context
from src.vae.base import BaseCriterion, LossDict

from src.vae.criteria.gaussian_recon import GaussianRecon
from src.vae.criteria.mdl_recon import MDLRecon
from src.vae.criteria.perceptual_loss import PerceptualLoss
from src.vae.criteria.ssemi_loss import SemiSupervisedLoss
from src.vae.config import BetaVAECriterionConfig, GaussianReconConfig, MDLReconConfig


class BetaVAECriterion(BaseCriterion):
    """
    Beta VAE loss:
    - ReconstructionLoss (optional + perceptual)
    - Kullback-Leibler Loss (gaussian neck)
    """
    def __init__(self, cfg: BetaVAECriterionConfig):
        super().__init__()
        self.cfg = cfg
        self.beta = float(cfg.beta)

        if isinstance(cfg.recon, GaussianReconConfig):
            self.inner = GaussianRecon(cfg.recon)
        elif isinstance(cfg.recon, MDLReconConfig):
            self.inner = MDLRecon(cfg.recon)

        self.perc   = PerceptualLoss(cfg.perc)
        self.ssuper = SemiSupervisedLoss(cfg.ssuper)

    def init_perc(
        self,
        *,
        device: Optional[str] = None,
        encoder: Optional[nn.Module] = None,
        features_fn: Optional[Callable[[Tensor], Dict[str, Tensor]]] = None,
    ) -> None:
        """External api"""
        self.perc.init(device=device, encoder=encoder, features_fn=features_fn)

    @staticmethod
    def _kld(mu: Tensor, logvar: Tensor, free_nats: float) -> Tensor:
        kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
        kl = torch.clamp(kl - free_nats, min=0.0)
        return kl.sum(1).mean()

    def forward(self, logits: GenLogits, target: GenTargets, ctx: Optional[Context] = None) -> LossDict:
        x = target.img if hasattr(target, "img") else target

        # recon term
        recon = self.inner.recon_loss(logits.params, x)
        rec = float(self.cfg.perc.pix_weight) * recon

        # perceptual term
        if self.perc.enabled:
            hatx = self.inner.mean_image(logits)
            perc_term = self.perc(hatx, x)
            rec = rec + float(self.cfg.perc.perc_weight) * perc_term
        
        # Semisupervised term
        ssuper = logits.params.new_zeros(())
        labels = None
        if hasattr(target, "meta") and isinstance(target.meta, dict):
            labels = target.meta.get("labels", None)
        ssuper = self.ssuper(ctx["mu"], labels)

        # KL term
        if ctx is None or ("mu" not in ctx or "logvar" not in ctx):
            z = torch.zeros((), device=logits.params.device)
            return {"loss": rec, "loss/recon": rec, "loss/kld": z}
        kld = self._kld(ctx["mu"], ctx["logvar"], ctx.get("free_n", 0.0))

        total = rec + self.beta * kld + ssuper
        return {"loss": total, "loss/recon": rec, "loss/kld": kld, "loss/ssuper": ssuper}


    @torch.no_grad()
    def predict(self, logits: GenLogits, ctx: Optional[Context] = None):
        """Map to [0,1] for visualization."""
        hatx = self.inner.mean_image(logits)
        return [hatx.mul(0.5).add(0.5).clamp(0.0, 1.0)]
