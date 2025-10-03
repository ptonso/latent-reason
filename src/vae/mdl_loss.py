from dataclass import dataclass
from typing import Mapping, Optional, Any, Callable, Dict, Tuple
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.vae.types import GenLogits, Context
from src.vae.base  import BaseCriterion, LossDict
from src.vae.config import MDLCriterionConfig
from src.vae.perceptual import PerceptualLoss



class MDLCriterion(BaseCriterion):
    """
    Mixture of Discretized Logitstics (MDL/DMOL) recon + KL (+ optional perceptual)

    params = concat[ π_logits[K], μ[3K], log_s[3K] ]  → total channels = 7K
    targets: x ∈ [-1,1], 8-bit bins with Δ = 2/255.

    L_MDL(x | params) : mixture of discretized logistics. per-image sum mean over batch    .
    KL(mu,logvar)     : diagonal gaussian KL with free-nats; per-image sum, mean over batch.
    L_perc(xhat,x)    : {L1, L2} on φ(hatx)-φ(x), compares features; reduced to scalar.
    
    total = pix_weight*L_MDL + perc_weight*L_perc + beta*KL


   L_MDL = - Σ_pixels log Σ_{k=1..K} softmax(π)_k * Π_{c∈{R,G,B}}
                                 [σ((x_c + Δ/2 - μ_ck)/s_ck) - σ((x_c - Δ/2 - μ_ck)/s_ck)]
                            where σ is the logistic CDF. We compute in log-space with clamping for stability
    """

    def __init__(self, cfg: MDLCriterionConfig):
        super().__init__()
        self.cfg = cfg
        self.beta = float(cfg.beta)
        self.K: int = int(cfg.K)
        # 8-bit discretization over [-1,1]
        self.delta: float = 2.0 / 255.0
        self.perc = PerceptualLoss(cfg.perc)


    def _split_params(self, params: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        params: (B, 7K, H, W)  → pi_logits: (B,K,H,W), means: (B,3,K,H,W), log_s: (B,3,K,H,W)
        Shared mixture weights across channels (RGB).
        """
        B, C, H, W = params.shape
        K = self.K
        assert C == 7 * K, f"Expected {7*K} channels, got {C}."
        pi_logits = params[:, 0*K:1*K, :, :]
        means     = params[:, 1*K:4*K, :, :].reshape(B, 3, K, H, W)
        log_s     = params[:, 4*K:7*K, :, :].reshape(B, 3, K, H, W)
        return pi_logits, means, log_s

    def _logistic_cdf(self, t: Tensor) -> Tensor:
        return torch.sigmoid(t)
    
    def _log_prob_bin(self, x: Tensor, means: Tensor, log_s: Tensor) -> Tensor:
        """
        x:     (B,3,H,W) in [-1,1]
        means: (B,3,K,H,W)
        log_s: (B,3,K,H,W)
        returns: log p(x | component k), aggregated over channels -> (B,K,H,W)
        """
        log_s = log_s.clamp(min=-7., max=7.)
        inv_s = torch.exp(-log_s) # 1/s

        # broadcast x to (B,3,K,H,W)
        x = x.unsqueeze(2)

        # bin edges
        delta = self.delta
        plus  = (x + 0.5 * delta - means) * inv_s
        minus = (x - 0.5 * delta - means) * inv_s

        # CDF difference per channel
        cdf_plus  = self._logistic_cdf(plus)
        cdf_minus = self._logistic_cdf(minus)
        p = (cdf_plus - cdf_minus).clamp(min=1e-12)

        log_p_ch = torch.log(p)
        log_p    = log_p_ch.sum(dim=1)
        return log_p

    def _nll(self, params: Tensor, x: Tensor) -> Tensor:
        """
        Negative log-likelihood with mixture over components.
        """
        pi_logits, means, log_s = self._split_params(params)         # (B,K,H,W)
        log_pi = torch.log_softmax(pi_logits, dim=1)
        log_p_x_given_k = self._log_prob_bin(x, means, log_s)
        log_prob = torch.logsumexp(log_pi + log_p_x_given_k, dim=1)  # (B,H,W)
        nll = -log_prob.flatten(1).sum(1).mean()                     # sum over pixels, mean over batch
        return nll


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
        
        x = target.img if hasattr(target, "img") else target
        mdl = self._nll(logits.img, x)

        perc = x.new_zeros(())
        if self.perc.enabled:
            with torch.no_grad():
                hatx = self._mixture_mean(logits.img)
            perc = self.perc(hatx, x)
        
        rec = float(self.cfg.pix_weight) * mdl + float(self.cfg.perc_weight) * perc

        if ctx is None or "mu" not in ctx:
            z = torch.zeros((), device=logits.img.device)
            return {"loss": rec, "loss/recon": rec, "loss/kld": z}
        
        kld = self._kld(ctx["mu"], ctx["logvar"], ctx.get("free_n", 0.0))
        total = rec + self.beta * kld
        return {"loss": total, "loss/recon": rec, "loss/kld": kld}


    @torch.no_grad()
    def predict(self, logits: GenLogits, ctx: Optional[Context] = None):
        """
        Deterministic reconstruction for visualization: mixture MEAN per channel,
        then map [-1,1] -> [0,1].
        """
        hatx = self._mixture_mean(logits.img) # (B,3,H,W) in [-1,1]

    
    def _mixture_mean(self, params: Tensor) -> Tensor:
        """
        mixture mean per channel: E[x] = Σ_k π_k μ_k (independent channels)
        returns tensor in [-1,1]
        """
        B, C, H, W = params.shape
        K = self.K
        pi_logits, means, _ = self._split_params(params)
        pi = torch.softmax(pi_logits, dim=1)
        hatx = (pi.unsqueeze(1) * means).sum(dim=2)
        return hatx