from dataclasses import dataclass
from typing import Mapping, Literal, Optional, Any, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.vae.types import GenLogits, GenTargets, Context
from src.vae.base import BaseCriterion, LossDict
from src.vae.config import BetaVAECriterionConfig


class BetaVAECriterion(BaseCriterion):
    """
    Pixel recon + KL (+ optional perceptual)
    L_pix(recon,x)  : {L2, L1, Huber} on [-1,1] images; reduced to scalar
    KL(mu,logvar)   : diagonal gaussian KL with free-nats; per-image sum then mean.
    L_perc(recon,x) : {L1, L2} on φ(^x)-φ(x), compares features; reduced to scalar.
    
    total = pix_weight*L_pix + perc_weight*L_perc + beta*KL
    """
    def __init__(self, cfg: BetaVAECriterionConfig):
        super().__init__()
        self.cfg = cfg
        self.beta = float(cfg.beta)

        self._perc_enabled: bool = False
        self._perc_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None
        self._to_3ch: Optional[Callable[[Tensor], Tensor]] = None



    def _recon(self, recon: Tensor, x: Tensor) -> Tensor:
        rt = self.cfg.recon_type
        if   rt in ("l2","mse"):  per = F.mse_loss(recon, x, reduction="none")
        elif rt in ("l1","mae"):  per = F.l1_loss(recon, x, reduction="none")
        else:                     per = F.smooth_l1_loss(recon, x, reduction="none", beta=self.cfg.huber_delta)
        pix = per.flatten(1).sum(1).mean()

        if self._perc_enabled and self._perc_fn is not None:
            perc = self._perc_fn(recon, x)
            return float(self.cfg.pix_weight) * pix + float(self.cfg.perc_weight) * perc
        else:
            return float(self.cfg.pix_weight) * pix

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
        if self.cfg.perc_source == "none" or self.cfg.perc_weight == 0.0:
            self._perc_enabled = False
            self._perc_fn = None
            self._to_3ch = None
            return
        
        if self.cfg.perc_source == "lpips":
            try:
                import lpips
            except Exception as e:
                raise ImportError("perc_source='lpips' requires `pip install lpips`.") from e

            m = lpips.LPIPS(net=self.cfg.lpips_net)
            m.eval()
            for p in m.parameters(): p.requires_grad_(False)
            if device is not None:
                m = m.to(device)

            MIN_SIDE = 64

            def to3(t: Tensor) -> Tensor:
                return t if t.size(1) == 3 else t.repeat(1, 3, 1, 1)

            def resize_safe(t: Tensor) -> Tensor:
                _, _, h, w = t.shape
                if min(h, w) >= MIN_SIDE:
                    return t
                return F.interpolate(t, size=(MIN_SIDE, MIN_SIDE), mode="bilinear", align_corners=False)
                
            def perc_fn(xr: Tensor, x: Tensor) -> Tensor:
                nonlocal m
                xr3 = resize_safe(to3(xr))
                x3  = resize_safe(to3(x))
                return m(xr3, x3).mean()

            self._perc_fn = perc_fn
            self._perc_enabled = True
            self._to_3ch = to3
            return

        if self.cfg.perc_source == "encoder":
            if features_fn is None:
                if encoder is None:
                    raise RuntimeError("encoder-mode perceptual requires `encoder` or `features_fn`.")
                enc = copy.deepcopy(encoder).eval()
                for p in enc.parameters():
                    p.requires_grad_(False)
                if not hasattr(enc, "extract_perc_features"):
                    raise AttributeError("encoder must implement .extract_perc_features(x)->Dict[str,Tensor].")
                features_fn = enc.extract_perc_features

            layers: Mapping[str, float] = self.cfg.perc_layers or {}
            if not layers:
                raise ValueError("perc_layers must be provided for encoder-mode perceptual.")

            use_l1 = bool(self.cfg.perc_use_l1)

            def to3(t: Tensor) -> Tensor:
                return t if t.size(1) == 3 else t.repeat(1, 3, 1, 1)

            def perc_fn(xr: Tensor, x: Tensor) -> Tensor:
                fr = features_fn(to3(xr))
                fx = features_fn(to3(x))
                loss = xr.new_zeros(())
                for name, w in layers.items():
                    if name not in fr or name not in fx:
                        raise KeyError(f"Missing layer '{name}' in feature dict.")
                    diff = fr[name] - fx[name]
                    term = diff.abs().mean() if use_l1 else diff.pow(2).mean()
                    loss = loss + float(w) * term
                return loss

            self._perc_fn = perc_fn
            self._to_3ch = to3
            self._perc_enabled = True
            return

        raise ValueError(f"Unknown perc_source: {self.cfg.perc_source}")


    def forward(
        self, 
        logits: GenLogits, 
        target: Any, 
        ctx: Optional[Context] = None
    ) -> LossDict:
        
        x   = target.img if hasattr(target, "img") else target
        rec = self._recon(logits.img, x)
        
        if ctx is None or "mu" not in ctx:
            return {"loss": rec, "loss/recon": rec, "loss/kld": torch.zeros((), device=logits.img.device)}
        
        kld   = self._kld(ctx["mu"], ctx["logvar"], ctx.get("free_n", 0.0))
        total = rec + self.beta * kld
        return {"loss": total, "loss/recon": rec, "loss/kld": kld}


    @torch.no_grad
    def predict(self, logits: GenLogits, ctx: Optional[Context] = None):
        return [(logits.img + 1.0) * 0.5]