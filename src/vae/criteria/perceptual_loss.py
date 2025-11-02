from typing import Callable, Dict, Mapping, Optional
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.vae.config import PerceptualConfig

class PerceptualLoss:
    """
    Reusable perceptual loss wrapper (LPIPS or encoder features).

    Usage:
        perc = PerceptualHelper(cfg.perc)
        perc.init(device=device, encoder=my_encoder)  # optional
        loss = perc(hatx, x)  # returns scalar tensor (or zeros if disabled)
    """
    def __init__(self, cfg: PerceptualConfig):
        self.cfg = cfg
        self.enabled: bool = False
        self._fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None
        self._to3: Optional[Callable[[Tensor], Tensor]] = None

    def init(
        self,
        *,
        device: Optional[str] = None,
        encoder: Optional[nn.Module] = None,
        features_fn: Optional[Callable[[Tensor], Dict[str, Tensor]]] = None,
    ) -> None:
        pcfg = self.cfg
        if pcfg.source == "none" or pcfg.perc_weight == 0.0:
            self.enabled = False
            self._fn = None
            self._to3 = None
            return

        if pcfg.source == "lpips":
            try:
                import lpips
            except Exception as e:
                raise ImportError("perc.source='lpips' requires `pip install lpips`.") from e

            m = lpips.LPIPS(net=pcfg.lpips_net)
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
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
            def fn(xr: Tensor, x: Tensor) -> Tensor:
                xr3 = resize_safe(to3(xr))
                x3  = resize_safe(to3(x))
                return m(xr3, x3).mean()

            self._fn = fn
            self._to3 = to3
            self.enabled = True
            return

        if pcfg.source == "encoder":
            if features_fn is None:
                if encoder is None:
                    raise RuntimeError("encoder-mode perceptual requires `encoder` or `features_fn`.")
                enc = copy.deepcopy(encoder).eval()
                for p in enc.parameters():
                    p.requires_grad_(False)
                if not hasattr(enc, "extract_perc_features"):
                    raise AttributeError("encoder must implement .extract_perc_features(x)->Dict[str,Tensor].")
                features_fn = enc.extract_perc_features

            layers: Mapping[str, float] = pcfg.layers or {}
            if not layers:
                raise ValueError("perc.layers must be provided for encoder-mode perceptual.")

            use_l1 = bool(pcfg.use_l1)

            def to3(t: Tensor) -> Tensor:
                return t if t.size(1) == 3 else t.repeat(1, 3, 1, 1)

            def fn(xr: Tensor, x: Tensor) -> Tensor:
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

            self._fn = fn
            self._to3 = to3
            self.enabled = True
            return

        raise ValueError(f"Unknown perc.source: {pcfg.source}")

    def __call__(self, hatx: Tensor, x: Tensor) -> Tensor:
        if not self.enabled or self._fn is None:
            return x.new_zeros(())
        return self._fn(hatx, x)
