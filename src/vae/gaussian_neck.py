from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn

from src.vae.types import FeatureDict, Context
from src.vae.base import BaseNeck
from src.vae.utils import Norm, act
from src.vae.config import GaussianNeckConfig


class GaussianNeck(BaseNeck):
    """{in_name} (B×C×H×W) ↔ {out_name} (B×C×H×W) via MLP: feat → [mu,logvar] → feat_hat."""
    def __init__(
        self,
        cfg: GaussianNeckConfig, *,
        in_name: str,
        in_ch: int, in_h: int, in_w: int,
        out_name: str
    ):
        super().__init__()
        self.cfg      = cfg
        self.in_name  = in_name
        self.out_name = out_name
        self.in_ch, self.in_h, self.in_w = in_ch, in_h, in_w
        self._oc = {out_name: in_ch}
        self._st = {out_name: 1}

        flat = in_ch * in_h * in_w
        d2   = 2 * cfg.latent_dim
        self.mlp_enc = self._make_mlp(in_dim=flat, out_dim=d2)
        self.mlp_dec = self._make_mlp(in_dim=d2,   out_dim=flat)
        self.apply(self._init_weights)

    def _make_mlp(self, in_dim: int, out_dim: int) -> nn.Sequential:
        layers, c = [], in_dim
        for _ in range(self.cfg.fc_layers):
            layers += [nn.Linear(c, self.cfg.fc_units),
                       Norm(self.cfg.norm_type, self.cfg.fc_units),
                       act(self.cfg.activation)]
            c = self.cfg.fc_units
        layers += [nn.Linear(c, out_dim)]
        return nn.Sequential(*layers)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            with torch.no_grad():
                GaussianNeck.truncated_normal_(m.weight, 0.0, 0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = logvar.clamp(min=-30, max=20)
        std    = torch.exp(0.5 * logvar)
        eps    = torch.randn_like(std)
        return mu + eps * std


    def out_channels(self) -> Dict[str, int]: return dict(self._oc)
    def strides(self)      -> Dict[str, int]: return dict(self._st)

    def forward(self, feats: FeatureDict, ctx: Context | None = None) -> FeatureDict:
        x  = feats[self.in_name].flatten(1)           # (B, C*H*W)
        s  = self.mlp_enc(x)                          # (B, 2D)
        mu, lv = s.chunk(2, dim=1)                    # (B,D), (B,D)
        if ctx is not None:
            ctx["mu"], ctx["logvar"], ctx["free_n"] = mu, lv, self.cfg.free_nats
        s_dec   = torch.cat([mu, lv], dim=1)          # (B, 2D)
        feat    = self.mlp_dec(s_dec).view(x.size(0), self.in_ch, self.in_h, self.in_w)
        return {self.out_name: feat}

    @staticmethod
    def truncated_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 0.02):
        with torch.no_grad():
            tensor.normal_(mean, std)
            while True:
                mask = (tensor < mean - 2*std) | (tensor > mean + 2*std)
                if not mask.any(): break
                tensor[mask] = torch.randn_like(tensor[mask]) * std + mean


