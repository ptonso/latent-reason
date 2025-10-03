from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
import torch.nn as nn

from src.vae.types import FeatureDict, Context, GenLogits
from src.vae.utils import Norm, act
from src.vae.base import BaseDecoder
from src.vae.config import CNNDecoderConfig



class CNNDecoder(BaseDecoder):
    """{_in_name} (B×C×H×W) → img."""
    def __init__(
        self,
        cfg: CNNDecoderConfig, *,
        in_name: str,in_ch: int, 
        in_h: int, in_w: int,
        out_h: int, out_w: int
    ):
        super().__init__()
        self.cfg    = cfg
        self.in_name = in_name
        self.in_ch, self.in_h, self.in_w = in_ch, in_h, in_w
        self.deconv = self._make_deconv(in_ch)

        # quick shape check
        with torch.no_grad():
            x = torch.zeros(1, in_ch, in_h, in_w)
            y = self.forward({self.in_name: x}, None).img
            if y.shape[-2:] != (out_h, out_w):
                raise ValueError(f"decoder output {y.shape[-2:]} != target {(out_h, out_w)}")

    def _make_deconv(self, c_in: int) -> nn.Sequential:
        layers = []
        Act = lambda c: [Norm(self.cfg.norm_type, c), act(self.cfg.activation)]

        Ks, Ss, Ps, Cs = self.cfg.kernels, self.cfg.strides, self.cfg.paddings, self.cfg.channels
        n = len(Cs)
        for i, (c_out, k, s, p) in enumerate(zip(Cs, Ks, Ss, Ps)):
            is_last = (i == n - 1)
            if is_last:
                layers += [nn.ConvTranspose2d(c_in, self.cfg.out_channels, k, s, p),
                           nn.Tanh()]
            else:
                layers += [nn.ConvTranspose2d(c_in, c_out, k, s, p), *Act(c_out)]
                c_in = c_out
        return nn.Sequential(*layers)

    def expects(self) -> Optional[Dict[str, int]]:
        return {self.in_name: -1}

    def forward(self, feats: FeatureDict, ctx: Optional[Context] = None) -> GenLogits:
        x   = feats[self.in_name]  # (B, C, H, W) from neck
        img = self.deconv(x)
        return GenLogits(img=img, meta={})


