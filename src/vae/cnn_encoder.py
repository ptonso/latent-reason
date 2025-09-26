from dataclasses import dataclass, field
from typing import Dict
import torch
import torch.nn as nn

from src.vae.types import FeatureDict, Context
from src.vae.base import BaseEncoder
from src.vae.utils import Norm, act
from src.vae.config import CNNEncoderConfig



class CNNEncoder(BaseEncoder):
    """Image -> {name: B×C×H×W}."""
    def __init__(self, cfg: CNNEncoderConfig, in_h: int, in_w: int, out_name: str = "s32"):
        super().__init__()
        self.cfg = cfg
        self.out_name = out_name

        self.conv = self._make(cfg, in_h, in_w)
        self._out_ch = cfg.channels[-1]
        stride_total = 1
        for s in cfg.strides:
            stride_total *= int(s)
        self._stride = stride_total

    def _make(self, cfg: CNNEncoderConfig, in_h: int, in_w: int) -> nn.Sequential:
        c_in = cfg.in_channels
        layers = []
        h, w = in_h, in_w
        for c_out, k, s, p in zip(cfg.channels, cfg.kernels, cfg.strides, cfg.paddings):
            layers += [
                nn.Conv2d(c_in, c_out, k, s, p),
                Norm(cfg.norm_type, c_out),
                act(cfg.activation),
            ]
            c_in = c_out
            h = (h + 2 * p - k) // s + 1
            w = (w + 2 * p - k) // s + 1
        self.out_h, self.out_w = h, w
        return nn.Sequential(*layers)

    def out_channels(self) -> Dict[str, int]:
        return {self.out_name: self._out_ch}

    def strides(self) -> Dict[str, int]:
        return {self.out_name: self._stride}

    def forward(self, x: torch.Tensor, ctx: Context | None = None) -> FeatureDict:
        return {self.out_name: self.conv(x)}
