# src/vae/model/mlp_encoder_decoder.py

import torch
import torch.nn as nn
from typing import Tuple
from src.vae.model.base_vae import BaseVAE
from src.vae.model.config import MLPEncoderConfig, MLPDecoderConfig



class MLPEncoder(BaseVAE):
    """
    Flatten (B,C,H,W) -> MLP -> 2*latent (mu, logvar).
    """
    def __init__(self, cfg: MLPEncoderConfig, in_height: int, in_width: int, latent_dim: int):
        super().__init__()
        self.cfg = cfg
        self.in_c = cfg.in_channels
        self.in_h = in_height
        self.in_w = in_width

        act = self.make_activation(cfg.activation)
        layers = []

        in_dim = self.in_c * self.in_h * self.in_w
        for h in cfg.hidden:
            layers += [nn.Linear(in_dim, h), self.build_norm(cfg.norm_type, h), act]
            in_dim = h

        layers += [nn.Linear(in_dim, 2 * latent_dim)]
        self.mlp = nn.Sequential(*layers)

        # for parity with CNN path (used by BetaVAE to grab shapes)
        self.conv_out_h = 1
        self.conv_out_w = 1
        self.conv_out_dim = self.in_c * self.in_h * self.in_w

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        mu, logvar = self.mlp(x).chunk(2, dim=1)
        return mu, logvar


class MLPDecoder(BaseVAE):
    """
    z -> MLP -> logits reshaped to (B, C, H, W).
    No final activation (BCEWithLogits in loss / sigmoid in BaseVAE.decode_prob).
    """
    def __init__(
        self,
        cfg: MLPDecoderConfig,
        target_h: int,
        target_w: int,
        latent_dim: int
    ):
        super().__init__()
        self.cfg = cfg
        self.out_c = cfg.out_channels
        self.out_h = target_h
        self.out_w = target_w

        act = self.make_activation(cfg.activation)
        layers = []
        in_dim = latent_dim

        for h in cfg.hidden:
            layers += [nn.Linear(in_dim, h), self.build_norm(cfg.norm_type, h), act]
            in_dim = h

        layers += [nn.Linear(in_dim, self.out_c * self.out_h * self.out_w)]
        self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        # sanity check: produce one dummy forward
        with torch.no_grad():
            z = torch.zeros(1, latent_dim)
            out = self.forward(z)
            if out.shape[-2:] != (self.out_h, self.out_w):
                raise ValueError(
                    f"Decoder produces {tuple(out.shape[-2:])}, expected {(self.out_h, self.out_w)}."
                )

    def forward(self, z: torch.Tensor):
        x = self.mlp(z)
        return x.view(z.size(0), self.out_c, self.out_h, self.out_w)




def build_mlp_enc_dec(
    enc_cfg: MLPEncoderConfig,
    dec_cfg: MLPDecoderConfig,
    in_height:  int,
    in_width:   int,
    target_h:   int,
    target_w:   int,
    latent_dim: int,
    ) -> Tuple[MLPEncoder, MLPDecoder]:

    return (
        MLPEncoder(enc_cfg, in_height, in_width, latent_dim),
        MLPDecoder(dec_cfg, target_h, target_w, latent_dim)
        )

