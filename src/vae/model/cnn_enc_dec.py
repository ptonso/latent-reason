import torch
import torch.nn as nn
from typing import Tuple
from src.vae.model.base_vae import BaseVAE
from src.vae.model.config import CNNEncoderConfig, CNNDecoderConfig

def conv_out(h: int, k: int, s: int, p: int) -> int:
    return (h + 2 * p - k) // s + 1

def deconv_out(h: int, k: int, s: int, p: int, op: int) -> int:
    return (h - 1) * s - 2 * p + k + op

class CNNEncoder(BaseVAE):
    def __init__(
        self,
        cfg: CNNEncoderConfig,
        in_height: int,
        in_width: int,
        latent_dim: int
    ):
        super().__init__()
        act = self.make_activation(cfg.activation)
        layers = []
        c_in = cfg.in_channels
        h, w = in_height, in_width

        # build conv stack and track output size
        for c_out, k, s, p in zip(cfg.channels, cfg.kernels, cfg.strides, cfg.paddings):
            layers += [nn.Conv2d(c_in, c_out, k, s, p), self.build_norm(cfg.norm_type, c_out), act]
            c_in = c_out
            h = conv_out(h, k, s, p)
            w = conv_out(w, k, s, p)
        self.conv = nn.Sequential(*layers)

        # record bottleneck spatial dims
        self.conv_out_h = h
        self.conv_out_w = w
        self.conv_out_dim = h * w * cfg.channels[-1]

        # build MLP to 2×latent
        mlp_layers = []
        in_dim = self.conv_out_dim
        for _ in range(cfg.fc_layers):
            mlp_layers += [nn.Linear(in_dim, cfg.fc_units), self.build_norm(cfg.norm_type, cfg.fc_units), act]
            in_dim = cfg.fc_units
        mlp_layers.append(nn.Linear(in_dim, 2 * latent_dim))
        self.mlp = nn.Sequential(*mlp_layers)

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor):
        x = self.conv(x).flatten(1)
        mu, logvar = self.mlp(x).chunk(2, dim=1)
        return mu, logvar

class CNNDecoder(BaseVAE):
    def __init__(
        self,
        cfg: CNNDecoderConfig,
        bottleneck_h: int,
        bottleneck_w: int,
        target_h: int,
        target_w: int,
        latent_dim: int
    ):
        super().__init__()

        act = self.make_activation(cfg.activation)

        # store bottleneck shape
        self.bot_c = cfg.channels[0]
        self.bot_h = bottleneck_h
        self.bot_w = bottleneck_w

        # build MLP from latent to feature map
        fc_layers = []
        in_dim = latent_dim
        for _ in range(cfg.fc_layers):
            fc_layers += [nn.Linear(in_dim, cfg.fc_units), self.build_norm(cfg.norm_type, cfg.fc_units), act]
            in_dim = cfg.fc_units
        fc_layers.append(nn.Linear(in_dim, self.bot_c * bottleneck_h * bottleneck_w))
        self.fc = nn.Sequential(*fc_layers)

        # build deconv stack
        layers = []
        c_in = self.bot_c
        for c_out, k, s, p in zip(cfg.channels[1:], cfg.kernels[1:], cfg.strides[1:], cfg.paddings[1:]):
            layers += [nn.ConvTranspose2d(c_in, c_out, k, s, p), self.build_norm(cfg.norm_type, c_out), act]
            c_in = c_out
        layers.append(nn.ConvTranspose2d(c_in, cfg.out_channels, cfg.kernels[0], cfg.strides[0], cfg.paddings[0]))
        layers.append(nn.Tanh())
        self.deconv = nn.Sequential(*layers)

        self.apply(self._init_weights)

        with torch.no_grad():
            z = torch.zeros(1, latent_dim)
            out = self.forward(z)
            h_out, w_out = out.shape[-2], out.shape[-1]
            if (h_out, w_out) != (target_h, target_w):
                raise ValueError(
                    f"Decoder produces {h_out}×{w_out}, expected {target_h}×{target_w}. "
                    "Check your kernels/strides/paddings configuration."
                )

    def forward(self, z: torch.Tensor):
        x = self.fc(z).view(z.size(0), self.bot_c, self.bot_h, self.bot_w)
        return self.deconv(x)



def build_cnn_enc_dec(
    enc_cfg: CNNEncoderConfig,
    dec_cfg: CNNDecoderConfig,
    in_height:    int,
    in_width:     int,
    target_h:     int,
    target_w:     int,
    latent_dim:   int,
    ) -> Tuple[CNNEncoder, CNNDecoder]:

    enc = CNNEncoder(enc_cfg, in_height, in_width, latent_dim)
    bot_h, bot_w = enc.conv_out_h, enc.conv_out_w
    dec = CNNDecoder(dec_cfg, bot_h, bot_w, target_h, target_w, latent_dim)

    return enc, dec
