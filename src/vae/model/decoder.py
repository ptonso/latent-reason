import torch
import torch.nn as nn
from src.vae.model.base_vae import BaseVAE
from src.vae.model.encoder import Norm, make_activation
from src.vae.model.config import DecoderConfig

class Decoder(BaseVAE):
    def __init__(self, cfg: DecoderConfig, out_height: int, out_width: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        act = make_activation(cfg.activation)

        forward_params = list(zip(cfg.kernels[1:], cfg.strides[1:], cfg.paddings[1:])) + [
            (cfg.kernels[0], cfg.strides[0], cfg.paddings[0])
        ]
        inverse_params = list(reversed(forward_params))

        h = out_height
        w = out_width
        for k, s, p in inverse_params:
            h = (h - k + 2 * p) // s + 1
            w = (w - k + 2 * p) // s + 1

        self.bottleneck_ch = cfg.channels[0]
        self.bottleneck_h = h
        self.bottleneck_w = w

        layers = []
        in_dim = latent_dim
        for _ in range(cfg.fc_layers):
            layers += [nn.Linear(in_dim, cfg.fc_units), Norm(cfg.norm_type, cfg.fc_units), act]
            in_dim = cfg.fc_units
        layers.append(
            nn.Linear(in_dim, self.bottleneck_ch * self.bottleneck_h * self.bottleneck_w)
        )
        self.fc = nn.Sequential(*layers)

        dlayers = []
        c_in = self.bottleneck_ch
        for c_out, k, s, p in zip(
            cfg.channels[1:], cfg.kernels[1:], cfg.strides[1:], cfg.paddings[1:]
        ):
            dlayers += [nn.ConvTranspose2d(c_in, c_out, k, s, p), Norm(cfg.norm_type, c_out), act]
            c_in = c_out
        dlayers.append(
            nn.ConvTranspose2d(c_in, cfg.out_channels, cfg.kernels[0], cfg.strides[0], cfg.paddings[0])
        )
        self.deconv = nn.Sequential(*dlayers)
        self.apply(self._init_weights)

    def forward(self, z: torch.Tensor):
        x = self.fc(z)
        x = x.view(z.size(0), self.bottleneck_ch, self.bottleneck_h, self.bottleneck_w)
        return self.deconv(x)
