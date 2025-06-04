import torch
import torch.nn as nn
import torch.nn.functional as F
from src.vae.model.base_vae import BaseVAE
from src.vae.model.encoder import Norm, make_activation
from src.vae.model.config import DecoderConfig

class Decoder(BaseVAE):
    def __init__(self, cfg: DecoderConfig, out_height: int, out_width: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        act = make_activation(cfg.activation)

        # FC to “bottleneck”
        layers = []
        in_dim = latent_dim
        for _ in range(cfg.fc_layers):
            layers += [
                nn.Linear(in_dim, cfg.fc_units),
                Norm(cfg.norm_type, cfg.fc_units),
                act
            ]
            in_dim = cfg.fc_units

        self.bottleneck_ch = cfg.channels[0]
        self.bottleneck_h = out_height // (2 ** len(cfg.channels))
        self.bottleneck_w = out_width  // (2 ** len(cfg.channels))
        layers.append(
            nn.Linear(
                in_dim, 
                self.bottleneck_ch * self.bottleneck_h * self.bottleneck_w
                )
                )
        self.fc = nn.Sequential(*layers)

        # Deconv stack
        dlayers = []
        c_in = self.bottleneck_ch
        shape_h, shape_w = self.bottleneck_h, self.bottleneck_w
        for c_out,k,s,p in zip(cfg.channels[1:], cfg.kernels[1:], cfg.strides[1:], cfg.paddings[1:]):
            dlayers += [
                nn.ConvTranspose2d(c_in, c_out, k, s, p),
                Norm(cfg.norm_type, c_out),
                act
            ]
            c_in = c_out
            shape_h *= s
            shape_w *= s

        # final upsample to exact size
        dlayers.append(nn.ConvTranspose2d(c_in, cfg.out_channels,
                                          cfg.kernels[0],
                                          cfg.strides[0],
                                          cfg.paddings[0]))
        dlayers.append(nn.Tanh())
        self.deconv = nn.Sequential(*dlayers)

        self.out_height = out_height
        self.out_width  = out_width
        self.apply(self._init_weights)

    def forward(self, z: torch.Tensor):
        x = self.fc(z)
        B = z.size(0)
        x = x.view(
            B,
            self.bottleneck_ch,
            self.bottleneck_h,
            self.bottleneck_w
        )
        return self.deconv(x)

        # return F.interpolate(
        #     x, 
        #     (self.out_height, self.out_width),
        #     mode='bilinear', 
        #     align_corners=False
        #     )
