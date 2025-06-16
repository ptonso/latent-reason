import torch
import torch.nn as nn
from src.vae.model.base_vae import BaseVAE
from src.vae.model.config import EncoderConfig, DecoderConfig

def conv_out(h: int, k: int, s: int, p: int) -> int:
    return (h + 2 * p - k) // s + 1

def deconv_out(h: int, k: int, s: int, p: int, op: int) -> int:
    return (h - 1) * s - 2 * p + k + op

class Encoder(BaseVAE):
    def __init__(
        self,
        cfg: EncoderConfig,
        in_height: int,
        in_width: int,
        latent_dim: int
    ):
        super().__init__()
        act = make_activation(cfg.activation)
        layers = []
        c_in = cfg.in_channels
        h, w = in_height, in_width

        # build conv stack and track output size
        for c_out, k, s, p in zip(cfg.channels, cfg.kernels, cfg.strides, cfg.paddings):
            layers += [nn.Conv2d(c_in, c_out, k, s, p), Norm(cfg.norm_type, c_out), act]
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
            mlp_layers += [nn.Linear(in_dim, cfg.fc_units), Norm(cfg.norm_type, cfg.fc_units), act]
            in_dim = cfg.fc_units
        mlp_layers.append(nn.Linear(in_dim, 2 * latent_dim))
        self.mlp = nn.Sequential(*mlp_layers)

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor):
        x = self.conv(x).flatten(1)
        mu, logvar = self.mlp(x).chunk(2, dim=1)
        return mu, logvar

class Decoder(BaseVAE):
    def __init__(
        self,
        cfg: DecoderConfig,
        bottleneck_h: int,
        bottleneck_w: int,
        target_h: int,
        target_w: int,
        latent_dim: int
    ):
        super().__init__()
        act = make_activation(cfg.activation)

        # store bottleneck shape
        self.bot_c = cfg.channels[0]
        self.bot_h = bottleneck_h
        self.bot_w = bottleneck_w

        # build MLP from latent to feature map
        fc_layers = []
        in_dim = latent_dim
        for _ in range(cfg.fc_layers):
            fc_layers += [nn.Linear(in_dim, cfg.fc_units), Norm(cfg.norm_type, cfg.fc_units), act]
            in_dim = cfg.fc_units
        fc_layers.append(nn.Linear(in_dim, self.bot_c * bottleneck_h * bottleneck_w))
        self.fc = nn.Sequential(*fc_layers)

        # build deconv stack
        layers = []
        c_in = self.bot_c
        for c_out, k, s, p in zip(cfg.channels[1:], cfg.kernels[1:], cfg.strides[1:], cfg.paddings[1:]):
            layers += [nn.ConvTranspose2d(c_in, c_out, k, s, p), Norm(cfg.norm_type, c_out), act]
            c_in = c_out
        layers.append(nn.ConvTranspose2d(c_in, cfg.out_channels, cfg.kernels[0], cfg.strides[0], cfg.paddings[0]))
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






def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    return nn.Identity()

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            # (B,C,H,W)
            var = x.pow(2).mean(dim=1, keepdim=True)
            return x / torch.sqrt(var + self.eps) * self.weight.view(1, -1, 1, 1)
        else:
            # (B,Features)
            var = x.pow(2).mean(dim=-1, keepdim=True)
            return x / torch.sqrt(var + self.eps) * self.weight

class Norm(nn.Module):
    def __init__(self, norm_type: str, num_channels: int):
        super().__init__()
        nt = norm_type.lower()
        if nt == "batch":
            self.norm = nn.BatchNorm2d(num_channels)
        elif nt == "layer":
            self.norm = nn.LayerNorm(num_channels)
        elif nt == "group":
            self.norm = nn.GroupNorm(32, num_channels)
        elif nt == "rms":
            self.norm = RMSNorm(num_channels)
        else:
            self.norm = nn.Identity()
        self.norm_type = nt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # layer‐norm over channels in conv
        if self.norm_type == "layer" and x.dim() == 4:
            b,c,h,w = x.shape
            x = x.permute(0,2,3,1)
            x = self.norm(x)
            return x.permute(0,3,1,2)
        return self.norm(x)