import torch
import torch.nn as nn
from src.vae.model.base_vae import BaseVAE
from src.vae.model.config import EncoderConfig

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

class Encoder(BaseVAE):
    def __init__(self, cfg: EncoderConfig, in_height: int, in_width: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        act = make_activation(cfg.activation)
        layers = []
        c_in = cfg.in_channels
        for c_out,k,s,p in zip(cfg.channels, cfg.kernels, cfg.strides, cfg.paddings):
            layers += [
                nn.Conv2d(c_in, c_out, k, s, p),
                Norm(cfg.norm_type, c_out),
                act
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)

        # compute flattened conv‐output dim
        with torch.no_grad():
            dummy = torch.zeros(1, cfg.in_channels, in_height, in_width)
            self.conv_out_dim = self.conv(dummy).numel()

        # MLP to 2×latent
        mlp = []
        in_dim = self.conv_out_dim
        for _ in range(cfg.fc_layers):
            mlp += [
                nn.Linear(in_dim, cfg.fc_units),
                Norm(cfg.norm_type, cfg.fc_units),
                act
            ]
            in_dim = cfg.fc_units

        mlp.append(nn.Linear(in_dim, 2 * self.latent_dim)) # mu, logvar
        self.mlp = nn.Sequential(*mlp)

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        mu, logvar = self.mlp(x).chunk(2, dim=1)
        return mu, logvar
