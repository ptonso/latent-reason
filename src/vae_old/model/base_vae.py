import torch
import torch.nn as nn
from typing import Optional, Union
from src.vae.trainer import Trainer
from src.vae.model.config import TrainConfig


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
    



class BaseVAE(nn.Module):
    """
    Base class for all VAE modules.
    """

    def __init__(self):
        super().__init__()


    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            BaseVAE.truncated_normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    @staticmethod
    def truncated_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 0.02):
        """Fill tensor with samples from N(mean,std²), truncated to ±2σ."""
        with torch.no_grad():
            tensor.normal_(mean, std)
            while True:
                mask = (tensor < mean - 2*std) | (tensor > mean + 2*std)
                if not mask.any():
                    break
                tensor[mask] = torch.randn_like(tensor[mask]) * std + mean


    @torch.no_grad()
    def decode_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent z → float in [0,1] using sigmoid.
        Assumes decoder(z) ends in tanh [-1,1].
        """
        return (self.decoder(z) + 1.0) * 0.5

    @torch.no_grad()
    def decode_uint8(self, z: torch.Tensor) -> torch.ByteTensor:
        """
        Decode latent z → uint8 {0,…,255}.  Convenience wrapper for saving.
        """
        return (self.decode_prob(z) * 255.0).clamp(0, 255).to(torch.uint8)
    

    def make_activation(self, name: str) -> nn.Module:
        name = name.lower()
        if name == "silu":
            return nn.SiLU()
        if name == "relu":
            return nn.ReLU()
        return nn.Identity()


    def run(self, train_cfg: TrainConfig, resume:bool) -> Union[nn.Module, Trainer]:

        if train_cfg is None:
            train_cfg = TrainConfig(data_yaml=train_cfg.data_yaml)

        trainer = Trainer(self, self.config, train_cfg)
        trainer.train(resume=resume)
        return trainer

    def build_norm(self, norm_type: str, num_channels: int):
        return Norm(norm_type, num_channels)


