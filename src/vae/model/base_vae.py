import torch
import torch.nn as nn

class BaseVAE(nn.Module):
    """Weight‐init base for all VAE modules."""
    def __init__(self):
        super().__init__()

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            BaseVAE.truncated_normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def truncated_normal_(tensor: torch.Tensor, mean=0.0, std=0.02):
        """Fill tensor with samples from N(mean,std²), truncated to ±2σ."""
        with torch.no_grad():
            tensor.normal_(mean, std)
            while True:
                mask = (tensor < mean - 2*std) | (tensor > mean + 2*std)
                if not mask.any():
                    break
                tensor[mask] = torch.randn_like(tensor[mask]) * std + mean
