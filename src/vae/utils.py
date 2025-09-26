import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """RMSNorm over last dim or channels."""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps    = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            var = x.pow(2).mean(dim=1, keepdim=True)
            return x / torch.sqrt(var + self.eps) * self.weight.view(1, -1, 1, 1)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x / torch.sqrt(var + self.eps) * self.weight

class Norm(nn.Module):
    """Batch/Layer/Group/RMS/Identity."""
    def __init__(self, norm_type: str, num_channels: int):
        super().__init__()
        nt = norm_type.lower()
        self.nt = nt
        if   nt == "batch": self.norm = nn.BatchNorm2d(num_channels)
        elif nt == "layer": self.norm = nn.LayerNorm(num_channels)
        elif nt == "group": self.norm = nn.GroupNorm(32, num_channels)
        elif nt == "rms":   self.norm = RMSNorm(num_channels)
        else:               self.norm = nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.nt == "layer" and x.dim() == 4:
            b,c,h,w = x.shape
            x = x.permute(0,2,3,1)
            x = self.norm(x)
            return x.permute(0,3,1,2)
        return self.norm(x)

def act(name: str) -> nn.Module:
    name = name.lower()
    if name == "silu": return nn.SiLU()
    if name == "relu": return nn.ReLU()
    return nn.Identity()
