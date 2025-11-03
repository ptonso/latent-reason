import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from src.vae.config import SemiSupervisedConfig

class SemiSupervisedLoss(nn.Module):
    """Latent-wise binary cross entropy in mu with epsilon margin."""
    def __init__(self, cfg: SemiSupervisedConfig, eps: float = 1e-4):
        super().__init__()
        self.cfg = cfg
        self.weights = float(cfg.weight)
        self.eps = float(eps)

    def active(self, labels) -> bool:
        return self.cfg.cap > 0 and labels is not None

    def forward(self, mu: Tensor, labels: Tensor | None) -> Tensor:
        if not self.active(labels):
            return mu.new_zeros(())
        y = labels.to(mu.dtype)
        D = min(mu.size(1), y.size(1))
        mu, y = mu[:, :D], y[:, :D]

        mask = torch.isfinite(y)
        if mask.sum() == 0:
            return mu.new_zeros(())

        y = y.clamp(0.0, 1.0)
        y = y * (1 - 2 * self.eps) + self.eps

        loss = F.binary_cross_entropy_with_logits(mu[mask], y[mask], reduction="mean")
        return loss * self.weights
