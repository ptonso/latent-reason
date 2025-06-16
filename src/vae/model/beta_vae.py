# src/vae/model/beta_vae.py

import torch
import torch.nn.functional as F
from torch import nn
from src.vae.model.base_vae import BaseVAE
from src.vae.model.encoder import Encoder
from src.vae.model.decoder import Decoder
from src.vae.model.config import VAEConfig

class BetaVAE(BaseVAE):
    """
    A β-VAE: single interface.  Instantiate and call .train(...)
    """

    def __init__(
        self,
        vae_cfg: VAEConfig,
        in_height: int,
        in_width:  int
    ):
        super().__init__()
        self.config     = vae_cfg
        self.device     = vae_cfg.device
        self.encoder    = Encoder(vae_cfg.encoder, in_height, in_width, vae_cfg.latent_dim)
        self.decoder    = Decoder(vae_cfg.decoder, in_height, in_width, vae_cfg.latent_dim)
        self.latent_dim = vae_cfg.latent_dim
        self.beta       = vae_cfg.beta
        self.C_max      = vae_cfg.C_max
        self.gamma      = vae_cfg.gamma
        self.C_start    = vae_cfg.C_start
        self.C_stop     = vae_cfg.C_stop
        self.apply(self._init_weights)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decoder(z)
        return recon, mu, logvar

    def loss_function(
        self,
        recon: torch.Tensor,
        x:     torch.Tensor,
        mu:    torch.Tensor,
        logvar:torch.Tensor,
        global_iter: int = 0
    ):
        """
        β-VAE loss with capacity-annealing (Burgess et al. 2018).

        Args
        ----
        recon : raw logits from decoder  (no tanh/sigmoid!)
        x     : targets in [0,1]
        mu, logvar : Gaussian parameters
        global_iter : training step counter (start at 0)
        """

        # --- reconstruction ---
        recon_loss = F.binary_cross_entropy_with_logits(
            recon, x, reduction='sum'
        )

        # --- KL divergence (mean over batch) ---
        kld = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(1).mean()

        # --- capacity schedule C(t) ---
        if global_iter < self.C_start:
            C = 0.0
        elif global_iter > self.C_stop:
            C = self.C_max
        else:
            C = self.C_max * (global_iter - self.C_start) / (self.C_stop - self.C_start)

        kld_loss = self.gamma * torch.abs(kld - C)

        total = recon_loss + kld_loss
        return total, recon_loss.detach(), kld.detach()
