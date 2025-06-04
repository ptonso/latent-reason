import torch
import torch.nn.functional as F
from torch import nn
from src.vae.model.base_vae import BaseVAE
from src.vae.model.encoder import Encoder
from src.vae.model.decoder import Decoder
from src.vae.model.config import VAEConfig

class BetaVAE(BaseVAE):
    def __init__(self,
                 vae_cfg: VAEConfig,
                 in_height: int,
                 in_width:  int):
        super().__init__()
        self.config  = vae_cfg
        self.device = vae_cfg.device
        self.encoder = Encoder(vae_cfg.encoder, in_height, in_width, vae_cfg.latent_dim)
        self.decoder = Decoder(vae_cfg.decoder, in_height, in_width, vae_cfg.latent_dim)
        self.latent_dim = vae_cfg.latent_dim
        self.beta       = vae_cfg.beta
        self.apply(self._init_weights)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decoder(z)
        return recon, mu, logvar

    def loss_function(self, recon, x, mu, logvar):
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        
        if self.beta == 0:
            return recon_loss, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        kld_element = mu.pow(2) + logvar.exp() - logvar - 1
        kld_per_sample = 0.5 * kld_element.sum(dim=1)
        budget = self.config.free_nats * self.latent_dim
        kld = torch.clamp(kld_per_sample - budget, min=0.0).sum()
        total_loss = recon_loss + self.beta * kld
        return total_loss, recon_loss, kld
