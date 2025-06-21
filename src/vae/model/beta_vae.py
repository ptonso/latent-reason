import torch
import torch.nn.functional as F
from typing import Tuple
from src.vae.model.base_vae     import BaseVAE
from src.vae.model.encoder_decoder import Encoder, Decoder
from src.vae.model.config       import VAEConfig

class BetaVAE(BaseVAE):
    def __init__(self, vae_cfg: VAEConfig):
        super().__init__()
        self.config    = vae_cfg
        img_size       = vae_cfg.img_size
        self.beta      = vae_cfg.beta
        self.free_nats = vae_cfg.free_nats
        self.device    = vae_cfg.device

        self.encoder = Encoder(vae_cfg.encoder, img_size, img_size, vae_cfg.latent_dim)
        bot_h, bot_w = self.encoder.conv_out_h, self.encoder.conv_out_w
        self.decoder = Decoder(
            vae_cfg.decoder, bot_h, bot_w, img_size, img_size, vae_cfg.latent_dim
        )

        self.latent_dim = vae_cfg.latent_dim
        self.apply(self._init_weights)


    @staticmethod
    def reparameterize(mu, logvar):
        logvar = logvar.clamp(min=-30, max=20)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, x):
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decoder(z)
        return recon, mu, logvar


    def loss_function(
        self,
        recon:   torch.Tensor,   # tanh B×C×H×W
        x:       torch.Tensor,   # targets in [-1,1]
        mu:      torch.Tensor,   # B×d
        logvar:  torch.Tensor,   # B×d
    ) -> Tuple[torch.Tensor,...]:
        """
        compute L = E[ln(x)] + beta * KL(q(z|x) || p(z))

        L = 1/B * sum_{b=1}^B [recon_loss_b + beta * kld_loss_b]
        recon_loss_b = sum_{i=1}^PXs MSE(x_ib, recon_ib)
        kld_loss_b   = sum_{k=1}^dim max(0, KL(q(z_b|x_b) || p(z)) - free_nats)
        
        recon_loss [1,] : MSE(x, recon) sum over pixels, average over batch
        kld_loss   [1,] : KL (nats) sum over latent dimension, average over batch 
        """
        recon_loss = F.mse_loss(recon, x, reduction="none")
        recon_loss = recon_loss.flatten(start_dim=1).sum(dim=1).mean()

        if self.beta == 0:
            zero = torch.tensor(0., device=x.device)
            return recon_loss, recon_loss, zero
        
        kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)   # shape (B,d)
        kl_after_free = torch.clamp(kl_dim - self.free_nats, min=0.0)
        kld = kl_after_free.sum(1).mean()
        
        total = recon_loss + self.beta * kld
        return total, recon_loss, kld

