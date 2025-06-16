import torch
import torch.nn.functional as F
from src.vae.model.base_vae     import BaseVAE
from src.vae.model.encoder_decoder import Encoder, Decoder
from src.vae.model.config       import VAEConfig

class BetaVAE(BaseVAE):
    def __init__(self, vae_cfg: VAEConfig):
        super().__init__()
        self.config   = vae_cfg
        img_size      = vae_cfg.img_size
        self.beta     = vae_cfg.beta
        self.free_nats = vae_cfg.free_nats
        self.device   = vae_cfg.device

        self.encoder = Encoder(vae_cfg.encoder, img_size, img_size, vae_cfg.latent_dim)
        bot_h, bot_w = self.encoder.conv_out_h, self.encoder.conv_out_w
        self.decoder = Decoder(
            vae_cfg.decoder, bot_h, bot_w, img_size, img_size, vae_cfg.latent_dim
        )

        self.latent_dim = vae_cfg.latent_dim
        self.apply(self._init_weights)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decoder(z)
        return recon, mu, logvar


    def loss_function(
        self,
        recon:   torch.Tensor,   # logits B×C×H×W
        x:       torch.Tensor,   # targets in [0,1]
        mu:      torch.Tensor,   # B×d
        logvar:  torch.Tensor,   # B×d
    ):
        # 1 ─ reconstruction (mean over pixels & batch)
        recon_loss = F.binary_cross_entropy_with_logits(
            recon, x, reduction="mean"
        )

        if self.beta == 0:
            zero = torch.tensor(0., device=x.device)
            return recon_loss, zero, zero

        # 2 ─ per-latent KL: B×d
        kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)   # shape (B,d)

        # 3 ─ free-nats per latent
        kl_after_free = torch.clamp(kl_dim - self.free_nats, min=0.0)
        kld = kl_after_free.sum(1).mean()

        total = recon_loss + self.beta * kld
        return total, recon_loss, kld
