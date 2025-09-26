from src.vae.model.config import *
from src.vae.model.beta_vae import BetaVAE
from src.utils import set_seed

def main():

    set_seed(42, deterministic=False)

    enc = CNNEncoderConfig(
        in_channels  = 3,
        channels     = [  64, 128, 192, 256],
        kernels      = [   4,   4,   4,   4],
        strides      = [   2,   2,   2,   2],
        paddings     = [   1,   1,   1,   1],
        fc_layers    = 2,
        fc_units     = 512,
        activation   = "relu",
        norm_type    = "none",
    )

    dec = CNNDecoderConfig(
        out_channels = 3,
        channels     = [ 256, 192, 128,  64],
        kernels      = [   4,   4,   4,   4],
        strides      = [   2,   2,   2,   2],
        paddings     = [   1,   1,   1,   1],
        fc_layers    = 2,
        fc_units     = 512,
        activation   = "relu",
        norm_type    = "none",
    )

    vae_cfg = VAEConfig(
        latent_dim  = 64,
        img_size    = 64,
        beta        = 5.0,
        free_nats   = 0.5,
        recon_type  = "l2",
        device      = "cuda",
        encoder     = enc,
        decoder     = dec,
    )

    train_cfg = TrainConfig(
        lr               = 1e-3,
        batch_size       = 512,
        max_epochs       = 100,
        beta_warmup      = 20,
        patience         = 10,
        optimizer_type   = "adamw",
        weight_decay     = 1e-2,
        scheduler        = "cosine",
        num_workers     = 4,
        data_yaml       = "data/01--clean/celebA/data.yaml",
        project_name    = "celebA-vae",
        experiment_name = "beta0-baseline-cosine-2",
    )

    vae = BetaVAE(vae_cfg)
    vae.run(train_cfg=train_cfg, resume=False)

if __name__ == "__main__":
    main()
