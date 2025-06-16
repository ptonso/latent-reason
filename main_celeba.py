from src.vae.model.config import *
from src.vae.model.beta_vae import BetaVAE

def main():
    enc = EncoderConfig(
        in_channels = 3,
        channels    = [  64, 128, 192, 256, 256],
        kernels     = [   4,   4,   4,   4,   4],
        strides     = [   2,   2,   2,   2,   2],
        paddings    = [   1,   1,   1,   1,   1],
        fc_layers   = 2,
        fc_units    = 512,
        activation  = "relu",
        norm_type   = "none",
    )

    dec = DecoderConfig(
        out_channels = 3,
        channels     = [ 256, 256, 192, 128,  64],
        kernels      = [   4,   4,   4,   4,   4],
        strides      = [   2,   2,   2,   2,   2],
        paddings     = [   1,   1,   1,   1,   1],
        fc_layers    = 2,
        fc_units     = 512,
        activation   = "relu",
        norm_type    = "none",
    )

    vae_cfg = VAEConfig(
        latent_dim = 32,
        img_size   = 64,
        beta       = 2.0,
        free_nats  = 0.5,
        device     = "cuda",
        encoder    = enc,
        decoder    = dec,
    )

    train_cfg = TrainConfig(
        lr              = 1e-3,
        batch_size      = 512,
        max_epochs      = 300,
        warmup_epochs   = 30,
        patience        = 30,
        scheduler       = "cosine",
        data_yaml       = "data/01--clean/celebA/data.yaml",
        experiment_name = "beta_vae_celebA",
        project_name    = "celebA_vae",
    )

    vae = BetaVAE(vae_cfg)
    vae.run(train_cfg=train_cfg, resume="celebA_vae/beta_vae_celebA_20250616_194845/weights/last.pt")

if __name__ == "__main__":
    main()
