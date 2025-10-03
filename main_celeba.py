from src.vae.config import *
from src.vae.trainer import Trainer
from src.utils import set_seed
from src.vae.config import TrainConfig


def main():

    set_seed(42, deterministic=False)


    enc = CNNEncoderConfig(
        in_channels = 3,
        channels     = [  64, 128, 192, 256],
        kernels      = [   4,   4,   4,   4],
        strides      = [   2,   2,   2,   2],
        paddings     = [   1,   1,   1,   1],
        activation  = "relu",
        norm_type   = "none",
    )


    neck = GaussianNeckConfig(
        latent_dim  = 64,
        fc_layers   = 2,
        fc_units    = 512,
        norm_type   = "relu",
        activation  = "none",
        free_nats   = 0.5,
    )

    dec = CNNDecoderConfig(
        out_channels = 3,
        channels     = [ 256, 192, 128,  64],
        kernels      = [   4,   4,   4,   4],
        strides      = [   2,   2,   2,   2],
        paddings     = [   1,   1,   1,   1],
        activation   = "relu",
        norm_type    = "none",
    )


    criterion = BetaVAECriterionConfig(
        beta         = 5.0,
        recon_type   = "l2",
        huber_delta  = 1.0,
        perc_source  = "lpips",
        perc_weight  = 1.0,
        pix_weight   = 1.0,
        perc_use_l1  = True,
        lpips_net    = "alex",
    )

    vae_cfg = BetaVAEConfig(
        img_size  = 64,
        enc_name  = "s32",
        neck_name = "z",
        encoder   = enc,
        neck      = neck,
        decoder   = dec,
        criterion = criterion,
    )


    train_cfg = TrainConfig(
        lr               = 1e-4,
        batch_size       = 512,
        max_epochs       = 100,
        beta_warmup      = 20,
        patience         = 10,
        optimizer_type   = "adamw",
        weight_decay     = 1e-2,
        scheduler        = "cosine",
        num_workers      = 4,
        data_yaml        = "data/01--clean/celebA/data.yaml",
        project_name     = "celebA-vae",
        experiment_name  = "beta5-perc",
    )

    Trainer.run(model_cfg=vae_cfg, train_cfg=train_cfg, resume=False)

if __name__ == "__main__":
    main()

