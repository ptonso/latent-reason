from src.vae.config import *
from src.vae.trainer import Trainer
from src.utils import set_seed
from src.vae.config import TrainConfig


def main():

    set_seed(42, deterministic=False)


    enc = CNNEncoderConfig(
        in_channels = 1, # MNIST is grayscale
        channels     = [  64, 128, 192],
        kernels      = [   4,   4,   3],
        strides      = [   2,   2,   1],
        paddings     = [   1,   1,   1],
        activation  = "silu",
        norm_type   = "layer",
    )


    neck = GaussianNeckConfig(
        latent_dim  = 8,
        fc_layers   = 1,
        fc_units    = 128,
        norm_type   = "layer",
        activation  = "silu",
        free_nats   = 0.5,
    )

    dec = CNNDecoderConfig(
        out_channels = 1,
        channels     = [ 128,  64,  32],
        kernels      = [   3,   4,   4],
        strides      = [   1,   2,   2],
        paddings     = [   1,   1,   1],
        activation   = "silu",
        norm_type    = "layer",
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
        img_size  = 28,
        enc_name  = "s32",
        neck_name = "z",
        encoder   = enc,
        neck      = neck,
        decoder   = dec,
        criterion = criterion,
    )


    train_cfg = TrainConfig(
        lr               = 1e-4,
        batch_size       = 1024,
        max_epochs       = 100,
        beta_warmup      = 30,
        patience         = 10,
        optimizer_type   = "adamw",
        weight_decay     = 1e-2,
        scheduler        = "onecycle",
        scheduler_kwargs = dict(
            max_lr           = 4e-3,
            pct_start        = 0.1,
            div_factor       = 25,
            final_div_factor = 1e4,
        ),
        num_workers      = 4,
        data_yaml        = "data/01--clean/mnist/data.yaml",
        project_name     = "mnist-vae",
        experiment_name  = "beta5-baseline",
    )

    Trainer.run(model_cfg=vae_cfg, train_cfg=train_cfg, resume=False)

if __name__ == "__main__":
    main()

