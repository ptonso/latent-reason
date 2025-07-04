from src.vae.model.config import *
from src.vae.model.beta_vae import BetaVAE
from src.utils import set_seed

def main():

    set_seed(42, deterministic=False)

    enc = EncoderConfig(
        in_channels  = 1,    # MNIST is grayscale
        channels     = [  32,  64, 128],
        kernels      = [   4,   4,   3],
        strides      = [   2,   2,   1],
        paddings     = [   1,   1,   1],
        fc_layers    = 1,
        fc_units     = 128,
        activation   = "relu",
        norm_type    = "none"

    )

    dec = DecoderConfig(
        out_channels = 1,
        channels     = [ 128,  64,  32],
        kernels      = [   3,   4,   4],
        strides      = [   1,   2,   2],
        paddings     = [   1,   1,   1],
        fc_layers    = 1,
        fc_units     = 128,
        activation   = "relu",
        norm_type    = "none"
    )

    vae_cfg = VAEConfig(
        latent_dim   =  8,
        img_size     = 28,
        beta         = 5.0,
        free_nats    = 0.5,
        device       = "auto",
        encoder      = enc,
        decoder      = dec,
    )

    train_cfg = TrainConfig(
        lr               = 1e-4,
        batch_size       = 2048,
        max_epochs       = 200,
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

        num_workers     = 4,
        data_yaml       = "data/01--clean/mnist/data.yaml",
        project_name    = "mnist-vae",
        experiment_name = "beta5-baseline",
    )


    vae = BetaVAE(vae_cfg)
    vae.run(train_cfg=train_cfg) # , resume="mnist"

if __name__ == "__main__":
    main()

