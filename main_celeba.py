from src.vae.model.config import *
from src.vae.model.beta_vae import BetaVAE
from src.utils import set_seed

def main():

    set_seed(42, deterministic=False)

    enc = EncoderConfig(
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

    dec = DecoderConfig(
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
        device      = "cuda",
        encoder     = enc,
        decoder     = dec,
    )

    train_cfg = TrainConfig(
        lr              = 1e-3,
        batch_size      = 768,
        max_epochs      = 200,
        beta_warmup     = 30,
        patience        = 10,
        optimizer_type  = "adamw",
        weight_decay    = 1e-2,
        scheduler       = "onecycle",
        scheduler_kwargs = dict(
            max_lr           = 4e-3,
            pct_start        = 0.1,
            div_factor       = 25,
            final_div_factor = 1e4,
        ),
        num_workers     = 6,
        data_yaml       = "data/01--clean/celebA/data.yaml",
        project_name    = "celebA-vae",
        experiment_name = "beta5-fn.5",
    )

    vae = BetaVAE(vae_cfg)
    vae.run(train_cfg=train_cfg, resume="beta5-fn.5")

if __name__ == "__main__":
    main()
