from src.vae.model.config import *
from src.vae.model.beta_vae import BetaVAE
from src.utils import set_seed

def main():

    set_seed(42, deterministic=False)

    enc = MLPEncoderConfig(
        in_channels=3,
        hidden=[2048, 1024, 512],
        activation="silu",
        norm_type="layer",
    )

    dec = MLPDecoderConfig(
        out_channels=3,
        hidden=[512, 1024, 2048],
        activation="silu",
        norm_type="layer",
    )


    vae_cfg = VAEConfig(
        latent_dim=64,
        img_size=64,
        beta=4.0,
        free_nats=2.0,
        device="auto",
        encoder=enc,
        decoder=dec,
    )

    vae = BetaVAE(vae_cfg)


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
        project_name    = "mnist-mlp",
        experiment_name = "beta5-baseline-onecycle",
    )
    
    vae = BetaVAE(vae_cfg)
    vae.run(train_cfg=train_cfg, resume=True)

if __name__ == "__main__":
    main()




