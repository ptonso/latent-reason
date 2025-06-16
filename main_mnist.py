#!/usr/bin/env python

import os
from src.vae.model.config import *
from src.vae.model.beta_vae import BetaVAE

def main():

    encoder_cfg = EncoderConfig(
        in_channels=1,        # MNIST is grayscale
        channels=[32],        # one conv layer
        kernels=[4],
        strides=[2],
        paddings=[1],
        fc_layers=0,          # no hidden MLP layers, just final linear → 2·latent_dim
        fc_units=0,           # unused when fc_layers=0
        activation="relu",
        norm_type="batch"
    )

    decoder_cfg = DecoderConfig(
        out_channels=1,
        channels=[32],        # bottleneck channels
        kernels=[4],
        strides=[2],
        paddings=[1],
        fc_layers=0,
        fc_units=0,
        activation="relu",
        norm_type="batch"
    )

    # 4. VAE config with β=1.0
    vae_cfg = VAEConfig(
        latent_dim=8,
        img_size=28,
        beta=1.0,
        free_nats=0.0,
        encoder=encoder_cfg,
        decoder=decoder_cfg,
        device="auto"
    )

    train_cfg = TrainConfig(
        lr=1e-4,
        batch_size=512,
        max_epochs=100,
        warmup_epochs=10,
        patience=10,
        scheduler="plateau",
        scheduler_patience=5,
        scheduler_factor=0.5,
        min_lr=1e-6,

        experiment_name="beta_vae_mnist",
        project_name="mnist_vae",
        data_yaml="data/01--clean/mnist/data.yaml",

    )


    vae = BetaVAE(vae_cfg, in_height=28, in_width=28)
    trainer = vae.run(train_cfg=train_cfg)

if __name__ == "__main__":
    main()

