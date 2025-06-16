#!/usr/bin/env python

import os
from src.vae.model.config import *
from src.vae.model.beta_vae import BetaVAE

def main():

    encoder_cfg = EncoderConfig(
        in_channels = 3,
        channels   = [32, 32, 64, 64, 128],
        kernels    = [ 4,  4,  4,  4,   4],
        strides    = [ 2,  2,  2,  2,   2],
        paddings   = [ 1,  1,  1,  1,   1],
        fc_layers  = 2,
        fc_units   = 256,
        activation = "relu",
        norm_type  = "none",
    )

    decoder_cfg = DecoderConfig(
        out_channels= 3,
        channels    =  [128, 64, 64, 32, 32],
        kernels     =  [ 4,  4,  4,  4,  4],
        strides     =  [ 2,  2,  2,  2,  2],
        paddings    =  [ 1,  1,  1,  1,  1],
        fc_layers   = 2,
        fc_units    = 256,
        activation  = "relu",
        norm_type   = "none"        # optional: "group" for very small batches
    )


    vae_cfg = VAEConfig(
        latent_dim    = 20,
        img_size      = 64,
        beta          = 1.0, 
        # gamma         = 1000.0, 
        free_nats     = 0.0,
        device        = "cuda",

        encoder = encoder_cfg,
        decoder = decoder_cfg
    )

    train_cfg = TrainConfig(
        lr            = 1e-4,
        batch_size    = 1024,
        max_epochs    = 300,
        patience      = 30,
        scheduler     = "plateau",
    
        data_yaml = "data/01--clean/celebA/data.yaml",
        experiment_name="beta_vae_celebA",
        project_name="celebA_vae",
    )


    img_size = 64
    vae = BetaVAE(vae_cfg, in_height=img_size, in_width=img_size)
    trainer = vae.run(train_cfg=train_cfg)

if __name__ == "__main__":
    main()

