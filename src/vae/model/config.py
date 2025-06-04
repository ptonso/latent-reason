from dataclasses import dataclass, field
from typing import List


@dataclass
class EncoderConfig():
    in_channels: int     = 3
    channels: List[int]  = field(default_factory=lambda: [  64, 128, 256, 512])
    kernels:  List[int]  = field(default_factory=lambda: [   4,   4,   4,   4])
    strides:  List[int]  = field(default_factory=lambda: [   2,   2,   2,   2])
    paddings: List[int]  = field(default_factory=lambda: [   1,   1,   1,   1])
    fc_layers:  int      = 2
    fc_units:   int      = 512
    activation: str      = "silu"
    norm_type:  str      = "layer"

@dataclass
class DecoderConfig():
    out_channels: int    = 3
    channels: List[int]  = field(default_factory=lambda:  [ 512, 256, 128,  64])
    kernels:  List[int]  = field(default_factory=lambda:  [   4,   4,   4,   4])
    strides:  List[int]  = field(default_factory=lambda:  [   2,   2,   2,   2])
    paddings: List[int]  = field(default_factory=lambda:  [   1,   1,   1,   1])
    fc_layers:    int    = 2
    fc_units:     int    = 512
    activation:   str    = "silu"
    norm_type:    str    = "layer"


@dataclass
class VAEConfig:
    # shared across VAE variants
    latent_dim:    int   = 64
    lr:            float = 5e-4
    batch_size:    int   = 256
    max_epochs:    int   = 150
    warmup_epochs: int   = 30
    patience:      int   = 10
    device:        str   = "cuda"

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    # β for β-VAE; plain VAE will ignore this (β=1)
    beta:        float = 0.8
    free_nats:   float = 1.0
