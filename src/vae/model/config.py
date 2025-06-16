from dataclasses import dataclass, field
from typing import List

"""
User need to ensure that:
1. kernel, strides, padding of decoder to be a 1-to-1 map (not identical) to encoder.
2. no stride greater than current image size.
3. no kernel size greater than current image size.
"""


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
    latent_dim:    int     = 64
    img_size:      int     = 64

    beta:         float   = 2.0
    free_nats:    float   = 0.5 # /latent
    device:       str     = "cuda"

    encoder:       EncoderConfig = field(default_factory=EncoderConfig)
    decoder:       DecoderConfig = field(default_factory=DecoderConfig)



@dataclass
class TrainConfig:
    experiment_name:  str   = "vae_experiment"
    project_name:     str   = "vae_project"
    data_yaml:        str   = "data.yaml"

    lr:           float   = 5e-4
    batch_size:   int     = 256
    max_epochs:   int     = 150
    warmup_epochs:int     = 30
    patience:     int     = 10

    scheduler:          str   = "plateau"  # or "cosine"
    scheduler_patience: int   = 5
    scheduler_factor:   float = 0.5
    min_lr:             float = 1e-6

    num_workers:      int   = 4
    pin_memory:       bool  = True

    device:           str   = "auto"
    save_period:      int   = 1

    def __post_init__(self):
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"