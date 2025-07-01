from dataclasses import dataclass, field
from typing import *

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
    beta_warmup:  int     = 30
    patience:     int     = 10

    scheduler: Literal["plateau", "cosine", "onecycle"] = "plateau"
    scheduler_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "scheduler_patience": 5,
        "scheduler_factor": 0.5,
        "min_lr": 1e-6,
    })

    optimizer_type: str = "adam"    # "adam" or "adamw"
    weight_decay: float = 0.0       # only used if optimizer_type=="adamw"

    num_workers:      int   = 4
    pin_memory:       bool  = True

    device:           str   = "auto"
    save_period:      int   = 1

    def __post_init__(self):
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        defaults = {
            "plateau": {
                "scheduler_patience": 5,
                "scheduler_factor": 0.5,
                "min_lr": 1e-6,
            },
            "cosine": {
                "eta_min": 1e-6,
            },
            "onecycle": {
                "max_lr": self.lr,
                "pct_start": 0.3,
                "div_factor": 25,
                "final_div_factor": 1e2,
            },
        }
        base = defaults[self.scheduler].copy()
        base.update(self.scheduler_kwargs)
        self.scheduler_kwargs = base
