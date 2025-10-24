
from dataclasses import dataclass, field
from typing import *

@dataclass
class CNNDecoderConfig:
    out_channels: int        = 3
    channels:     list[int]  = field(default_factory=lambda: [256, 192, 128, 64])
    kernels:      list[int]  = field(default_factory=lambda: [  4,   4,   4,  4])
    strides:      list[int]  = field(default_factory=lambda: [  2,   2,   2,  2])
    paddings:     list[int]  = field(default_factory=lambda: [  1,   1,   1,  1])
    activation:   str        = "silu"
    norm_type:    str        = "layer"

@dataclass
class GaussianNeckConfig:
    latent_dim:  int   = 64
    fc_layers:   int   = 2
    fc_units:    int   = 512
    norm_type:   str   = "layer"
    activation:  str   = "silu"
    free_nats:   float = 0.5

@dataclass
class CNNEncoderConfig:
    in_channels: int        = 3
    channels:    list[int]  = field(default_factory=lambda: [64, 128, 256,  512])
    kernels:     list[int]  = field(default_factory=lambda: [ 4,   4,   4,   4 ])
    strides:     list[int]  = field(default_factory=lambda: [ 2,   2,   2,   2 ])
    paddings:    list[int]  = field(default_factory=lambda: [ 1,   1,   1,   1 ])
    activation:  str        = "silu"
    norm_type:   str        = "layer"


@dataclass
class GaussianReconConfig:
    huber_delta: float      = 1.0
    recon_type:  Literal["l1","l2","smooth_l1"]  = "l2"


@dataclass
class MDLReconConfig:
    K: int        = 10

@dataclass
class PerceptualConfig:
    pix_weight:  float                             = 1.0
    perc_weight: float                             = 1.0
    use_l1: bool                                   = True
    source: Literal["none", "lpips", "encoder"]    = "none"
    lpips_net:   Literal["alex", "vgg", "squeeze"] = "alex"
    enc_layers: Optional[Mapping[str, float]]      = None

@dataclass
class SemiSupervisedConfig:
    ...

@dataclass
class BetaVAECriterionConfig:
    beta: float = 2.0
    perc: PerceptualConfig = field(default_factory=PerceptualConfig)
    recon: Union[
        GaussianReconConfig,
        MDLReconConfig
     ] = field(default_factory=GaussianReconConfig)
    ssuper: SemiSupervisedConfig = field(default_factory=SemiSupervisedConfig)
    

@dataclass
class BetaVAEConfig():
    img_size:  Union[int, Tuple[int, int]] = 64
    enc_name: str = "s32"
    neck_name: str = "z"
    encoder:   CNNEncoderConfig        = field(default_factory=CNNEncoderConfig)
    neck:      GaussianNeckConfig      = field(default_factory=GaussianNeckConfig)
    decoder:   CNNDecoderConfig        = field(default_factory=CNNDecoderConfig)
    criterion: BetaVAECriterionConfig  = field(default_factory=BetaVAECriterionConfig)



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
