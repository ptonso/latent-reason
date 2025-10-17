from dataclasses import dataclass
#from typing import Mapping, Literal, Optional, Any, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

#from src.vae.types import GenLogits, GenTargets, Context
from src.vae.base import BaseCriterion, LossDict

#from src.vae.criteria.gaussian_loss import GaussianRecon
#from src.vae.criteria.mdl_loss import MDLRecon
#from src.vae.criteria.perceptual import PerceptualLoss
from src.vae.config import BetaVAECriterionConfig, GaussianReconConfig, MDLReconConfig


class SemiSupervisedLoss(BaseCriterion):
    """
    Pixel Reconstruction + KL + SemiSupervision
    """
    def __init__(self, cfg: BetaVAECriterionConfig):
        super().__init__()
        self.cfg = cfg
        self.beta = float(cfg.beta)
        