import torch
from torch import nn
from torchvision import transforms
from dataclasses import dataclass
from typing import *


@dataclass
class DatasetConfig:
    name: str
    dataset_cls: Type[torch.utils.data.Dataset]
    transform: transforms.Compose
    input_size: Tuple[int, int, int]
    num_classes: int


@dataclass
class ModelConfig:
    name: str
    model_cls: Type[nn.Module]
    args: Dict

@dataclass
class BenchmarkConfig:
    name: str       = "baseline"
    device: str     = "cuda"
    seed: int       = [0, 1, 2]
    batch_size: int = 128
    epochs: int     = 20
    patience: int   = 5


@dataclass
class BenchmarkResult:
    method: str
    dataset: str
    mean: float
    std: float