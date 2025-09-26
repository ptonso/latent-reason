from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Iterable

import torch
import torch.nn as nn
from src.vae.types import FeatureDict, Context

LossDict = Mapping[str, torch.Tensor]


class BaseEncoder(nn.Module, ABC):
    """Maps raw inputs to a named FeatureDict; publishes channels and strides."""
    @abstractmethod
    def out_channels(self) -> Dict[str, int]:
        ...

    @abstractmethod
    def strides(self) -> Dict[str, int]:
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor, ctx: Context | None = None) -> FeatureDict:
        ...

    def names(self) -> Iterable[str]:
        return self.out_channels().keys()



class BaseNeck(nn.Module, ABC):
    """Transforms a FeatureDict into a FeatureDict; publishes channels and strides."""
    @abstractmethod
    def out_channels(self) -> Dict[str, int]:
        ...

    @abstractmethod
    def strides(self) -> Dict[str, int]:
        ...

    @abstractmethod
    def forward(self, feats: FeatureDict, ctx: Context | None = None) -> FeatureDict:
        ...



class BaseDecoder(nn.Module, ABC):
    """
    Base interface for task-specific decoders/heads.
    """

    @abstractmethod
    def expects(self) -> Optional[Dict[str, int]]:
        """
        declare required backbone/neck feature names and their channel counts.
        Optional[Dict[str, int]]: mapping feature_name -> channels.
        """
        ...

    @abstractmethod
    def forward(self, feats: FeatureDict, ctx: Optional[Context] = None) -> Any:
        """
        produce natural parameters ("logits") for the task.
        The exact structure of `logits` is task-defined
        """
        ...


class BaseCriterion(nn.Module, ABC):
    """
    Base interface that *bundles* loss (training) and link/post-proc (inference).
    """

    @abstractmethod
    def forward(self, logits: Any, target: Any, ctx: Optional[Context] = None) -> LossDict:
        """
        Compute the training objective from logits and targets.
        Must include key "loss": scalar Tensor on the correct device.
        May include extra logged terms (e.g., "loss/cls", "loss/box").
        """
        ...

    @abstractmethod
    def predict(self, logits: Any, ctx: Optional[Context] = None) -> Any:
        """
        Map logits to predictions in the target space (link/post-processing).
        Shape/structure is task-defined (e.g., argmax mask, NMS'ed boxes).
        """
        ...
