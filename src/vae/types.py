from collections import OrderedDict
from dataclasses import dataclass, field
from typing import NamedTuple, Any, Dict, List, Union, Optional, Tuple, OrderedDict as OD

import torch
from torch import Tensor


class GenTargets(NamedTuple):
    img:  torch.Tensor
    meta: Dict[str, Any] = field(default_factory=dict)

class GenLogits(NamedTuple):
    img:  torch.Tensor
    meta: Dict[str, Any] = field(default_factory=dict)

GenBatch = List[GenTargets]
GenOut   = List[torch.Tensor]

FeatureDict = OrderedDict[str, torch.Tensor]

@dataclass
class Context:
    """Side info and conditioning used along the network."""
    img_sizes: Optional[List[Tuple[int, int]]] = None
    attn_mask: Optional[torch.Tensor] = None
    timestep: Optional[torch.Tensor] = None
    text_emb: Optional[torch.Tensor] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelBatch:
    """Canonical batch: x, y, meta."""
    x: Union[Tensor, List[Tensor]]
    y: Dict[str, Any]
    meta: Optional[Dict[str, Any]] = None
