from dataclasses import is_dataclass, fields
from pathlib import Path
from typing import Any, Mapping
from omegaconf import OmegaConf
from hydra.utils import instantiate

def _to_hydra(obj: Any) -> Any:
    if is_dataclass(obj):
        d = {"_target_": f"{obj.__class__.__module__}.{obj.__class__.__name__}"}
        for f in fields(obj):
            d[f.name] = _to_hydra(getattr(obj, f.name))
        return d
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_hydra(x) for x in obj)
    if isinstance(obj, Mapping):
        return {k: _to_hydra(v) for k, v in obj.items()}
    return obj

def save_yaml(path: str | Path, obj: Any) -> None:
    cfg = OmegaConf.create(_to_hydra(obj))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(path))

def load_yaml(path: str | Path, *, recursive: bool = True) -> Any:
    cfg = OmegaConf.load(str(path))
    return instantiate(cfg, _recursive_=recursive)

def dataclass_to_hydra_dict(obj: Any) -> dict:
    return OmegaConf.to_container(OmegaConf.create(_to_hydra(obj)), resolve=True)


def load_config(path_yaml: Path, cls: type, *, recursive: bool = True) -> Any:
    if not path_yaml.exists():
        raise FileNotFoundError(f"config not found: {path_yaml}")
    cfg = OmegaConf.load(str(path_yaml))
    try:
        obj = instantiate(cfg, _recursive_=recursive)
        if isinstance(obj, cls):
            return obj
    except Exception:
        pass
    data = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(data, Mapping):
        return cls(**data)
    return data