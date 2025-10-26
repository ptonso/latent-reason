from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
from PIL import Image
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.vae.types import GenTargets

def _convert_rgb(img):
    return img.convert("RGB")

_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

class ReconstructionDataset(Dataset):
    def __init__(
        self,
        data: Union[str, Dict],
        split: str = "train",
        img_size: int = 224,
        channels: int = 3,
        augment: bool = False,
        supervision_cap: float = 0.0,
    ):
        if isinstance(data, (str, Path)) and str(data).lower().endswith((".yml", ".yaml")):
            cfg = yaml.safe_load(Path(data).read_text())
            root = Path(cfg[split])
        elif isinstance(data, dict):
            root = Path(data[split])
        else:
            root = Path(data)

        self.img_size = img_size
        self.channels = channels
        self.augment = augment
        self.supervision_cap = float(supervision_cap)

        if root.is_dir() and (root / "images").is_dir():
            self.images_dir = root / "images"
            self.labels_dir = root / "labels" if (root / "labels").is_dir() else None
        elif root.is_dir():
            if root.name == "images" and root.parent.joinpath("labels").is_dir():
                self.images_dir = root
                self.labels_dir = root.parent / "labels"
            else:
                self.images_dir = root
                self.labels_dir = None
        else:
            self.images_dir = root
            self.labels_dir = None

        self.image_paths = self._find_images(self.images_dir)
        self.label_paths = self._match_labels(self.image_paths, self.labels_dir)
        self.label_dim = self._infer_label_dim()
        self._apply_supervision_cap()
        self._setup_transforms()

    def _find_images(self, d: Path) -> List[Path]:
        if d.is_file() and d.suffix.lower() in _EXT:
            return [d]
        out: List[Path] = []
        for ext in _EXT:
            out += list(d.rglob(f"*{ext}"))
            out += list(d.rglob(f"*{ext.upper()}"))
        return sorted(out)

    def _match_labels(self, imgs: List[Path], labels_dir: Optional[Path]) -> Dict[int, Optional[Path]]:
        lp: Dict[int, Optional[Path]] = {}
        if labels_dir is None:
            for i in range(len(imgs)):
                lp[i] = None
            return lp
        for i, p in enumerate(imgs):
            cand = labels_dir / f"{p.stem}.txt"
            lp[i] = cand if cand.exists() else None
        return lp

    def _infer_label_dim(self) -> Optional[int]:
        for p in self.label_paths.values():
            if p is None:
                continue
            try:
                a = np.loadtxt(p, ndmin=1)
                a = np.asarray(a, dtype=np.float32).reshape(-1)
                if a.size:
                    return int(a.size)
            except Exception:
                pass
        return None

    def _apply_supervision_cap(self):
        N = len(self.image_paths)
        labeled_idx = [i for i, lp in self.label_paths.items() if lp is not None]
        L = len(labeled_idx)
        if self.supervision_cap <= 0.0 or L == 0 or self.label_dim is None:
            self._is_labeled = set()
            return
        if self.supervision_cap >= 1.0:
            self._is_labeled = set(labeled_idx)
            return
        target = min(L, int(np.floor(self.supervision_cap * N)))
        self._is_labeled = set(sorted(labeled_idx)[:target])

    def _setup_transforms(self):
        tfms = [transforms.Resize((self.img_size, self.img_size))]
        if self.augment:
            tfms += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ]
        if self.channels == 1:
            tfms.append(transforms.Grayscale(1))
        else:
            tfms.append(transforms.Lambda(_convert_rgb))
        tfms.append(transforms.ToTensor())
        tfms.append(transforms.Normalize(mean=[0.5] * self.channels, std=[0.5] * self.channels))
        self.transform = transforms.Compose(tfms)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> GenTargets:
        img = Image.open(self.image_paths[index])
        x = self.transform(img) if self.transform else img

        arr = None
        if index in self._is_labeled and self.label_paths.get(index) is not None:
            try:
                a = np.loadtxt(self.label_paths[index], ndmin=1)
                arr = np.asarray(a, dtype=np.float32).reshape(-1)
            except Exception:
                arr = None

        if arr is None:
            if self.label_dim is None:
                arr = np.empty((0,), dtype=np.float32)
            else:
                arr = np.full((self.label_dim,), np.nan, dtype=np.float32)

        return GenTargets(img=x, meta={"labels": arr})
