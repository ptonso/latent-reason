# src/vae/dataset.py

import os
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ReconstructionDataset(Dataset):
    """Dataset for image reconstruction tasks."""
    
    def __init__(
        self,
        data_path: str,
        img_size: int = 224,
        channels: int = 3,
        augment: bool = False,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    ):
        self.data_path  = Path(data_path)
        self.img_size   = img_size
        self.channels   = channels
        self.augment    = augment
        self.extensions = extensions
        
        # Gather all image files
        self.image_paths = self._find_images()
        # Build the transforms pipeline
        self._setup_transforms()
        
    def _find_images(self) -> List[Path]:
        """Recursively collect all image files under data_path."""
        image_paths: List[Path] = []
        if self.data_path.is_file():
            if self.data_path.suffix.lower() in self.extensions:
                image_paths.append(self.data_path)
        else:
            for ext in self.extensions:
                image_paths += list(self.data_path.rglob(f"*{ext}"))
                image_paths += list(self.data_path.rglob(f"*{ext.upper()}"))
        return sorted(image_paths)
        
    def _setup_transforms(self):
        """Construct a Compose pipeline that always yields `channels` output."""
        tfms = [
            transforms.Resize((self.img_size, self.img_size)),
        ]
        if self.augment:
            tfms += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1,
                    saturation=0.1, hue=0.05
                ),
            ]

        if self.channels == 1:
            tfms.append(transforms.Grayscale(1))
        else: 
            tfms.append(transforms.Lambda(lambda img: img.convert("RGB")))

        tfms.append(transforms.ToTensor())
        tfms.append(transforms.Normalize(
            mean=[0.5]*self.channels,
            std=[0.5]*self.channels)) # normalize to [-1, 1]

        self.transform = transforms.Compose(tfms)
            
    def __len__(self) -> int:
        return len(self.image_paths)
        
    def __getitem__(self, index: int):
        """Load an image (any mode) and convert it to `channels`-C, tensor in [-1,1]."""
        path = self.image_paths[index]
        img  = Image.open(path)          # no manual .convert( ) here
        if self.transform:
            img = self.transform(img)
        return img, 0 



import io, pickle, lmdb
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LMDBReconstructionDataset(Dataset):
    """
    LMDB-backed dataset for image reconstruction tasks.
    Stores raw image bytes under keys "00000000", "00000001", …,
    and a "__keys__" entry with the list of filenames (for counting).
    """

    def __init__(self,
                 lmdb_path: str,
                 img_size:  int  = 224,
                 channels:  int  = 3,
                 augment:   bool = False):
        self.lmdb_path = Path(lmdb_path)
        self.img_size  = img_size
        self.channels  = channels
        self.augment   = augment

        is_dir = self.lmdb_path.is_dir()
        self.env = lmdb.open(
            str(self.lmdb_path),
            readonly=True, lock=False, readahead=False, meminit=False,
            subdir=is_dir, max_readers=64
        )

        # use __keys__ only to get the number of samples
        with self.env.begin() as txn:
            meta = txn.get(b"__keys__")
            if meta:
                filenames = pickle.loads(meta)
                self.num_samples = len(filenames)
            else:
                # fallback: count all non-meta keys
                all_keys = [k for k, _ in txn.cursor() if k != b"__keys__"]
                self.num_samples = len(all_keys)

        # build numeric key list: "00000000", "00000001", ...
        self.keys = [f"{i:08}" for i in range(self.num_samples)]

        self._setup_transforms()

    def _setup_transforms(self):
        tfms = [transforms.Resize((self.img_size, self.img_size))]
        if self.augment:
            tfms += [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            ]
        if self.channels == 1:
            tfms.append(transforms.Grayscale(1))
        else:
            tfms.append(transforms.Lambda(lambda img: img.convert("RGB")))
        tfms += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*self.channels,
                                 std=[0.5]*self.channels),
        ]
        self.transform = transforms.Compose(tfms)

    def __len__(self) -> int:
        return self.num_samples

    def _worker_init(self, worker_id: int):
        # one long-lived transaction per worker
        self.txn = self.env.begin(buffers=True)

    def __getitem__(self, index: int):
        # for num_workers=0, init on first call
        if not hasattr(self, "txn"):
            self.txn = self.env.begin(buffers=True)

        key = self.keys[index].encode()   # b"00000000", etc.
        buf = self.txn.get(key)
        if buf is None:
            raise KeyError(f"Key {key!r} not found in {self.lmdb_path}")
        img = Image.open(io.BytesIO(buf))
        if self.transform:
            img = self.transform(img)
        return img, 0
