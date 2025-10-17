from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
import pandas as pd
import numpy as np


def convert_rgb(img):
    return img.convert("RGB")


class ReconstructionDataset(Dataset):
    """Dataset for image reconstruction tasks."""
    
    def __init__(
        self,
        data_path: str,
        labels_path: str = None,
        img_size: int = 224,
        channels: int = 3,
        augment: bool = False,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
        supervision_rate: float = 1.0
    ):
        self.data_path  = Path(data_path)
        self.has_labels = labels_path is not None
        self.labels_path = Path(labels_path) if self.has_labels else None
        self.img_size   = img_size
        self.channels   = channels
        self.augment    = augment
        self.extensions = extensions
        self.supervision_rate = supervision_rate

        self.empty_label = torch.zero(1)
        
        # Gather all image files
        self.image_paths = self._find_images()
        self.used_supervision = int(self.supervision_rate * len(self))

        # Gather_labels
        if self.has_labels:
            self.labels = self._find_labels()

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
                #image_paths += list(self.data_path.rglob(f"*{ext.upper()}"))
        return sorted(image_paths)
    
    def _find_labels(self) -> pd.DataFrame:
        labels = pd.read_csv(self.labels_path)
        labels = labels.query(f"ativo <= {self.used_supervision}")
        labels = labels.drop("ativo", axis=1)
        return labels
        
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
            tfms.append(transforms.Lambda(convert_rgb))

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

        if self.has_labels and index in self.labels:
            labels = np.array([self.labels_path[index, :]])
            is_labeled = True
        else:
            labels = self.empty_label
            is_labeled = False
            

        return img, labels, is_labeled
        #return img, 0 


