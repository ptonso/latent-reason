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
