#!/usr/bin/env python3
import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA
from src.vae.model.config import VAEConfig

# where Kaggle will unzip everything
RAW_ROOT = os.path.join("data", "00--raw", "celeba")
IMG_DIR  = os.path.join(RAW_ROOT, "img_align_celeba")

def download_celebA():
    api = KaggleApi()
    api.authenticate()

    if not os.path.isdir(RAW_ROOT):
        os.makedirs(RAW_ROOT, exist_ok=True)
        print("⏬ Downloading CelebA via Kaggle…")
        api.dataset_download_files(
            "jessicali9530/celeba-dataset",
            path=RAW_ROOT,
            unzip=True,
            quiet=False
        )

        # Kaggle unzips into RAW_ROOT/celeba-dataset/
        subdir = os.path.join(RAW_ROOT, "celeba-dataset")
        if os.path.isdir(subdir):
            print("↳ Flattening directory structure…")
            for entry in os.listdir(subdir):
                shutil.move(os.path.join(subdir, entry), RAW_ROOT)
            shutil.rmtree(subdir)

    # final check
    if not os.path.isdir(IMG_DIR):
        raise RuntimeError(f"Missing images at {IMG_DIR}")

def configure_celebA(cfg: VAEConfig = VAEConfig()):
    height, width = 64, 64
    tf = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ])

    # now that everything’s in RAW_ROOT, no more download=True
    train_ds = CelebA(root=RAW_ROOT, split="train", download=False, transform=tf)
    val_ds   = CelebA(root=RAW_ROOT, split="valid", download=False, transform=tf)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,   batch_size=cfg.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader

if __name__ == "__main__":
    download_celebA()
    train, val = configure_celebA()
    print(f"👍 Loaded {len(train.dataset)} train / {len(val.dataset)} val samples")
