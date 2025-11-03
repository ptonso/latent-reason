import shutil
from pathlib import Path
import numpy as np
import yaml
from PIL import Image
import torch
from torchvision.datasets import MNIST


def download_mnist(output_dir: str = "data/00--raw/mnist") -> None:
    """
    Download the MNIST dataset (train+test) into <output_dir>/MNIST/
    using torchvision. Images will be stored in the processed .pt files.
    """
    raw_root = Path(output_dir)
    raw_root.mkdir(parents=True, exist_ok=True)

    # this will fetch and write into output_dir/MNIST/{raw,processed}
    MNIST(root=str(raw_root), train=True, download=True)
    MNIST(root=str(raw_root), train=False, download=True)

    print(f"✅ MNIST downloaded into {raw_root}")


def clean_mnist(
    input_dir: str = "data/00--raw/mnist",
    output_dir: str = "data/01--clean/mnist",
    train_frac: float = 0.8,
    val_frac: float   = 0.1,
    test_frac: float  = 0.1,
    seed: int         = 42
) -> None:
    """
    Load MNIST (both splits), combine into one big array,
    randomly split into train/val/test folders of PNGs,
    and write data.yaml with paths.
    """
    raw_root = Path(input_dir)
    # load via torchvision (uses processed .pt under raw_root/MNIST)
    ds_train = MNIST(root=str(raw_root), train=True,  download=False)
    ds_test  = MNIST(root=str(raw_root), train=False, download=False)

    # combine all images into a single numpy array
    all_imgs = torch.cat([ds_train.data, ds_test.data], dim=0).numpy()
    N = all_imgs.shape[0]

    # sanity check fractions
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1")

    # shuffle indices
    idxs = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idxs)

    # compute split sizes
    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)

    splits = {
        "train": idxs[:n_train],
        "val":   idxs[n_train:n_train + n_val],
        "test":  idxs[n_train + n_val:]
    }

    # make output dirs
    out_root = Path(output_dir)
    for split in splits:
        (out_root / split).mkdir(parents=True, exist_ok=True)

    # save each image as a PNG
    for split, indices in splits.items():
        split_dir = out_root / split
        print(f"↳ Writing {len(indices)} images to {split_dir}/")
        for idx in indices:
            arr = all_imgs[idx].astype(np.uint8)      # shape (28,28), values 0–255
            img = Image.fromarray(arr, mode="L")
            img.save(split_dir / f"{idx:05d}.png")

    # write data.yaml
    cfg = { split: str(out_root / split) for split in splits }
    with (out_root / "data.yaml").open("w") as f:
        yaml.safe_dump(cfg, f)

    print(f"✅ Cleaned MNIST ready at {out_root} (data.yaml created)")


if __name__ == "__main__":
    raw_path   = "data/00--raw/mnist"
    clean_path = "data/01--clean/mnist"

    download_mnist(output_dir=raw_path)
    clean_mnist(
        input_dir=raw_path, 
        output_dir=clean_path,
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
        seed=42
    )
