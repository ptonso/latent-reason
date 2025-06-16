import requests
import shutil
from pathlib import Path
import numpy as np
import yaml
from PIL import Image


def download_dsprites(output_dir: str = "data/00--raw") -> None:
    """
    Download the dSprites .npz archive into:
        <output_dir>/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
    Source: https://github.com/google-deepmind/dsprites-dataset
    """
    raw_root = Path(output_dir)
    raw_root.mkdir(parents=True, exist_ok=True)

    url = (
        "https://github.com/google-deepmind/dsprites-dataset/"
        "raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    )
    dest = raw_root / Path(url).name

    if dest.exists():
        print("dSprites archive already present, skipping download.")
        return

    print(f"⏬ Downloading dSprites archive to {dest}…")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    print("✅ Download complete.")


def clean_dsprites(
    input_dir: str = "data/00--raw/dsprites",
    output_dir: str = "data/01--clean/dsprites",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42
) -> None:
    """
    Load the full dSprites .npz, split into train/val/test,
    and write out each image as a PNG under:
        <output_dir>/train/*.png, val/*.png, test/*.png
    Plus a data.yaml pointing at those directories.
    """
    raw_root = Path(input_dir)
    npz_path = raw_root / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing archive: {npz_path}")

    # load arrays
    data = np.load(npz_path, allow_pickle=True, encoding="latin1")
    imgs = data["imgs"]            # shape (737280, 64, 64), values 0/1
    metadata = data["metadata"].item()

    # check fractions
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train+val+test fractions must sum to 1")

    # shuffle indices
    N = imgs.shape[0]
    idxs = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idxs)
    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)

    splits = {
        "train": idxs[:n_train],
        "val":   idxs[n_train:n_train + n_val],
        "test":  idxs[n_train + n_val:]
    }

    # make output folders
    out_root = Path(output_dir)
    for split in splits:
        (out_root / split).mkdir(parents=True, exist_ok=True)

    # save each image as PNG
    for split, indices in splits.items():
        split_dir = out_root / split
        print(f"↳ Writing {len(indices)} images to {split_dir}/")
        for idx in indices:
            arr = (imgs[idx] * 255).astype(np.uint8)       # 0 or 255
            img = Image.fromarray(arr, mode="L")           # grayscale
            # name by the original index, zero-padded
            fn = f"{idx:06d}.png"
            img.save(split_dir / fn)

    # write data.yaml
    cfg = { split: str(out_root / split) for split in splits }
    with (out_root / "data.yaml").open("w") as f:
        yaml.safe_dump(cfg, f)

    print(f"✅ Cleaned dSprites ready at {out_root} (data.yaml created)")


if __name__ == "__main__":
    raw_path   = "data/00--raw/dsprites"
    clean_path = "data/01--clean/dsprites"

    download_dsprites(output_dir=raw_path)
    clean_dsprites(
        input_dir=raw_path,
        output_dir=clean_path,
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
        seed=42
    )
