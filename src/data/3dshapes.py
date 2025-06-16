import os
import requests
from pathlib import Path
import h5py
from PIL import Image
import yaml
import random
from tqdm.auto import tqdm
from multiprocessing import Pool

def download_3dshapes(
    output_dir: str = "data/00--raw",
    url: str = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
) -> None:
    """
    Fetches 3dshapes.h5 from Google Cloud Storage into `output_dir/3dshapes.h5`.
    Uses HTTP streaming to avoid high memory usage.
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    dst = out_root / "3dshapes.h5"
    if dst.exists():
        print(f"✅ Already downloaded: {dst}")
        return

    print(f"⏬ Downloading 3D Shapes HDF5 from {url}…")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=4*1024*1024):
                if chunk:
                    f.write(chunk)
    print(f"💾 Saved to {dst}")

def _init_worker(h5_path: str):
    global _IMAGES
    _IMAGES = h5py.File(h5_path, "r")["images"]

def _save_index(args):
    idx, out_file = args
    arr = _IMAGES[idx]
    Image.fromarray(arr).save(out_file)

def clean_3dshapes(
    input_dir: str = "data/00--raw",
    output_dir: str = "data/01--clean",
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    seed: int = 42,
    num_workers: int = None
) -> None:
    """
    Loads `input_dir/3dshapes.h5`, splits images into train/val/test
    according to ratios, and parallel-saves PNGs using multiprocessing.
    Finally writes `output_dir/data.yaml`.
    """
    h5_file = Path(input_dir) / "3dshapes.h5"
    if not h5_file.exists():
        raise FileNotFoundError(f"Missing HDF5 at {h5_file}")

    out_root = Path(output_dir)
    splits = ["train", "val", "test"]
    for split in splits:
        (out_root / split).mkdir(parents=True, exist_ok=True)

    # Read and shuffle indices
    with h5py.File(h5_file, "r") as f:
        total = f["images"].shape[0]
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)

    n_train = int(total * train_ratio)
    n_val   = int(total * val_ratio)
    split_idx = {
        "train": indices[:n_train],
        "val":   indices[n_train:n_train + n_val],
        "test":  indices[n_train + n_val:]
    }

    # Prepare tasks
    tasks = []
    for split, idx_list in split_idx.items():
        dest = out_root / split
        print(f"💾 Queueing {len(idx_list)} images for {split}")
        for i, j in enumerate(idx_list):
            out_path = dest / f"{i:06d}.png"
            tasks.append((j, out_path))

    # Parallel save
    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(str(h5_file),)
    ) as pool:
        list(tqdm(pool.imap_unordered(_save_index, tasks), total=len(tasks), desc="Saving"))

    # Emit data.yaml
    cfg = {s: str(out_root / s) for s in splits}
    with open(out_root / "data.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"✅ 3D Shapes cleaned under {out_root} (data.yaml created)")



if __name__ == "__main__":

    raw_path = "data/00--raw/3dshapes"
    clean_path = "data/01--clean/3dshapes"


    download_3dshapes(output_dir=raw_path)
    clean_3dshapes(
        input_dir=raw_path,
        output_dir=clean_path,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
        num_workers=4
    )
