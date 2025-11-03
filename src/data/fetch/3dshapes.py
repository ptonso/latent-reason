import random
import requests
from pathlib import Path
from multiprocessing import Pool
import h5py
from PIL import Image
import yaml
from tqdm.auto import tqdm

FACTORS = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]

def download_3dshapes(output_dir: str = "data/00--raw/3dshapes", url: str = "https://storage.googleapis.com/3d-shapes/3dshapes.h5") -> Path:
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    dst = out_root / "3dshapes.h5"
    if dst.exists():
        return dst
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=128 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dst

def _init_worker(h5_path: str):
    global _IMAGES
    _IMAGES = h5py.File(h5_path, "r")["images"]

def _save_index(args):
    idx, out_file = args
    arr = _IMAGES[idx]
    Image.fromarray(arr).save(out_file, format="JPEG", quality=95, subsampling=0, optimize=False)

def clean_3dshapes(
    input_dir: str = "data/00--raw/3dshapes",
    output_dir: str = "data/01--clean/3dshapes",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int | None = None
) -> None:
    h5_file = Path(input_dir) / "3dshapes.h5"
    if not h5_file.exists():
        raise FileNotFoundError(h5_file)

    out_root = Path(output_dir)
    splits = ["train", "val", "test"]
    for s in splits:
        (out_root / s / "images").mkdir(parents=True, exist_ok=True)
        (out_root / s / "labels").mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_file, "r") as f:
        total = f["images"].shape[0]
        labels = f["labels"][:]

    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)

    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    split_idx = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }

    for split, idx_list in split_idx.items():
        lbl_dir = out_root / split / "labels"
        sorted_idx = sorted(idx_list)
        for i, j in enumerate(sorted_idx, start=1):
            line = " ".join(str(x) for x in labels[j].tolist())
            with open(lbl_dir / f"{i:06d}.txt", "w") as f:
                f.write(line + "\n")

    tasks = []
    for split, idx_list in split_idx.items():
        img_dir = out_root / split / "images"
        sorted_idx = sorted(idx_list)
        for i, j in enumerate(sorted_idx, start=1):
            out_path = img_dir / f"{i:06d}.jpg"
            tasks.append((j, str(out_path)))

    with Pool(processes=num_workers, initializer=_init_worker, initargs=(str(h5_file),)) as pool:
        list(tqdm(pool.imap_unordered(_save_index, tasks, chunksize=512), total=len(tasks), desc="Saving images"))

    data_cfg = {
        "train": str((out_root / "train" / "images")),
        "val": str((out_root / "val" / "images")),
        "test": str((out_root / "test" / "images")),
        "train_labels": str((out_root / "train" / "labels")),
        "val_labels": str((out_root / "val" / "labels")),
        "test_labels": str((out_root / "test" / "labels")),
        "factors": FACTORS,
    }
    with open(out_root / "data.yaml", "w") as f:
        yaml.safe_dump(data_cfg, f, sort_keys=False)

if __name__ == "__main__":
    raw_path = "data/00--raw/3dshapes"
    clean_path = "data/01--clean/3dshapes"
    download_3dshapes(raw_path)
    clean_3dshapes(
        input_dir=raw_path,
        output_dir=clean_path,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
        num_workers=6
    )
