import shutil
import csv
from pathlib import Path
import yaml
from kaggle.api.kaggle_api_extended import KaggleApi


def download_celebA(output_dir: str = "data/00--raw") -> None:
    """
    Download and extract the CelebA dataset into:
        <output>/img_align_celeba/
    """
    raw_root = Path(output_dir)
    img_dir = raw_root / "img_align_celeba"
    api = KaggleApi()
    api.authenticate()

    raw_root.mkdir(parents=True, exist_ok=True)
    if not img_dir.exists():
        print(f"⏬ Downloading CelebA to {raw_root}…")
        api.dataset_download_files(
            "jessicali9530/celeba-dataset",
            path=str(raw_root),
            unzip=True,
            quiet=False
        )
        # Kaggle may unpack into a subfolder named "celeba-dataset"
        subdir = raw_root / "celeba-dataset"
        if subdir.exists():
            print("↳ Flattening directory structure…")
            for entry in subdir.iterdir():
                entry.rename(raw_root / entry.name)
            subdir.rmdir()

    if not (raw_root / "img_align_celeba").exists():
        raise RuntimeError(f"Missing images at {img_dir}")


def clean_celebA(
    input_dir: str = "data/00--raw",
    output_dir: str = "data/01--clean/celebA",
    partition_file: str = "list_eval_partition.csv"
) -> None:
    """
    Split CelebA images into train/val/test based on the CSV partition file,
    robustly finding .jpg files even if nested under img_align_celeba/.
    Then write data.yaml with keys train, val, test.
    """
    raw_root = Path(input_dir)
    part_path = raw_root / partition_file
    if not part_path.exists():
        raise FileNotFoundError(f"Partition file not found: {part_path}")

    # load partitions: filename → 0=train,1=val,2=test
    partitions: dict[str,int] = {}
    with part_path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for name, part in reader:
            partitions[name] = int(part)

    # find the image-folder root
    img_root = raw_root / "img_align_celeba"
    if not img_root.exists():
        raise FileNotFoundError(f"Couldn't find img_align_celeba under {raw_root}")

    # build a map: filename → full path, walking recursively
    jpg_map: dict[str, Path] = {
        p.name: p for p in img_root.rglob("*.jpg")
    }
    if not jpg_map:
        raise RuntimeError(f"No .jpg files found under {img_root}")

    # prepare output directories
    out_root = Path(output_dir)
    for split in ("train", "val", "test"):
        (out_root / split).mkdir(parents=True, exist_ok=True)

    # copy each file into its split
    split_names = {0: "train", 1: "val", 2: "test"}
    for fname, part in partitions.items():
        src = jpg_map.get(fname)
        if src is None or not src.exists():
            print(f"⚠️ warning: {fname} not found under {img_root}, skipping")
            continue
        dst = out_root / split_names[part] / fname
        shutil.copy2(src, dst)

    # write data.yaml
    cfg = { split: str(out_root / split) for split in ("train", "val", "test") }
    with (out_root / "data.yaml").open("w") as f:
        yaml.safe_dump(cfg, f)

    print(f"✅ Cleaned CelebA ready at {out_root} (data.yaml created)")


if __name__ == "__main__":
    raw_path = "data/00--raw/celebA"
    clean_path = "data/01--clean/celebA"

    download_celebA(output_dir=raw_path)
    clean_celebA(input_dir=raw_path, output_dir=clean_path)
