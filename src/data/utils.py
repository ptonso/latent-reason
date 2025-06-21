import pickle
from pathlib import Path
from typing import Dict

import lmdb
from tqdm import tqdm


def write_split_lmdb(samples: Dict[str, Path], lmdb_path: Path) -> None:
    """Write {filename: path} mapping to an LMDB of raw JPEG bytes."""
    env = lmdb.open(str(lmdb_path), map_size=50_000_000_000, subdir=False,
                    readonly=False, lock=True, meminit=False, map_async=True)
    with env.begin(write=True) as txn:
        for idx, (fname, fpath) in enumerate(tqdm(samples.items(),
                                                  desc=f"→ {lmdb_path.name}")):
            key = f"{idx:08}".encode()          # e.g. b"00001234"
            with open(fpath, "rb") as img_f:
                txn.put(key, img_f.read())
        # store a tiny metadata record (list of original names)
        meta = [fname for fname in samples.keys()]
        txn.put(b"__keys__", pickle.dumps(meta))
    env.sync(); env.close()
