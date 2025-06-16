#!/usr/bin/env python3
import os
import requests
import zipfile
import shutil

# URL for COIL-100 dataset (publicly accessible ZIP) :contentReference[oaicite:0]{index=0}
DATASET_URL = 'https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip'

# Where to place the images
TARGET_DIR = os.path.join('data', 'coil100')

def download_coil100():
    # 1. Ensure target directory exists
    os.makedirs(TARGET_DIR, exist_ok=True)
    zip_path = os.path.join(TARGET_DIR, 'coil-100.zip')

    # 2. Download ZIP if not already present
    if not os.path.exists(zip_path):
        print(f"⏬ Downloading COIL-100 to {zip_path} …")
        resp = requests.get(DATASET_URL, stream=True)
        resp.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("✔ Download complete.")
    else:
        print("✔ ZIP already downloaded, skipping.")

    # 3. Extract
    print("📦 Extracting archive…")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(TARGET_DIR)

    # 4. Flatten if the ZIP created a single subdirectory
    entries = [e for e in os.listdir(TARGET_DIR) if e != 'coil-100.zip']
    if len(entries) == 1 and os.path.isdir(os.path.join(TARGET_DIR, entries[0])):
        subdir = os.path.join(TARGET_DIR, entries[0])
        for item in os.listdir(subdir):
            shutil.move(os.path.join(subdir, item), TARGET_DIR)
        shutil.rmtree(subdir)
        print("↳ Flattened subdirectory structure.")

    # 5. Final check
    if not any(fname.lower().endswith(('.png','.jpg','.ppm','.bmp','.gif')) 
               for fname in os.listdir(TARGET_DIR)):
        raise RuntimeError(f"No image files found in {TARGET_DIR} after extraction.")
    print(f"✅ COIL-100 is ready at {TARGET_DIR}")

if __name__ == "__main__":
    download_coil100()
