#!/usr/bin/env python3
import os, shutil, numpy as np, trimesh, pyrender
from kaggle.api.kaggle_api_extended import KaggleApi
from pyrender import MetallicRoughnessMaterial, Mesh as PyrenderMesh
from PIL import Image

# ─── PATHS ────────────────────────────────────────────────────────────────────
DATASET     = "balraj98/modelnet40-princeton-3d-object-dataset"
TARGET_DIR  = os.path.join("data", "00--raw", "modelnet40")
MESH_DIR    = os.path.join(TARGET_DIR, "ModelNet40")
RENDER_DIR  = os.path.join(TARGET_DIR, "renders")

# ─── RENDER SETTINGS ─────────────────────────────────────────────────────────
ANGLES      = list(range(0, 360, 30))     # 0,30,…,330°
RESOLUTION  = (256, 256)                  # output WxH
BG_COLOR    = [25, 25, 25, 255]           # dark-gray background RGBA

# ─── UTILS ───────────────────────────────────────────────────────────────────
def look_at(cam_pos, target=np.zeros(3), up=np.array([0, 1, 0])):
    """Return 4×4 camera pose that looks from cam_pos to target."""
    z = (cam_pos - target)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = cam_pos
    return pose

# ─── DOWNLOAD ────────────────────────────────────────────────────────────────
def download_modelnet40():
    if os.path.isdir(MESH_DIR) and os.listdir(MESH_DIR):
        print("✅ ModelNet40 already present – skipping download.")
        return

    api = KaggleApi(); api.authenticate()
    os.makedirs(TARGET_DIR, exist_ok=True)
    print("⏬ Downloading ModelNet40 …")
    api.dataset_download_files(DATASET, path=TARGET_DIR, unzip=True, quiet=False)

    # Kaggle may unzip into an extra folder – flatten it
    extra = os.path.join(TARGET_DIR, DATASET.split("/")[-1])
    if os.path.isdir(extra):
        for f in os.listdir(extra):
            shutil.move(os.path.join(extra, f), TARGET_DIR)
        shutil.rmtree(extra)

    if not os.path.isdir(MESH_DIR):
        raise RuntimeError("Download finished, but ModelNet40 folder not found.")
    print("✅ Meshes ready:", MESH_DIR)

# ─── RENDER ONE MESH ─────────────────────────────────────────────────────────
def render_mesh_views(off_path: str, out_dir: str):
    mesh = trimesh.load(off_path, force='mesh')
    mesh.apply_translation(-mesh.centroid)                 # center at origin
    scale = 1.0 / np.max(mesh.extents)                     # unit-normalize
    mesh.apply_scale(scale)

    material = MetallicRoughnessMaterial(
        baseColorFactor=[0.2, 0.6, 0.9, 1.0],              # light-blue
        metallicFactor=0.0, roughnessFactor=0.6
    )
    pmesh = PyrenderMesh.from_trimesh(mesh, material=material, smooth=False)

    scene = pyrender.Scene(bg_color=BG_COLOR, ambient_light=[0.15, 0.15, 0.15])
    mesh_node = scene.add(pmesh)

    # fixed camera straight in +Z, 2× object “radius” away
    radius = 2.0                                              # because we scaled to unit box
    cam_pos = np.array([0.0, 0.0, radius])
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4)
    cam_node = scene.add(cam, pose=look_at(cam_pos))

    # single directional light same place as camera
    scene.add(pyrender.DirectionalLight(color=[1.0]*3, intensity=3.0),
              pose=look_at(cam_pos))

    renderer = pyrender.OffscreenRenderer(*RESOLUTION)
    base = os.path.splitext(os.path.basename(off_path))[0]

    os.makedirs(out_dir, exist_ok=True)
    for az in ANGLES:
        # rotate mesh around Y axis
        R = trimesh.transformations.rotation_matrix(np.radians(az), [0, 1, 0])
        scene.set_pose(mesh_node, pose=R)

        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        Image.fromarray(color).save(os.path.join(out_dir, f"{base}_{az:03d}.png"))

    renderer.delete()

# ─── BATCH RENDER ALL MESHS ──────────────────────────────────────────────────
def batch_render():
    if not os.path.isdir(MESH_DIR):
        raise RuntimeError("Mesh directory missing; run download first.")

    print("⏬ Rendering all meshes …")
    for root, _, files in os.walk(MESH_DIR):
        for fn in files:
            if fn.lower().endswith('.off'):
                off = os.path.join(root, fn)
                class_name = os.path.relpath(off, MESH_DIR).split(os.sep)[0]
                out_cls   = os.path.join(RENDER_DIR, class_name)
                render_mesh_views(off, out_cls)
    print("✅ Finished – renders in:", RENDER_DIR)

# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    download_modelnet40()
    batch_render()
