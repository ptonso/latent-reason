# download data
import os
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image

# 150GB!!
# from huggingface_hub import snapshot_download
# dataset_path = snapshot_download("imagenet-1k", repo_type="dataset", cache_dir="data/imagenet")


def prepare_cifar100_imagefolder(root="data/cifar100"):
    if os.path.exists(os.path.join(root, "train")) and os.path.exists(os.path.join(root, "val")):
        print("CIFAR-100 ImageFolder already prepared.")
        return root

    print("Preparing CIFAR-100 in ImageFolder format...")
    os.makedirs(root, exist_ok=True)

    transform = transforms.Compose([transforms.Resize(224)])  # Resize here for simplicity

    train_set = datasets.CIFAR100(root="data", train=True, download=True)
    test_set = datasets.CIFAR100(root="data", train=False, download=True)

    def save_images(dataset, split):
        for idx, (img, label) in enumerate(tqdm(dataset, desc=f"Processing {split}")):
            cls_dir = os.path.join(root, split, str(label))
            os.makedirs(cls_dir, exist_ok=True)
            img = transform(img)
            img.save(os.path.join(cls_dir, f"{idx}.png"))

    save_images(train_set, "train")
    save_images(test_set, "val")
    return root

dataset_path = prepare_cifar100_imagefolder()