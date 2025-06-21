import os, random
import numpy as np
import torch

def set_seed(seed: int, deterministic: bool = True):
    # 1) Python
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)

    # 2) NumPy
    np.random.seed(seed)

    # 3) PyTorch CPU & CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 4) cuDNN: deterministic algorithms, no auto-tuner
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark     = not deterministic

    # 5) Enforce PyTorch deterministic ops (>=1.8)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)