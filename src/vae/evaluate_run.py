
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

from src.vae.model.beta_vae import BetaVAE
from src.vae.model.config import VAEConfig, EncoderConfig, DecoderConfig
from src.vae.dataset import ReconstructionDataset


def imshow_tensor(ax, img_tensor):
    """
    Show a C×H×W tensor in [0,1].  C==1 → grayscale, else RGB.
    """
    img = img_tensor.detach().cpu()
    if img.shape[0] == 1:
        ax.imshow(img.squeeze(), cmap="gray")
    else:
        ax.imshow(img.permute(1, 2, 0))
    ax.axis("off")


def load_vae(run_dir: str):
    run = Path(run_dir)

    model_cfg = yaml.safe_load(open(run / "model_config.json"))
    train_cfg = yaml.safe_load(open(run / "train_config.json"))

    enc_cfg  = EncoderConfig(**model_cfg["encoder"])
    dec_cfg  = DecoderConfig(**model_cfg["decoder"])
    top_cfg  = {k: v for k, v in model_cfg.items() if k not in ("encoder", "decoder")}
    vae_cfg  = VAEConfig(**top_cfg, encoder=enc_cfg, decoder=dec_cfg)

    device = torch.device(
        "cuda" if (vae_cfg.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    vae = BetaVAE(
        vae_cfg,
        in_height=vae_cfg.img_size,
        in_width=vae_cfg.img_size,
    ).to(device).eval()

    ckpt = torch.load(run / "weights" / "best.pt", map_location=device)
    vae.load_state_dict(ckpt["model_state_dict"])
    return vae, train_cfg, device


def val_loader_from_yaml(data_yaml: str, img_size: int, bs: int, in_ch: int):
    cfg = yaml.safe_load(open(data_yaml))
    folder = cfg["val"]

    ds = ReconstructionDataset(
        data_path=folder,
        img_size=img_size,
        channels=in_ch,
        augment=False,
    )
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)


def basic_metrics(vae, loader, device):
    xb, _ = next(iter(loader))
    xb = xb.to(device)
    mu, logvar = vae.encoder(xb)

    with torch.no_grad():
        recon_mu   = vae.decode_prob(mu)                       # expected recon
        recon_zero = vae.decode_prob(torch.zeros_like(mu))     # baseline recon

    print(f"MSE(x, recon_mu):   {F.mse_loss(recon_mu, xb, reduction='mean'):.6f}")
    print(f"MSE(x, recon_zero): {F.mse_loss(recon_zero, xb, reduction='mean'):.6f}")

    kld = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(1)
    budgeted = torch.clamp(kld - vae.config.free_nats * vae.latent_dim, min=0.0)
    print(f"True KL  (mean):    {kld.mean():.6f}")
    print(f"Budgeted KL (mean): {budgeted.mean():.6f}\n")


def mu_variance(vae, loader, device):
    mus = []
    with torch.no_grad():
        for xb, _ in loader:
            mu, _ = vae.encoder(xb.to(device))
            mus.append(mu.detach().cpu())
    mus = torch.cat(mus)
    var = mus.var(0)
    print(f"μ-variance (mean over dims): {var.mean():.6f}")
    print("Top-10 dims by variance:     ",
          var.sort(descending=True)[0][:10].tolist())
    return mus.numpy()


def plot_recon(vae, loader, device, n=6):
    imgs, _ = next(iter(loader))
    with torch.no_grad():
        logits, _, _ = vae(imgs.to(device))          # forward returns logits
        recon = torch.sigmoid(logits).detach().cpu()          # → [0,1]
    imgs = imgs.cpu()

    fig, ax = plt.subplots(2, n, figsize=(2*n, 4))
    for i in range(n):
        imshow_tensor(ax[0, i], imgs[i])
        imshow_tensor(ax[1, i], recon[i])
    plt.suptitle("Originals (top) • Reconstructions (bottom)")
    plt.show()


def latent_traverse(vae, loader, device, dims=3, steps=None):
    if steps is None:
        steps = [-3, -2, -1, 0, 1, 2, 3]

    latent_dim = vae.latent_dim
    dims = min(dims, latent_dim)

    xb, _ = next(iter(loader))
    anchor_mu, _ = vae.encoder(xb.to(device))
    mu0   = anchor_mu[0].detach().cpu().numpy()
    sigma = anchor_mu.detach().cpu().numpy().std(0)

    fig, ax = plt.subplots(dims, len(steps),
                           figsize=(2*len(steps), 2*dims),
                           constrained_layout=True)
    for d in range(dims):
        for j, k in enumerate(steps):
            z = mu0.copy()
            z[d] += k * sigma[d]
            recon = vae.decode_prob(torch.tensor(z, device=device).unsqueeze(0))
            img   = recon[0].detach().cpu().permute(1, 2, 0).numpy()
            ax[d, j].imshow(img.clip(0, 1))
            ax[d, j].axis("off")
            if d == 0:
                ax[d, j].set_title(f"{k:+}", fontsize=8)
        ax[d, 0].set_ylabel(f"dim {d}", rotation=0, labelpad=20)
    plt.suptitle("Latent-dimension traversal")
    plt.show()


def pca_traverse(vae, loader, device, n_pcs=3, steps=None):
    if steps is None:
        steps = [-3, -1, 0, 1, 3]

    mus = []
    with torch.no_grad():
        for xb, _ in loader:
            mu, _ = vae.encoder(xb.to(device))
            mus.append(mu.cpu())
    mus = torch.cat(mus).numpy()

    pca = PCA(n_components=n_pcs).fit(mus)
    pcs, stds = pca.components_, np.sqrt(pca.explained_variance_)

    anchor = mus[np.random.randint(len(mus))]
    fig, ax = plt.subplots(n_pcs, len(steps),
                           figsize=(2*len(steps), 2*n_pcs),
                           constrained_layout=True)
    for i in range(n_pcs):
        for j, k in enumerate(steps):
            z = anchor + k * stds[i] * pcs[i]
            recon = vae.decode_prob(torch.tensor(z, device=device).unsqueeze(0))
            img   = recon[0].cpu().permute(1, 2, 0).numpy()
            ax[i, j].imshow(img.clip(0, 1))
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(f"{k:+}σ", fontsize=8)
        ax[i, 0].set_ylabel(f"PC {i+1}", rotation=0, labelpad=20)
    plt.suptitle("PCA traversal")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="runs/<project>/<run_name>")
    args = parser.parse_args()

    vae, train_cfg, device = load_vae(args.run_dir)

    val_loader = val_loader_from_yaml(
        data_yaml = train_cfg["data_yaml"],
        bs        = train_cfg["batch_size"],
        img_size  = vae.config.img_size,
        in_ch     = vae.config.encoder.in_channels,
    )

    print("\n== Basic metrics ==")
    basic_metrics(vae, val_loader, device)

    print("== μ-Variance ==")
    _ = mu_variance(vae, val_loader, device)

    print("\n== Reconstructions ==")
    plot_recon(vae, val_loader, device)

    print("\n== Latent-dimension traversal ==")
    latent_traverse(vae, val_loader, device, dims=3)

    print("\n== PCA traversal ==")
    pca_traverse(vae, val_loader, device, n_pcs=3)
