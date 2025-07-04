import torch
import numpy as np
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import *

import yaml
import torch.nn.functional as F
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

from src.vae.model.beta_vae import BetaVAE
from src.vae.model.config import VAEConfig, EncoderConfig, DecoderConfig
from src.vae.dataset import ReconstructionDataset


@dataclass
class LatentStats:
    mu_mean: np.ndarray
    mu_var:  np.ndarray
    kl_mean: np.ndarray


def imshow_tensor(ax, img_tensor):
    """
    Show a C×H×W tensor in [-1,1] or [0,1].  C==1 → grayscale, else RGB.
    """
    img = img_tensor.detach().cpu()
    # detect range: if values in [0,1], assume no scaling necessary; if in [-1,1], scale
    if img.min() >= 0 and img.max() <= 1:
        img = img.clamp(0, 1)
    else:
        img = (img + 1.0) / 2.0        # [-1,1] → [0,1]
        img = img.clamp(0, 1)
    if img.shape[0] == 1:
        ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
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

    vae = BetaVAE(vae_cfg).to(device).eval()

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
        recon_mu   = vae.decode_prob(mu) * 2.0 - 1.0                       # expected recon
        recon_zero = vae.decode_prob(torch.zeros_like(mu)) * 2.0 - 1.0     # baseline recon

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
        recon, _, _ = vae(imgs.to(device))          # forward returns logits in [0,1]
        recon = recon.detach().cpu()
    imgs = imgs.cpu()

    fig, ax = plt.subplots(2, n, figsize=(2*n, 4))
    for i in range(n):
        imshow_tensor(ax[0, i], imgs[i])
        imshow_tensor(ax[1, i], recon[i])
    plt.suptitle("Originals (top) • Reconstructions (bottom)")
    plt.show()


def latent_traverse(vae, loader, device, dims=3, steps=None, scale=1.0):
    """
    Traverse individual latent dimensions by ±k * std * scale.
    """
    if steps is None:
        steps = [-2, -1, 0, 1, 2]

    dims = float('inf') if dims < 0 else dims
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
            z[d] += k * sigma[d] * scale
            recon = vae.decode_prob(torch.tensor(z, device=device).unsqueeze(0))[0]
            recon = recon.detach().cpu()
            # show grayscale or RGB correctly
            if recon.shape[0] == 1:
                ax[d, j].imshow(recon.squeeze(), cmap="gray", vmin=0, vmax=1)
            else:
                ax[d, j].imshow(recon.permute(1,2,0).numpy().clip(0,1))
            ax[d, j].axis("off")
            if d == 0:
                ax[d, j].set_title(f"{k:+}", fontsize=8)
        ax[d, 0].set_ylabel(f"dim {d}", rotation=0, labelpad=20)
    plt.suptitle("Latent-dimension traversal")
    plt.show()


def plot_latent_stats_heatmap(stats: LatentStats) -> None:
    """
    Plot a heatmap of the three key statistics (μ mean, μ var, KL avg)
    across latent dimensions.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # stack into (3 × d) array
    arr = np.stack([stats.mu_mean, stats.mu_var, stats.kl_mean], axis=0)
    labels = ['μ mean', 'μ var', 'KL avg']
    d = arr.shape[1]

    fig, ax = plt.subplots(figsize=(max(8, d * 0.2), 3))
    im = ax.imshow(arr, aspect='auto', cmap='viridis')
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(d))
    ax.set_xticklabels(np.arange(d), rotation=90, fontsize=6)
    ax.set_xlabel('latent dimension')
    ax.set_title('Latent‐space statistics heatmap')
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05)
    cbar.ax.set_ylabel('value', rotation=270, labelpad=10)
    plt.tight_layout()
    plt.show()



def pca_traverse(vae, loader, device, n_pcs=3, steps=None, scale=1.0):
    """
    Traverse PCA principal components by ±k * std * scale.
    """
    if steps is None:
        steps = [-2, -1, 0, 1, 2]

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
            z = anchor + k * stds[i] * pcs[i] * scale
            recon = vae.decode_prob(torch.tensor(z, device=device).unsqueeze(0))[0]
            recon = recon.cpu()
            if recon.shape[0] == 1:
                ax[i, j].imshow(recon.squeeze(), cmap="gray", vmin=0, vmax=1)
            else:
                ax[i, j].imshow(recon.permute(1,2,0).numpy().clip(0,1))
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(f"{k:+}σ", fontsize=8)
        ax[i, 0].set_ylabel(f"PC {i+1}", rotation=0, labelpad=20)
    plt.suptitle("PCA traversal")
    plt.show()



def get_all_latents(
    vae: Any,
    loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Return posterior mu and logvar arrays for the entire loader."""
    mus, lvs = [], []
    vae.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            mu, lv = vae.encoder(xb)
            mus.append(mu.cpu().numpy())
            lvs.append(lv.cpu().numpy())
    return np.concatenate(mus, axis=0), np.concatenate(lvs, axis=0)


def compute_latent_stats(
    mu: np.ndarray,
    logvar: np.ndarray
) -> LatentStats:
    """Compute mean(mu), var(mu), and average KL per latent dim."""
    var   = np.exp(logvar)
    kl    = 0.5 * (mu**2 + var - logvar - 1)
    return LatentStats(
        mu_mean = mu.mean(axis=0),
        mu_var  = mu.var(axis=0),
        kl_mean = kl.mean(axis=0)
    )


def plot_pca_3d(mu: np.ndarray) -> None:
    """Scatter the first three PCA components of mu in 3D."""
    pca    = PCA(n_components=3)
    coords = pca.fit_transform(mu)
    fig    = plt.figure(figsize=(6, 6))
    ax     = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:, 0],
               coords[:, 1],
               coords[:, 2],
               s=2, alpha=.5)
    ax.set_title('3D PCA of latent μ')
    plt.tight_layout()
    plt.show()




def plot_pca_3d_interactive(mu: np.ndarray) -> None:
    """Interactive 3-D scatter of the first three PCA comps of μ."""
    pca = PCA(n_components=3)
    coords = pca.fit_transform(mu)
    fig = go.Figure(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(size=2, opacity=0.6),
        )
    )
    fig.update_layout(
        title="Interactive 3D PCA of latent μ",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.show()



def posterior_histogram(
    mu: np.ndarray,
    max_dims: Optional[int] = None,
    bins: int = 50,
    xlim: Tuple[float, float] = (-4, 4),
    ymax: float = 0.8
) -> None:
    """
    Plot per-dimension histograms of μ with a N(0,1) overlay,
    all sharing the same x- and y-axis scales.
    """
    d = min(max_dims, mu.shape[1])
    dims = list(range(d))

    # precompute histograms to find global max density
    max_density = 0.0
    hists = {}
    for dim in dims:
        counts, edges = np.histogram(mu[:, dim], bins=bins, range=xlim, density=True)
        hists[dim] = (counts, edges)
        max_density = min(ymax, max(max_density, counts.max()))

    # layout grid
    cols = min(4, len(dims))
    rows = (len(dims) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    axes = axes.flatten()

    # gaussian pdf for overlay
    x = np.linspace(xlim[0], xlim[1], 1000)
    pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    for i, dim in enumerate(dims):
        ax = axes[i]
        counts, edges = hists[dim]
        ax.bar(
            (edges[:-1] + edges[1:]) / 2,
            counts,
            width=(edges[1] - edges[0]),
            alpha=0.6,
            label="empirical",
        )
        ax.plot(x, pdf, linestyle="--", label="N(0,1)")
        ax.set_xlim(*xlim)
        ax.set_ylim(0, max_density * 1.05)
        ax.set_title(f"dim {dim}")
        ax.legend(fontsize=6)

    # remove unused axs
    for ax in axes[len(dims) :]:
        fig.delaxes(ax)

    plt.tight_layout()
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
