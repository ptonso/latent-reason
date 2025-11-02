import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from torch import Tensor
from torch.utils.data import DataLoader

from src.vae.beta_vae import BetaVAE
from src.vae.config import BetaVAEConfig, TrainConfig
from src.vae.dataset import ReconstructionDataset
from src.vae.hydra_io import load_config


@dataclass(frozen=True)
class LatentStats:
    mu_mean: np.ndarray
    mu_var: np.ndarray
    kl_mean: np.ndarray


def _save_fig(fig: plt.Figure, save_dir: Optional[str | Path], name: str) -> None:
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {path}")
    plt.show()


def _save_yaml(data: Mapping[str, Any], save_dir: Optional[str | Path], name: str) -> None:
    if save_dir is None:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def _save_plotly_html(fig: go.Figure, save_dir: Optional[str | Path], name: str) -> None:
    if save_dir is None:
        fig.show()
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{name}.html"
    fig.write_html(str(path), include_plotlyjs="cdn")


def imshow_tensor(ax: Any, img_tensor: Tensor) -> None:
    img = img_tensor.detach().cpu()
    if img.min() >= 0 and img.max() <= 1:
        img = img.clamp(0, 1)
    else:
        img = (img + 1.0) / 2.0
        img = img.clamp(0, 1)
    if img.shape[0] == 1:
        ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(img.permute(1, 2, 0))
    ax.axis("off")


def load_vae(run_dir: str) -> Tuple[BetaVAE, TrainConfig, torch.device]:
    run = Path(run_dir)
    vae_cfg: BetaVAEConfig = load_config(run / "model_config.yaml", BetaVAEConfig)
    train_cfg: TrainConfig = load_config(run / "train_config.yaml", TrainConfig)
    device = torch.device(train_cfg.device)
    vae = BetaVAE(vae_cfg, device=device).eval()
    ckpt_path = run / "weights" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run / "weights" / "last.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    vae.load_state_dict(ckpt["model_state_dict"])
    return vae, train_cfg, device


def val_loader_from_yaml(
    data_yaml: str,
    img_size: int,
    in_ch: int,
    bs: int,
    *,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    cfg = OmegaConf.load(data_yaml)
    val_path = str(cfg["val"])
    ds = ReconstructionDataset(
        val_path,
        img_size=img_size,
        channels=in_ch,
        augment=False,
    )
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


def basic_metrics(vae: BetaVAE, loader: DataLoader, device: torch.device, save_dir: Optional[str | Path] = None) -> None:
    xb, _ = next(iter(loader))
    xb = xb.to(device)
    mu, logvar = vae.encode(xb)
    with torch.no_grad():
        recon_mu = vae.decode_prob(mu) * 2.0 - 1.0
        recon_zero = vae.decode_prob(torch.zeros_like(mu)) * 2.0 - 1.0

    mse_mu = float(F.mse_loss(recon_mu, xb, reduction="mean"))
    mse_zero = float(F.mse_loss(recon_zero, xb, reduction="mean"))
    kld = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(1)
    kl_mean = float(kld.mean().detach())

    print(f"MSE(x, recon_mu):   {mse_mu:.6f}")
    print(f"MSE(x, recon_zero): {mse_zero:.6f}")
    print(f"True KL  (mean):    {kl_mean:.6f}")

    _save_yaml(
        {"mse_recon_mu": mse_mu, "mse_recon_zero": mse_zero, "kl_mean": kl_mean},
        save_dir,
        "basic_metrics",
    )


def mu_variance(vae: BetaVAE, loader: DataLoader, device: torch.device, save_dir: Optional[str | Path] = None) -> np.ndarray:
    mus: list[Tensor] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            mu, _ = vae.encode(xb)
            mus.append(mu.detach().cpu())
    mu_all = torch.cat(mus)
    var = mu_all.var(0)
    print(f"μ-variance (mean over dims): {var.mean():.6f}")
    print("Top-10 dims by variance:     ", var.sort(descending=True)[0][:10].tolist())
    _save_yaml(
        {"mu_var_mean": float(var.mean()), "mu_var_top10": var.sort(descending=True)[0][:10].tolist()},
        save_dir,
        "mu_variance",
    )
    return mu_all.numpy()


def plot_recon(vae: BetaVAE, loader: DataLoader, device: torch.device, n: int = 6, save_dir: Optional[str | Path] = None) -> None:
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        mu, _ = vae.encode(imgs)
        recon = vae.decode_prob(mu).detach().cpu()
    imgs = imgs.detach().cpu()
    fig, ax = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        imshow_tensor(ax[0, i], imgs[i])
        imshow_tensor(ax[1, i], recon[i])
    plt.suptitle("Originals (top) • Reconstructions (bottom)")
    _save_fig(fig, save_dir, "plot_recon")


def latent_traverse(
    vae: BetaVAE,
    loader: DataLoader,
    device: torch.device,
    *,
    dims: int = 3,
    steps: Optional[Sequence[int]] = None,
    scale: float = 1.0,
    save_dir: Optional[str | Path] = None,
) -> None:
    if steps is None:
        steps = [-2, -1, 0, 1, 2]
    dims = vae.latent_dim if dims < 0 else min(dims, vae.latent_dim)
    xb, _ = next(iter(loader))
    xb = xb.to(device)
    anchor_mu, _ = vae.encode(xb)
    mu0 = anchor_mu[0].detach().cpu().numpy()
    sigma = anchor_mu.detach().cpu().numpy().std(0)
    fig, ax = plt.subplots(dims, len(steps), figsize=(2 * len(steps), 2 * dims), constrained_layout=True)
    for d in range(dims):
        for j, k in enumerate(steps):
            z = mu0.copy()
            z[d] += k * sigma[d] * scale
            recon = vae.decode_prob(torch.tensor(z, device=device).unsqueeze(0))[0].cpu()
            if recon.shape[0] == 1:
                ax[d, j].imshow(recon.squeeze(), cmap="gray", vmin=0, vmax=1)
            else:
                ax[d, j].imshow(recon.permute(1, 2, 0).detach().numpy().clip(0, 1))
            ax[d, j].axis("off")
            if d == 0:
                ax[d, j].set_title(f"{k:+}", fontsize=8)
        ax[d, 0].set_ylabel(f"dim {d}", rotation=0, labelpad=20)
    plt.suptitle("Latent-dimension traversal")
    _save_fig(fig, save_dir, "latent_traverse")


def pca_traverse(
    vae: BetaVAE,
    loader: DataLoader,
    device: torch.device,
    *,
    n_pcs: int = 3,
    steps: Optional[Sequence[int]] = None,
    scale: float = 1.0,
    save_dir: Optional[str | Path] = None,
) -> None:
    if steps is None:
        steps = [-2, -1, 0, 1, 2]
    mus: list[Tensor] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            mu, _ = vae.encode(xb)
            mus.append(mu.cpu())
    mu_np = torch.cat(mus).numpy()
    pca = PCA(n_components=n_pcs).fit(mu_np)
    pcs, stds = pca.components_, np.sqrt(pca.explained_variance_)
    anchor = mu_np[np.random.randint(len(mu_np))]
    fig, ax = plt.subplots(n_pcs, len(steps), figsize=(2 * len(steps), 2 * n_pcs), constrained_layout=True)
    for i in range(n_pcs):
        for j, k in enumerate(steps):
            z = anchor + k * stds[i] * pcs[i] * scale
            recon = vae.decode_prob(torch.tensor(z, device=device).unsqueeze(0))[0].cpu()
            if recon.shape[0] == 1:
                ax[i, j].imshow(recon.squeeze(), cmap="gray", vmin=0, vmax=1)
            else:
                ax[i, j].imshow(recon.permute(1, 2, 0).detach().numpy().clip(0, 1))
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(f"{k:+}σ", fontsize=8)
        ax[i, 0].set_ylabel(f"PC {i + 1}", rotation=0, labelpad=20)
    plt.suptitle("PCA traversal")
    _save_fig(fig, save_dir, "pca_traverse")


def get_all_latents(
    vae: BetaVAE,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    mus: list[np.ndarray] = []
    lvs: list[np.ndarray] = []
    vae.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            mu, lv = vae.encode(xb)
            mus.append(mu.cpu().numpy())
            lvs.append(lv.cpu().numpy())
    return np.concatenate(mus, axis=0), np.concatenate(lvs, axis=0)


def compute_latent_stats(mu: np.ndarray, logvar: np.ndarray, save_dir: Optional[str | Path] = None) -> LatentStats:
    var = np.exp(logvar)
    kl = 0.5 * (mu ** 2 + var - logvar - 1.0)
    stats = LatentStats(mu_mean=mu.mean(axis=0), mu_var=mu.var(axis=0), kl_mean=kl.mean(axis=0))
    _save_yaml({k: v.tolist() for k, v in asdict(stats).items()}, save_dir, "latent_stats")
    return stats


def plot_pca_3d(mu: np.ndarray, save_dir: Optional[str | Path] = None) -> None:
    pca = PCA(n_components=3)
    coords = pca.fit_transform(mu)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=2, alpha=0.5)
    ax.set_title("3D PCA of latent μ")
    plt.tight_layout()
    _save_fig(fig, save_dir, "pca_3d")


def plot_pca_3d_interactive(mu: np.ndarray, save_dir: Optional[str | Path] = None) -> None:
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
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    _save_plotly_html(fig, save_dir, "pca_3d_interactive")


def posterior_histogram(
    mu: np.ndarray,
    max_dims: Optional[int] = None,
    bins: int = 50,
    xlim: Tuple[float, float] = (-4.0, 4.0),
    ymax: float = 0.8,
    save_dir: Optional[str | Path] = None,
) -> None:
    d_total = mu.shape[1]
    d = d_total if max_dims is None else min(max_dims, d_total)
    dims = list(range(d))
    max_density: float = 0.0
    hists: dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for dim in dims:
        counts, edges = np.histogram(mu[:, dim], bins=bins, range=xlim, density=True)
        hists[dim] = (counts, edges)
        max_density = min(ymax, max(max_density, float(counts.max())))
    cols = min(4, len(dims))
    rows = (len(dims) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    axes = np.atleast_1d(axes).ravel()
    x = np.linspace(xlim[0], xlim[1], 1000)
    pdf = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
    for i, dim in enumerate(dims):
        ax = axes[i]
        counts, edges = hists[dim]
        centers = (edges[:-1] + edges[1:]) / 2.0
        ax.bar(centers, counts, width=(edges[1] - edges[0]), alpha=0.6, label="empirical")
        ax.plot(x, pdf, linestyle="--", label="N(0,1)")
        ax.set_xlim(*xlim)
        ax.set_ylim(0.0, max_density * 1.05)
        ax.set_title(f"dim {dim}")
        ax.legend(fontsize=6)
    for ax in axes[len(dims):]:
        fig.delaxes(ax)
    plt.tight_layout()
    _save_fig(fig, save_dir, "posterior_histogram")


def plot_latent_stats_heatmap(stats: LatentStats, save_dir: Optional[str | Path] = None) -> None:
    arr: np.ndarray = np.stack([stats.mu_mean, stats.mu_var, stats.kl_mean], axis=0)
    labels: Sequence[str] = ("μ mean", "μ var", "KL avg")
    d: int = arr.shape[1]
    fig, ax = plt.subplots(figsize=(max(8, d * 0.2), 3))
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(d))
    ax.set_xticklabels(np.arange(d), rotation=90, fontsize=6)
    ax.set_xlabel("latent dimension")
    ax.set_title("Latent-space statistics heatmap")
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.05)
    cbar.ax.set_ylabel("value", rotation=270, labelpad=10)
    plt.tight_layout()
    _save_fig(fig, save_dir, "latent_stats_heatmap")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="runs/<project>/<run_name>")
    parser.add_argument("--save_dir", type=str, default=None, help="optional directory to save figures and YAMLs")
    args = parser.parse_args()

    vae, train_cfg, device = load_vae(args.run_dir)

    val_loader = val_loader_from_yaml(
        data_yaml=train_cfg.data_yaml,
        bs=train_cfg.batch_size,
        img_size=vae.config.img_size,
        in_ch=vae.config.encoder.in_channels,
    )

    print("\n== Basic metrics ==")
    basic_metrics(vae, val_loader, device, save_dir=args.save_dir)

    print("== μ-Variance ==")
    _ = mu_variance(vae, val_loader, device, save_dir=args.save_dir)

    print("\n== Reconstructions ==")
    plot_recon(vae, val_loader, device, save_dir=args.save_dir)

    print("\n== Latent-dimension traversal ==")
    latent_traverse(vae, val_loader, device, dims=3, save_dir=args.save_dir)

    print("\n== PCA traversal ==")
    pca_traverse(vae, val_loader, device, n_pcs=3, save_dir=args.save_dir)
