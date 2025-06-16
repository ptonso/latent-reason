# src/vae/trainer.py

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, fields, is_dataclass

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from src.vae.model.config import TrainConfig, VAEConfig
from src.vae.dataset import ReconstructionDataset


class Trainer:
    """Universal trainer for reconstruction models"""

    def __init__(self, model: nn.Module, model_config: Any, training_config: TrainConfig):
        self.model        = model
        self.model_cfg = model_config
        self.cfg          = training_config

        # Setup device
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)

        self.setup_directories()

        self.current_epoch    = 0
        self.best_fitness     = float('inf')
        self.best_epoch       = 0
        self.no_improve_count = 0
        self.global_iter      = 0

        self.train_losses = []
        self.val_losses   = []
        self.lr_history   = []

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.setup_scheduler()

        # Setup data loaders
        self.setup_data()

        # Save configurations
        self.save_configs()

    def setup_directories(self):
        """Setup directory structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name    = f"{self.cfg.experiment_name}_{timestamp}"
        self.run_dir     = Path(f"runs/{self.cfg.project_name}/{self.run_name}")
        self.weights_dir = self.run_dir / "weights"
        self.plots_dir   = self.run_dir / "plots"

        for d in (self.run_dir, self.weights_dir, self.plots_dir):
            d.mkdir(parents=True, exist_ok=True)

        print(f"📁 Experiment directory: {self.run_dir}")

    def setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.cfg.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.cfg.scheduler_factor,
                patience=self.cfg.scheduler_patience,
                min_lr=self.cfg.min_lr
            )
        elif self.cfg.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.max_epochs,
                eta_min=self.cfg.min_lr
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.cfg.scheduler}")

    def setup_data(self):
        """Setup data loaders from YAML config"""
        with open(self.cfg.data_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)

        n_ch = getattr(self.model_cfg.encoder, 'in_channels', 3)
        print(f"⚙️  Loading dataset with {n_ch} channels")

        train_ds = ReconstructionDataset(
            data_cfg['train'],
            img_size=self.model_cfg.img_size,
            channels=n_ch,
            augment=True
        )
        val_ds = ReconstructionDataset(
            data_cfg['val'],
            img_size=self.model_cfg.img_size,
            channels=n_ch,
            augment=False
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.num_workers > 0
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.num_workers > 0
        )

        print(f"📊 Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    def dataclass_to_dict(self, obj):
        """Convert dataclass to dictionary recursively"""
        if is_dataclass(obj):
            return {f.name: self.dataclass_to_dict(getattr(obj, f.name))
                    for f in fields(obj)}
        if isinstance(obj, list):
            return [self.dataclass_to_dict(o) for o in obj]
        if isinstance(obj, nn.Module):
            return obj.__class__.__name__
        return obj

    def save_configs(self):
        """Save training and model configurations"""
        # Training config
        with open(self.run_dir / "train_config.json", 'w') as f:
            json.dump(self.dataclass_to_dict(self.cfg), f, indent=2)
        # Model config
        with open(self.run_dir / "model_config.json", 'w') as f:
            json.dump(self.dataclass_to_dict(self.model_cfg), f, indent=2)
        # Data config
        with open(self.cfg.data_yaml, 'r') as f_src, \
             open(self.run_dir / "data_config.yaml", 'w') as f_dst:
            yaml.dump(yaml.safe_load(f_src), f_dst, default_flow_style=False)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch, return dict with sum & per-pixel metrics."""
        self.model.train()
        # warmup β
        if hasattr(self.model, 'beta') and hasattr(self.cfg, 'warmup_epochs'):
            frac = min(1.0, self.current_epoch / self.cfg.warmup_epochs)
            self.model.beta = self.model_cfg.beta * frac

        sum_losses = {'total': 0.0, 'recon': 0.0, 'kld': 0.0}
        pbar = tqdm(self.train_loader,
                    desc=f"Epoch {self.current_epoch}/{self.cfg.max_epochs} [Train]",
                    leave=False)

        for images, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            recon, mu, logvar = self.model(images)
            total, recon_l, kld_l = self.model.loss_function(
                recon, images, mu, logvar, self.global_iter)

            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()

            self.global_iter += 1

            bs = images.size(0)
            sum_losses['total'] += total.item()
            sum_losses['recon']  += recon_l.item()
            sum_losses['kld']    += kld_l.item() * bs

            avg = {k: v / (i+1) for i, (k, v) in enumerate(
                    zip(sum_losses.keys(), sum_losses.values()))}
            pbar.set_postfix(beta=self.model.beta)

        pbar.close()

        # compute averages per sample
        n = len(self.train_loader.dataset)
        avg_total = sum_losses['total'] / n
        avg_recon = sum_losses['recon']  / n
        avg_kld   = sum_losses['kld']    / n

        # per-pixel
        C = getattr(self.model_cfg.encoder, 'in_channels', 3)
        H = W = self.model_cfg.img_size
        px = C * H * W
        return {
            'total': avg_total,
            'recon': avg_recon,
            'kld':   avg_kld,
            'recon_per_pixel': avg_recon / px,
            'kld_per_pixel':   avg_kld   / px
        }

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch, return dict with sum & per-pixel metrics."""
        self.model.eval()
        sum_losses = {'total': 0.0, 'recon': 0.0, 'kld': 0.0}
        pbar = tqdm(self.val_loader,
                    desc=f"Epoch {self.current_epoch}/{self.cfg.max_epochs} [Val]",
                    leave=False)

        with torch.no_grad():
            for images, _ in pbar:
                images = images.to(self.device, non_blocking=True)
                recon, mu, logvar = self.model(images)
                total, recon_l, kld_l = self.model.loss_function(
                    recon, images, mu, logvar, self.global_iter)

                bs = images.size(0)
                sum_losses['total'] += total.item()
                sum_losses['recon']  += recon_l.item()
                sum_losses['kld']    += kld_l.item() * bs

            pbar.set_postfix()

        pbar.close()

        # compute averages per sample
        n = len(self.val_loader.dataset)
        avg_total = sum_losses['total'] / n
        avg_recon = sum_losses['recon']  / n
        avg_kld   = sum_losses['kld']    / n

        # per-pixel
        C = getattr(self.model_cfg.encoder, 'in_channels', 3)
        H = W = self.model_cfg.img_size
        px = C * H * W
        return {
            'total': avg_total,
            'recon': avg_recon,
            'kld':   avg_kld,
            'recon_per_pixel': avg_recon / px,
            'kld_per_pixel':   avg_kld   / px
        }

    def save_checkpoint(self, is_best: bool = False, is_last: bool = False):
        """Save model checkpoint"""
        ckpt = {
            'epoch':                self.current_epoch,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_fitness':         self.best_fitness,
            'best_epoch':           self.best_epoch,
            'train_losses':         self.train_losses,
            'val_losses':           self.val_losses,
            'lr_history':           self.lr_history,
            'model_config':         self.dataclass_to_dict(self.model_cfg),
            'training_config':      self.dataclass_to_dict(self.cfg)
        }

        if is_best:
            torch.save(ckpt, self.weights_dir / "best.pt")
            print(f"💾 Best model saved (epoch {self.current_epoch})")
        if is_last or (self.current_epoch % self.cfg.save_period == 0):
            torch.save(ckpt, self.weights_dir / "last.pt")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> bool:
        """Load checkpoint and resume training"""
        if not os.path.exists(checkpoint_path):
            return False
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.current_epoch    = ckpt['epoch']
        self.best_fitness     = ckpt['best_fitness']
        self.best_epoch       = ckpt['best_epoch']
        self.train_losses     = ckpt.get('train_losses', [])
        self.val_losses       = ckpt.get('val_losses', [])
        self.lr_history       = ckpt.get('lr_history', [])
        print(f"⚙️ Resumed from epoch {self.current_epoch}, best_fitness={self.best_fitness:.4f}")
        return True

    def save_losses_csv(self):
        """Save training history to CSV"""
        if not self.train_losses or not self.val_losses:
            return
        rows = []
        for i, (tr, val, lr) in enumerate(zip(self.train_losses, self.val_losses, self.lr_history), 1):
            row = {'epoch': i, 'lr': lr}
            for k, v in tr.items():
                row[f'train_{k}'] = v
            for k, v in val.items():
                row[f'val_{k}'] = v
            rows.append(row)
        pd.DataFrame(rows).to_csv(self.run_dir / "training_history.csv", index=False)

    def plot_losses(self):
        """Plot training and validation losses"""
        if not self.train_losses or not self.val_losses:
            return
        keys = list(self.train_losses[0].keys())
        epochs = list(range(1, len(self.train_losses) + 1))

        # plot each metric
        for k in keys:
            plt.figure()
            plt.plot(epochs, [l[k] for l in self.train_losses], 'o-', label=f'Train {k}')
            plt.plot(epochs, [l[k] for l in self.val_losses],   's-', label=f'Val {k}')
            plt.xlabel('Epoch')
            plt.ylabel(k)
            plt.title(f'{k} over epochs')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"{k}_loss.png", dpi=150)
            plt.close()

        # learning rate plot
        plt.figure()
        plt.plot(epochs, self.lr_history, 'o-')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.title('Learning Rate Schedule')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "learning_rate.png", dpi=150)
        plt.close()

    def train(self, resume: Optional[str] = None):
        """Main training loop"""
        print(f"🚀 Starting training: {self.run_name}")
        print(f"📱 Device: {self.device}")
        print(f"🎯 Total epochs: {self.cfg.max_epochs}")

        if resume:
            self.load_checkpoint(resume)

        start = time.time()
        try:
            for epoch in range(self.current_epoch + 1, self.cfg.max_epochs + 1):
                self.current_epoch = epoch

                tr = self.train_epoch()
                val = self.validate_epoch()

                # scheduler step
                if self.cfg.scheduler == "plateau":
                    self.scheduler.step(val['total'])
                else:
                    self.scheduler.step()

                self.train_losses.append(tr)
                self.val_losses.append(val)
                self.lr_history.append(self.optimizer.param_groups[0]['lr'])

                is_best = val['total'] < self.best_fitness
                if is_best:
                    self.best_fitness     = val['total']
                    self.best_epoch       = epoch
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1

                beta_info = f", β={self.model.beta:.3f}" if hasattr(self.model, 'beta') else ""
                print(
                    f"Epoch {epoch:3d}/{self.cfg.max_epochs}: "
                    f"Train sum={tr['recon']:.2f} (px={tr['recon_per_pixel']:.4f}), "
                    f"KL sum={tr['kld']:.2f} (px={tr['kld_per_pixel']:.4f}); "
                    f"Val sum={val['recon']:.2f} (px={val['recon_per_pixel']:.4f}), "
                    f"KL sum={val['kld']:.2f} (px={val['kld_per_pixel']:.4f}); "
                    f"LR={self.optimizer.param_groups[0]['lr']:.2e}{beta_info}"
                )

                if is_best:
                    print(f"    ↳ 🎯 New best! (patience: {self.no_improve_count}/{self.cfg.patience})")
                else:
                    print(f"    ↳ No improve (patience: {self.no_improve_count}/{self.cfg.patience})")

                self.save_checkpoint(is_best=is_best, is_last=True)

                if self.no_improve_count >= self.cfg.patience:
                    print(f"🛑 Early stopping after {epoch} epochs")
                    break

        except KeyboardInterrupt:
            print("\n⏸️ Training interrupted by user")

        finally:
            elapsed = time.time() - start
            print(f"⏱️ Training completed in {elapsed:.2f}s")
            print(f"🏆 Best epoch: {self.best_epoch} (fitness: {self.best_fitness:.4f})")
            self.save_losses_csv()
            self.plot_losses()
            print(f"📊 Results saved to: {self.run_dir}")




