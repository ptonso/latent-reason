# src/vae/trainer.py

import os, json, time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, fields, is_dataclass

import yaml
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
    )
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from src.vae.model.config import TrainConfig, VAEConfig
from src.vae.dataset import ReconstructionDataset


class Trainer:
    """Universal trainer for reconstruction models"""

    def __init__(self, model: nn.Module, model_config: Any, training_config: TrainConfig):
        self.model        = model
        self.model_cfg    = model_config
        self.cfg          = training_config

        if self.cfg.cudnn_benchmark:
            cudnn.benchmark = True

        # Setup device
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)

        self.current_epoch    = 0
        self.best_fitness     = float('inf')
        self.best_epoch       = 0
        self.no_improve_count = 0

        self.px = ( # C x W x H
            self.model_cfg.encoder.in_channels 
            * self.model_cfg.img_size ** 2
            )

        self.train_losses = []
        self.val_losses   = []
        self.lr_history   = []
        self.epoch_times  = []


        self.use_amp: bool = self.cfg.mixed_precision
        if self.use_amp:
            self.scaler = GradScaler(self.device.type)

        self.setup_data()
        self.setup_optimizer(model)
        self.setup_scheduler()


    def setup_directories(self, resume: Optional[str] = None):
        """Create (or reuse) run/, weights/ and plots/ directories"""
        if resume:
            p = Path(resume)
            if p.suffix == ".pt" and p.exists():
                # e.g. runs/.../run_name/weights/last.pt
                self.run_dir = p.parent.parent
            elif p.is_dir():
                self.run_dir = p
            else:
                root = Path("runs") / self.cfg.project_name
                if not root.exists():
                    raise FileNotFoundError(f"{root} does not exist")
                matches = [d for d in root.iterdir()
                           if d.is_dir() and d.name == resume]
                if not matches:
                    raise FileNotFoundError(
                        f"No run named '{resume}' found in {root}")
                self.run_dir = matches[0]
        else:

            base_root = Path("runs") / self.cfg.project_name
            run_name  = self._get_next_run_name(base_root, self.cfg.experiment_name)
            self.run_dir = base_root / run_name
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_name    = self.run_dir.name
        self.weights_dir = self.run_dir / "weights"
        self.plots_dir   = self.run_dir / "plots"

        for d in (self.run_dir, self.weights_dir, self.plots_dir):
            d.mkdir(parents=True, exist_ok=True)
        

        print(f"📁 Experiment directory: {self.run_dir}")


    def setup_optimizer(self, model: nn.Module):
        opt_type = self.cfg.optimizer_type.lower()
        if opt_type == "adamw":
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=self.cfg.lr,
                betas=(0.9, 0.95),
                weight_decay=self.cfg.weight_decay,
            )
        elif opt_type == "adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.cfg.lr,
                betas=(0.9, 0.999),
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.optimizer_type}")

    def setup_scheduler(self):
        """Setup learning rate scheduler"""
        skw = self.cfg.scheduler_kwargs
        if self.cfg.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode      = "min",
                factor    = skw["scheduler_factor"],
                patience  = skw["scheduler_patience"],
                min_lr    = skw["min_lr"]
            )
        elif self.cfg.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max     = self.cfg.max_epochs,
                eta_min   = skw["min_lr"]
            )
        elif self.cfg.scheduler == "onecycle":
            steps = len(self.train_loader)
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr     = skw["max_lr"],
                epochs     = self.cfg.max_epochs,
                steps_per_epoch=steps,
                pct_start  = skw.get("pct_start", 0.3),
                div_factor = skw.get("div_factor", 25),
                final_div_factor = skw.get("final_div_factor", 1e2),
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
        """Train for one epoch, return dict with per-image & per-pixel metrics."""
        self.model.train()
        # warmup β
        if self.cfg.warmup_epochs>0:
            frac = min(1.0, self.current_epoch / self.cfg.warmup_epochs)
            self.model.beta = self.model_cfg.beta * frac
        sum_losses = {'total': 0.0, 'recon': 0.0, 'kld': 0.0}
        pbar = tqdm(self.train_loader,
                    desc=f"Epoch {self.current_epoch}/{self.cfg.max_epochs} [Train]",
                    leave=False)

        for x, _ in pbar:
            x = x.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with autocast(self.device.type, dtype=torch.bfloat16):
                    recon, mu, logvar = self.model(x)
                    total, recon_l, kld_l = self.model.loss_function(recon, x, mu, logvar)
                self.scaler.scale(total).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                recon, mu, logvar = self.model(x)
                total, recon_l, kld_l = self.model.loss_function(recon, x, mu, logvar)
                total.backward()
                self.optimizer.step()

                if torch.isnan(total) or torch.isnan(recon_l) or torch.isnan(kld_l):
                    print("‼️ NaN detected  | ",
                        f"recon_l: {recon_l},  ",
                        f"min/max recon: {recon.min():.3f}/{recon.max():.3f}")
                    raise RuntimeError(
                        "NaN detected in loss, stopping training. "
                        "Check your model and data preprocessing."
                    )
                    # optional: raise RuntimeError to stop immediately


            bs = x.size(0)
            # accumulate sums over samples
            sum_losses['total']  += total.item()    * bs 
            sum_losses['recon']  += recon_l.item()  * bs
            sum_losses['kld']    += kld_l.item()    * bs
            pbar.set_postfix(beta=self.model.beta)

        pbar.close()

        # compute averages per sample
        n = len(self.train_loader.dataset)
        avg_total       = sum_losses['total'] / n
        avg_recon_image = sum_losses['recon'] / n
        avg_kld_image   = sum_losses['kld']   / n

        return {
            'total':            avg_total,
            'recon_image':      avg_recon_image * self.px,
            'recon_pixel':      avg_recon_image,
            'kld_image':        avg_kld_image,
            'kld_dim':          avg_kld_image / self.model_cfg.latent_dim  
        }

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch, return dict with per-image & per-pixel metrics."""
        self.model.eval()
        sum_losses = {'total': 0.0, 'recon': 0.0, 'kld': 0.0}
        pbar = tqdm(self.val_loader,
                    desc=f"Epoch {self.current_epoch}/{self.cfg.max_epochs} [Val]",
                    leave=False)

        with torch.no_grad():
            for x, _ in pbar:
                x = x.to(self.device, non_blocking=True)
                recon, mu, logvar = self.model(x)
                total, recon_l, kld_l = self.model.loss_function(recon, x, mu, logvar)

                bs = x.size(0)

                sum_losses['total']  += total.item()   * bs
                sum_losses['recon']  += recon_l.item() * bs
                sum_losses['kld']    += kld_l.item()   * bs

            pbar.set_postfix()

        pbar.close()

        # compute averages per sample
        n = len(self.val_loader.dataset)
        avg_total       = sum_losses['total'] / n
        avg_recon_image = sum_losses['recon'] / n
        avg_kld_image   = sum_losses['kld']   / n

        return {
            'total'      : avg_total,
            'recon_image': avg_recon_image * self.px,
            'recon_pixel': avg_recon_image,
            'kld_image'  : avg_kld_image,
            'kld_dim'    : avg_kld_image / self.model_cfg.latent_dim
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
            'no_improve_count':     self.no_improve_count,
            'train_losses':         self.train_losses,
            'val_losses':           self.val_losses,
            'lr_history':           self.lr_history,
            'epoch_times':          self.epoch_times,
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
        self.no_improve_count = ckpt['no_improve_count']
        self.train_losses     = ckpt.get('train_losses', [])
        self.val_losses       = ckpt.get('val_losses', [])
        self.lr_history       = ckpt.get('lr_history', [])
        print(f"⚙️ Resumed from epoch {self.current_epoch}, "
              f"best_fitness={self.best_fitness:.4f}, "
              f"patience: {self.no_improve_count}/{self.cfg.patience}")
        return True

    def save_losses_csv(self):
        """Save training history to CSV"""
        if not self.train_losses or not self.val_losses:
            return
        rows = []
        for i, (tr, val, lr, t) in enumerate(zip(
            self.train_losses, 
            self.val_losses, 
            self.lr_history,
            self.epoch_times
        ), 1):
            row = {'epoch': i, 'lr': lr, 'epoch_time_s': t}
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
        self.setup_directories(resume)
        self.save_configs()

        print(f"🚀 Starting training: {self.run_name}")
        print(f"📱 Device: {self.device}")
        print(f"🎯 Total epochs: {self.cfg.max_epochs}")

        if resume:
            p = Path(resume)
            path = p if p.suffix==".pt" else self.weights_dir/"last.pt"
            if self.load_checkpoint(str(path)):
                print(f"🔄 Resumed from checkpoint: {path}")
            else:
                print(f"⚠️ Checkpoint not found: {path}. Starting from scratch.")

        start = time.time()
        try:
            for epoch in range(self.current_epoch + 1, self.cfg.max_epochs + 1):
                self.current_epoch = epoch

                epoch_start = time.time()

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
                self.epoch_times.append(time.time() - epoch_start)

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
                    f"Train img={tr['recon_image']:.4f} (px={tr['recon_pixel']:.6f}), "
                    f"KL img={tr['kld_image']:.4f} (dim={tr['kld_dim']:.6f}); "
                    f"Val   img={val['recon_image']:.4f} (px={val['recon_pixel']:.6f}), "
                    f"KL img={val['kld_image']:.4f} (dim={val['kld_dim']:.6f}); "
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


    def _get_next_run_name(self, base_root: str, experiment_name: str) -> str:
        base_root = Path(base_root)
        base_root.mkdir(parents=True, exist_ok=True)
        existing = {d.name for d in base_root.iterdir() if d.is_dir()}
        if experiment_name not in existing:
            return experiment_name
        suffixes = [0]
        prefix = "-".join(experiment_name.split("-")[:-1])
        for name in existing:
            if name.startswith(prefix):
                tail = name[len(prefix):]
                if tail.isdigit():
                    suffixes.append(int(tail))
        next_idx = max(suffixes) + 1
        return f"{experiment_name}-{next_idx}"