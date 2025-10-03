import os, json, time
from pathlib import Path
from typing import *
from dataclasses import dataclass, fields, is_dataclass

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
)
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from src.vae.config import TrainConfig, BetaVAEConfig
from src.vae.beta_vae import BetaVAE
from src.vae.dataset import ReconstructionDataset


def _to_num(t: torch.Tensor) -> float: # robust .item() for bf16/fp16
    return t.float().item()

def _atomic_save(state: dict, path: Path):
    """ensure safe torch.save process"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)

def _default_run_dir(project: str, experiment: str) -> Path:
    return Path("runs") / project / experiment

class Trainer:
    """Universal trainer for reconstruction models"""

    def __init__(self, model: nn.Module, model_config: Any, training_config: TrainConfig):
        self.model        = model
        self.model_cfg    = model_config
        self.cfg          = training_config

        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)
        self.model._device = self.device

        self.current_epoch    = 0
        self.best_fitness     = float('inf')
        self.best_epoch       = 0
        self.no_improve_count = 0

        self.px = (
            self.model_cfg.encoder.in_channels * self.model_cfg.img_size ** 2
        )

        self.train_losses: list[Dict[str, float]] = []
        self.val_losses:   list[Dict[str, float]] = []
        self.lr_history:   list[float]            = []
        self.epoch_times:  list[float]            = []

        self.setup_data()
        self.setup_optimizer(model)
        self.setup_scheduler()

    @classmethod
    def run(
        cls, 
        model_cfg: BetaVAEConfig,
        train_cfg: TrainConfig, 
        resume:bool
        ):
        device = torch.device(train_cfg.device)
        model = BetaVAE(model_cfg, device=device)
        trainer = cls(model, model_cfg, train_cfg)
        trainer.train(resume=resume)
        

    def setup_directories(self, resume: bool = False):
        base_root = Path("runs") / self.cfg.project_name
        base_root.mkdir(parents=True, exist_ok=True)

        desired = _default_run_dir(self.cfg.project_name, self.cfg.experiment_name)

        if resume:
            if not desired.exists():
                raise FileNotFoundError(f"Resume dir not found: {desired}")
            self.run_dir = desired
        else:
            self.run_dir = Path(self._get_next_run_name(base_root, self.cfg.experiment_name))
            if not self.run_dir.is_absolute():
                self.run_dir = base_root / self.run_dir.name
            self.run_dir.mkdir(parents=True, exist_ok=True)

        self.run_name    = self.run_dir.name
        self.weights_dir = self.run_dir / "weights"
        self.plots_dir   = self.run_dir / "plots"
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Experiment directory: {self.run_dir}")

    def setup_optimizer(self, model: nn.Module):
        opt = self.cfg.optimizer_type.lower()
        lr = self.cfg.lr
        if self.cfg.scheduler == "onecycle":
            skw = self.cfg.scheduler_kwargs
            lr = skw["max_lr"] / skw.get("div_factor", 25)
        if opt == "adamw":
            self.optimizer = optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.95), 
                weight_decay=self.cfg.weight_decay
                )
        elif opt == "adam":
            self.optimizer = optim.Adam(
                model.parameters(), lr=lr, betas=(0.9, 0.999)
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.optimizer_type}")

    def setup_scheduler(self):
        skw = self.cfg.scheduler_kwargs

        if self.cfg.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min",
                factor   = skw.get("scheduler_factor", 0.5),
                patience = skw.get("scheduler_patience", 6),
                min_lr   = skw.get("min_lr", 1e-6),
            )
        elif self.cfg.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.cfg.max_epochs,
                eta_min=skw.get("min_lr", 1e-6)
            )
        elif self.cfg.scheduler == "onecycle":
            steps = len(self.train_loader)
            self.scheduler = OneCycleLR(
                self.optimizer, max_lr=skw.get("max_lr", 4e-3),
                epochs=self.cfg.max_epochs, steps_per_epoch=steps,
                pct_start        = skw.get("pct_start", 0.1),
                div_factor       = skw.get("div_factor", 25),
                final_div_factor = skw.get("final_div_factor", 1e4)
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.cfg.scheduler}")

    def setup_data(self):
        with open(self.cfg.data_yaml, "r") as f:
            data_cfg = yaml.safe_load(f)

        n_ch = getattr(self.model_cfg.encoder, "in_channels", 3)
        print(f"⚙️  Loading dataset with {n_ch} channels")

        train_ds = ReconstructionDataset(
            data_cfg["train"], img_size=self.model_cfg.img_size,
            channels=n_ch, augment=True
        )
        val_ds = ReconstructionDataset(
            data_cfg["val"], img_size=self.model_cfg.img_size,
            channels=n_ch, augment=False
        )

        self.train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=4,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=4,
        )

        print(f"📊 Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    def dataclass_to_dict(self, obj):
        if is_dataclass(obj):
            return {f.name: self.dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
        if isinstance(obj, list):
            return [self.dataclass_to_dict(o) for o in obj]
        if isinstance(obj, nn.Module):
            return obj.__class__.__name__
        return obj

    def save_configs(self):
        with open(self.run_dir / "train_config.json", "w") as f:
            json.dump(self.dataclass_to_dict(self.cfg), f, indent=2)
        with open(self.run_dir / "model_config.json", "w") as f:
            json.dump(self.dataclass_to_dict(self.model_cfg), f, indent=2)
        with open(self.cfg.data_yaml, "r") as src, \
             open(self.run_dir / "data_config.yaml", "w") as dst:
            yaml.dump(yaml.safe_load(src), dst)


    def train_epoch(self) -> Dict[str, float]:
        self.model.train()

        if self.cfg.beta_warmup > 0:
            base_beta: float = float(self.model.config.criterion.beta)
            frac: float = min(1.0, self.current_epoch / self.cfg.beta_warmup)
            beta_now: float = base_beta * frac
            self.model.crit.beta = beta_now

        s_total = torch.zeros((), device=self.device)
        s_recon = torch.zeros((), device=self.device)
        s_kld   = torch.zeros((), device=self.device)
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}/{self.cfg.max_epochs} [Train]", leave=False)

        for i, (x, _) in enumerate(pbar):
            x = x.to(self.device, non_blocking=True)
            ctx: Context = {}  # neck writes mu/logvar here

            logits: GenLogits = self.model(x, ctx)
            losses = self.model.crit(logits, x, ctx)

            total = losses['loss']
            recon = losses['loss/recon']
            kld   = losses['loss/kld']

            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            if self.cfg.scheduler == "onecycle":
                self.scheduler.step()

            bs = x.size(0)
            s_total = s_total + total.detach() * bs
            s_recon = s_recon + recon.detach() * bs
            s_kld   = s_kld   + kld.detach()   * bs
            
            if (i % 25) == 0:  # throttle UI updates
                pbar.set_postfix(beta=self.model.crit.beta)


        n  = len(self.train_loader.dataset)
        px = self.px
        ld = float(self.model.latent_dim)

        total_mean = (s_total / n).item()
        recon_mean = (s_recon / n).item()
        kld_mean   = (s_kld   / n).item()

        return {
            'total'       : total_mean,
            'recon_image' : recon_mean,
            'recon_pixel' : recon_mean / px,
            'kld_image'   : kld_mean,
            'kld_dim'     : kld_mean / ld,
        }


    def validate_epoch(self) -> Dict[str, float]:
        self.model.eval()

        s_total = torch.zeros((), device=self.device)
        s_recon = torch.zeros((), device=self.device)
        s_kld   = torch.zeros((), device=self.device)

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch}/{self.cfg.max_epochs} [Val]", leave=False)

        with torch.no_grad():
            for i, (x, _) in enumerate(pbar):
                x = x.to(self.device, non_blocking=True)
                ctx: Context = {}
                logits: GenLogits = self.model(x, ctx)
                losses = self.model.crit(logits, x, ctx)

                total = losses['loss']
                recon = losses['loss/recon']
                kld   = losses['loss/kld']

                bs = x.size(0)
                s_total = s_total + total * bs
                s_recon = s_recon + recon * bs
                s_kld   = s_kld   + kld   * bs
        

                if (i % 50) == 0:
                    pbar.set_postfix(beta=self.model.crit.cfg.beta)

        n  = len(self.val_loader.dataset)
        px = self.px
        ld = float(self.model.latent_dim)

        total_mean = (s_total / n).item()
        recon_mean = (s_recon / n).item()
        kld_mean   = (s_kld   / n).item()

        return {
            'total'       : total_mean,
            'recon_image' : recon_mean,
            'recon_pixel' : recon_mean / px,
            'kld_image'   : kld_mean,
            'kld_dim'     : kld_mean / ld,
        }


    def save_checkpoint(self, is_best: bool = False, is_last: bool = False):
        ckpt = {
            'epoch'               : self.current_epoch,
            'model_state_dict'    : self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_fitness'        : self.best_fitness,
            'best_epoch'          : self.best_epoch,
            'no_improve_count'    : self.no_improve_count,
            'train_losses'        : self.train_losses,
            'val_losses'          : self.val_losses,
            'lr_history'          : self.lr_history,
            'epoch_times'         : self.epoch_times,
            'model_config'        : self.dataclass_to_dict(self.model_cfg),
            'training_config'     : self.dataclass_to_dict(self.cfg)
        }
        if is_best:
            _atomic_save(ckpt, self.weights_dir / "best.pt")
            print(f"💾 Best model saved (epoch {self.current_epoch})")
        if is_last or (self.current_epoch % self.cfg.save_period == 0):
            _atomic_save(ckpt, self.weights_dir / "last.pt")

    def load_checkpoint(self, path: Union[str, Path]) -> bool:
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location=self.device)
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
        self.epoch_times      = ckpt.get('epoch_times', [])
        print(f"⚙️ Resumed from epoch {self.current_epoch}, best_fitness={self.best_fitness:.4f}, patience: {self.no_improve_count}/{self.cfg.patience}")
        return True


    def save_losses_csv(self):
        if not self.train_losses:
            return
        df_meta = pd.DataFrame({
            "epoch":        range(1, len(self.lr_history) + 1),
            "lr":           self.lr_history,
            "epoch_time_s": self.epoch_times
        })
        df_train = pd.DataFrame(self.train_losses).add_prefix("train_")
        df_val   = pd.DataFrame(self.val_losses).  add_prefix("val_")
        df = pd.concat([df_meta, df_train, df_val], axis=1)
        def fmt(v):
            return f"{v:.6f}" if isinstance(v, float) else str(v)

        str_df = df.map(fmt)

        col_widths = {
            col: max(str_df[col].str.len().max(), len(col)) + 2
            for col in str_df.columns
        }
        lines = []
        header = "".join(col.ljust(col_widths[col]) + "," for col in str_df.columns)
        lines.append(header)
        for _, row in str_df.iterrows():
            line = "".join(row[col].ljust(col_widths[col]) + "," for col in str_df.columns)
            lines.append(line)
        out_path = self.run_dir / "training_history.csv"
        with open(out_path, "w") as f:
            f.write("\n".join(lines))


    def plot_losses(self):
        if not self.train_losses or not self.val_losses:
            return
        keys   = list(self.train_losses[0].keys())
        epochs = list(range(1, len(self.train_losses) + 1))
        for k in keys:
            plt.figure()
            plt.plot(epochs, [l[k] for l in self.train_losses], 'o-', label=f'Train {k}')
            plt.plot(epochs, [l[k] for l in self.val_losses],   's-', label=f'Val {k}')
            plt.xlabel('Epoch'); plt.ylabel(k); plt.title(f'{k} over epochs')
            plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
            plt.savefig(self.plots_dir / f"{k}_loss.png", dpi=150); plt.close()

        plt.figure()
        plt.plot(epochs, self.lr_history, 'o-')
        plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.yscale('log')
        plt.title('Learning Rate Schedule'); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(self.plots_dir / "learning_rate.png", dpi=150); plt.close()


    def train(self, resume: bool = False):
        self.setup_directories(resume)
        self.save_configs()

        print(f"🚀 Starting training: {self.run_name}")
        print(f"📱 Device: {self.device}")
        print(f"🎯 Total epochs: {self.cfg.max_epochs}")

        if resume:
            path = self.weights_dir / "last.pt"
            if self.load_checkpoint(path):
                print(f"🔄 Resumed from checkpoint: {path}")
            else:
                print(f"⚠️ Checkpoint not found: {path}. Starting from scratch.")

        start = time.time()
        try:
            for epoch in range(self.current_epoch + 1, self.cfg.max_epochs + 1):
                self.current_epoch = epoch
                epoch_start = time.time()

                tr  = self.train_epoch()
                val = self.validate_epoch()

                if self.cfg.scheduler == "plateau":
                    self.scheduler.step(val['total'])
                elif self.cfg.scheduler == "cosine":
                    self.scheduler.step()

                self.train_losses.append(tr)
                self.val_losses.append(val)
                self.lr_history.append(self.optimizer.param_groups[0]['lr'])
                self.epoch_times.append(time.time() - epoch_start)

                is_best = val['total'] < self.best_fitness
                if is_best:
                    self.best_fitness = val['total']; self.best_epoch = epoch; self.no_improve_count = 0
                else:
                    self.no_improve_count += 1

                beta_info = f", β={self.model.crit.beta:.3f}"
                print(
                    f"Epoch {epoch:3d}/{self.cfg.max_epochs}: \n"
                    f"Train img={tr['recon_image']:.4f} (px={tr['recon_pixel']:.6f}), "
                    f"KL img={tr['kld_image']:.4f} (dim={tr['kld_dim']:.6f}); \n"
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
        base_root = Path(base_root); base_root.mkdir(parents=True, exist_ok=True)
        existing = {d.name for d in base_root.iterdir() if d.is_dir()}
        if experiment_name not in existing:
            return experiment_name
        nums = [0]
        pref = "-".join(experiment_name.split("-")[:-1])
        for n in existing:
            if n.startswith(pref):
                tail = n[len(pref):]
                if tail.isdigit(): nums.append(int(tail))
        return f"{experiment_name}-{max(nums)+1}"
