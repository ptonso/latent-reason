# src/vae/train.py

import os
import json
from tqdm.auto import tqdm
from dataclasses import is_dataclass, fields
import math

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from src.vae.model.config   import VAEConfig
from src.vae.model.beta_vae import BetaVAE


class Experiment:
    @staticmethod
    def dataclass2dict(obj):
        """Recursively turn dataclass instances → dict; modules → class-name."""
        if is_dataclass(obj):
            result = {}
            for f in fields(obj):
                value = getattr(obj, f.name)
                result[f.name] = Experiment.dataclass2dict(value)
            return result
        elif isinstance(obj, list):
            return [Experiment.dataclass2dict(v) for v in obj]
        elif isinstance(obj, torch.nn.Module):
            return obj.__class__.__name__
        else:
            return obj


def main():
    # 1) load config & device
    cfg    = VAEConfig()   # override defaults here if you like
    device = torch.device(cfg.device)

    # 2) early-stop & checkpoint settings
    patience    = cfg.patience
    experiment  = "modelnet40/beta.8_vae"
    model_name  = os.path.basename(experiment)
    ckpt_dir    = os.path.join("checkpoints", experiment)
    out_dir     = os.path.join("models", experiment)
    ckpt_path   = os.path.join(ckpt_dir, "checkpoint.pth")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # dump config JSON once (alongside your checkpoints)
    with open(os.path.join(ckpt_dir, "config.json"), "w") as fp:
        json.dump(Experiment.dataclass2dict(cfg), fp, indent=2)

    

    # 4) model, optimizer, scheduler
    model     = BetaVAE(cfg, in_height=height, in_width=width).to(device)
    opt       = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    # 5) resume if we have a checkpoint
    start_epoch   = 1
    best_val_loss = math.inf
    no_improve    = 0

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict( ckpt["optim_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"⚙️  Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # 6) training + early stopping
    for epoch in range(start_epoch, cfg.max_epochs + 1):
        # ——— train epoch
        curr_beta     = cfg.beta * min(1.0, epoch / cfg.warmup_epochs)
        model.beta    = curr_beta
        model.train()

        train_loss  = 0.0
        train_kld   = 0.0
        train_recon = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.max_epochs} [Train]", leave=False)
        for batch_idx, (x, _) in enumerate(pbar, start=1):
            x = x.to(device)
            xhat, mu, logvar = model(x)
            loss, recon, kld = model.loss_function(xhat, x, mu, logvar)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss  += loss.item()
            train_kld   += kld.item()
            train_recon += recon.item()
            pbar.set_postfix({
                'train_loss': train_loss / batch_idx,
                'train_kld':  train_kld  / batch_idx
            })
        pbar.close()

        n_train = len(train_loader.dataset)
        train_loss  /= n_train
        train_kld   /= n_train
        train_recon /= n_train

        train_weighted_kdl = model.beta * train_kld

        # ——— validation epoch
        model.eval()
        val_loss  = 0.0
        val_kld   = 0.0
        val_recon = 0.0
        pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.max_epochs} [Valid]", leave=False)
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(pbar, start=1):
                x = x.to(device)
                xhat, mu, logvar = model(x)
                loss, recon, kld = model.loss_function(xhat, x, mu, logvar)
                val_loss  += loss.item()
                val_kld   += kld.item()
                val_recon += recon.item()
                pbar.set_postfix({
                    'val_loss': val_loss / batch_idx,
                    'val_kld':  val_kld  / batch_idx
                })
        pbar.close()

        n_val = len(val_loader.dataset)
        val_loss  /= n_val
        val_kld   /= n_val
        val_recon /= n_val
        val_weighted_kdl = model.beta * val_kld
        scheduler.step(val_loss)
        current_lr = scheduler.get_last_lr()[0]
        avg_loss_per_pixel = val_loss / pixels_per_img

        print(f"Epoch {epoch:03d}  β={curr_beta:.3f}  LR={current_lr:.2e}")
        print(f"    Train: recon={train_recon:.4f}  KL×β={train_weighted_kdl:.4f}")
        print(f"    Valid: recon={val_recon:.4f}  KL×β={val_weighted_kdl:.4f} (per-pixel={avg_loss_per_pixel:.2f})")


        # ——— early-stop & checkpoint logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            print(f"    ↳ val improved! patience reset to 0/{patience}")
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optim_state":    opt.state_dict(),
                "best_val_loss":  best_val_loss
            }, ckpt_path)
        else:
            no_improve += 1
            print(f"    ↳ no_improve = {no_improve}/{patience}")
            if no_improve >= patience:
                print(f"🛑 Early stopping triggered (no improvement for {patience} epochs)")
                break

    # 7) finally rename the best checkpoint to your target name
    final_path = os.path.join(out_dir, f"{model_name}.pth")
    os.replace(ckpt_path, final_path)
    print(f"✅ Training complete. Best model saved to {final_path}")


if __name__ == "__main__":
    main()
