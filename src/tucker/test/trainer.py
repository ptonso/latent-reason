#!/usr/bin/env python3
from src.logger import setup_logger

import os
from typing import Type, Tuple, List, Dict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pandas as pd

from src.test.config import *

# module‐level logger
logger = setup_logger("api.log")



class EarlyStopper:
    """Stops training when validation accuracy plateaus."""
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_score: float = 0.0
        self.counter: int = 0

    def __call__(self, score: float) -> bool:
        if score <= self.best_score:
            self.counter += 1
            return self.counter >= self.patience
        self.best_score = score
        self.counter = 0
        return False


class Trainer:
    """Handles training, evaluation, checkpointing for one model+dataset."""
    def __init__(
        self,
        model_cfg: ModelConfig,
        data_cfg: DatasetConfig,
        device: torch.device,
        seeds: List[int]
    ):
        self.logger = logger
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.device = device
        self.seeds = seeds

    def get_data_loaders(
        self,
        batch_size: int
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and val loaders."""
        ds_cls = self.data_cfg.dataset_cls
        tf = self.data_cfg.transform

        train_ds = ds_cls(root="data",
                          train=True,
                          download=True,
                          transform=tf)
        val_ds   = ds_cls(root="data",
                          train=False,
                          download=True,
                          transform=tf)

        return (
            DataLoader(train_ds,
                       batch_size=batch_size,
                       shuffle=True,
                       num_workers=4),
            DataLoader(val_ds,
                       batch_size=batch_size,
                       shuffle=False,
                       num_workers=4)
        )

    def build_model(self) -> nn.Module:
        """Instantiate model on device."""
        model = self.model_cfg.model_cls(**self.model_cfg.args)
        return model.to(self.device)

    def train_and_evaluate(
        self,
        batch_size: int,
        epochs: int,
        patience: int
    ) -> List[float]:
        """Run multiple seeds, return list of best val accuracies."""
        results: List[float] = []

        for seed in self.seeds:
            torch.manual_seed(seed)

            train_loader, val_loader = \
                self.get_data_loaders(batch_size)

            model     = self.build_model()
            optimizer = optim.Adam(model.parameters())
            stopper   = EarlyStopper(patience)
            best_acc  = 0.0

            for epoch in range(1, epochs + 1):
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = model(xb)
                    loss   = nn.CrossEntropyLoss()(logits, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                val_acc = self.evaluate(model, val_loader)
                self.logger.info(
                    f"{self.model_cfg.name} | "
                    f"{self.data_cfg.name} | "
                    f"seed={seed} | epoch={epoch} | "
                    f"val_acc={val_acc:.4f}"
                )

                if val_acc > best_acc:
                    best_acc = val_acc
                    ckpt_dir = "checkpoints"
                    os.makedirs(ckpt_dir, exist_ok=True)
                    path = os.path.join(
                        ckpt_dir,
                        f"{self.model_cfg.name}_"
                        f"{self.data_cfg.name}_s{seed}.pt"
                    )
                    try:
                        torch.save(model.state_dict(), path)
                    except (OSError, IOError) as e:
                        self.logger.error(
                            f"saving failed: {e}"
                        )

                if stopper(best_acc):
                    self.logger.info(
                        f"early stopping at epoch {epoch}"
                    )
                    break

            results.append(best_acc)

        return results

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader
    ) -> float:
        """Compute accuracy over loader."""
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds  = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += yb.size(0)

        return correct / total if total else 0.0



