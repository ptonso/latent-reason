from dataclasses import dataclass, fields, replace, is_dataclass, asdict
from typing import Any, Union, Mapping, Tuple, Set, Optional, Dict, Type, TypeVar, Union

import torch
import torch.nn as nn
from torch import Tensor


from src.vae.cnn_encoder import CNNEncoder, CNNEncoderConfig
from src.vae.cnn_decoder import CNNDecoder, CNNDecoderConfig
from src.vae.gaussian_neck import GaussianNeckConfig, GaussianNeck
from src.vae.gaussian_loss import BetaVAECriterion, BetaVAECriterionConfig
from src.vae.types import GenLogits, Context, FeatureDict, ModelBatch
from src.vae.config import *


T = TypeVar("T")

def _to_hw(s: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return (int(s), int(s)) if isinstance(s, int) else (int(s[0]), int(s[1]))


class BetaVAE(nn.Module):
    """
    encoder -> neck -> decoder
    accepts sub-configs as objects or mappings; infers shared names.
    """
    NAME = "beta_vae"

    def __init__(self, vae_cfg: BetaVAEConfig, device: torch.device):
        super().__init__()

        self.config = vae_cfg
        self.device = device

        self.img_size = vae_cfg.img_size
        H, W = _to_hw(self.img_size)
        self.enc_name  = vae_cfg.enc_name
        self.neck_name = vae_cfg.neck_name

        self.latent_dim = self.config.neck.latent_dim

        self.enc  = CNNEncoder(
            self.config.encoder, out_name=self.enc_name, in_h=H, in_w=W
            )
        in_ch = self.enc.out_channels()[self.enc_name]
        in_h, in_w = self.enc.out_h, self.enc.out_w
        
        self.neck = GaussianNeck(
            self.config.neck, in_name=self.enc_name, 
            in_ch=in_ch, in_h=in_h, in_w=in_w, 
            out_name=self.neck_name
            )
        self.dec  = CNNDecoder(
            self.config.decoder, in_name=self.neck_name, 
            in_ch=in_ch, in_h=in_h, in_w=in_w,
            out_h=H, out_w=W
            )

        self.crit = BetaVAECriterion(self.config.criterion)
        
        self.to(self.device)
        
        perc_source = self.config.criterion.perc_source
        if perc_source == "encoder":
            self.crit.init_perc(encoder=self.enc)
        elif perc_source == "lpips":
            self.crit.init_perc(device=self.device)


    def forward(self, x: Tensor, ctx: Context | None = None) -> GenLogits:
        if ctx is None:
            ctx = {}
        feats = self.enc(x, ctx)
        feats = self.neck(feats, ctx)
        return self.dec(feats, ctx)


    def encode(self, x: Tensor, ctx: Context = {}) -> tuple[Tensor, Tensor]:
        _ = self.neck(self.enc(x, None), ctx)
        return ctx["mu"], ctx["logvar"]

    def decode_batch(batch: Any):
        return ModelBatch(x=batch, y=None, meta=None)

    @staticmethod
    def sample_z(mu: Tensor, logvar: Tensor) -> Tensor:
        logvar = logvar.clamp(min=-30, max=20)
        std = (0.5 * logvar).exp()
        return mu + torch.randn_like(std) * std


    def decode_from_z(self, z: Tensor) -> GenLogits:
        zero = torch.zeros_like(z)
        feats = {self.neck_name: self.neck.mlp_dec(torch.cat([z, zero], dim=1)).view(
            z.size(0), self.enc.out_channels()[self.enc_name], self.enc.out_h, self.enc.out_w
        )}
        return self.dec(feats, None)

    
    def decode_prob(self, z: Tensor) -> Tensor:
        """map Tanh output in [-1,1] to [0,1]"""
        imgs = self.decode_from_z(z).img
        return imgs.mul_(0.5).add_(0.5).clamp_(0, 1)


    def decode_uint8(self, z: Tensor) -> Tensor:
        return (self.decode_prob(z) * 255.0).round().to(torch.uint8)



    # --- NOT USED ---

    def training_step(self, batch: Any, batch_idx: int):
        mb: ModelBatch = self.decode_batch(batch)
        ctx: Context = {}
        logits = self.forward(mb.x, ctx)
        return self.crit(logits, mb.x, ctx)

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        mb: ModelBatch = self.decode_batch(batch)
        ctx: Context = {}
        with torch.no_grad():
            logits = self.forward(mb.x, ctx)
            return self.crit(logits, mb.x, ctx)

    def predict_step(self, batch: Any, batch_idx: int):
        mb: ModelBatch = self.decode_batch(batch)
        with torch.no_grad():
            return self.forward(mb.x, None)



if __name__ == "__main__":
    torch.manual_seed(0)

    img_size = 32
    model = BetaVAE(img_size=img_size)
    model.eval()

    B, C, H, W = 2, 3, img_size, img_size
    images = torch.rand(B, C, H, W) * 2 - 1  # [-1, 1]

    with torch.no_grad():
        logs = model.training_step((images, None, None), batch_idx=0)

    print("BetaVAE:", {k: float(v) for k, v in logs.items() if "loss" in k})

    with torch.no_grad():
        z = torch.randn(B, model.latent_dim)
        x_prob  = model.decode_prob(z)      # [B,3,H,W] in [0,1]
        x_uint8 = model.decode_uint8(z)     # [B,3,H,W] uint8
        print("sample decode:",
              dict(shape=list(x_prob.shape),
                   min=float(x_prob.min()), max=float(x_prob.max()),
                   dtype=str(x_uint8.dtype)))