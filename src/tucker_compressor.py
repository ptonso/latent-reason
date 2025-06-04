import torch
import torch.nn as nn
from typing import Tuple
from src.logger import setup_logger


class TuckerCompressor:
    def __init__(self) -> None:
        self.logger = setup_logger("api.log")

    def compress_conv2d(
        self,
        conv: nn.Conv2d,
        rank: Tuple[int, int]
    ) -> nn.Sequential:
        """Compress a Conv2d layer using Tucker decomposition (mode-0 and mode-1)."""
        C_out, C_in = conv.out_channels, conv.in_channels
        kh, kw = conv.kernel_size
        stride, padding, dilation = conv.stride, conv.padding, conv.dilation
        r_out, r_in = rank

        self.logger.info(
            f"Compressing Conv2d: Cin={C_in}, Cout={C_out}, kh={kh}, kw={kw}, rank={rank}"
        )

        weight = conv.weight.data  # (C_out, C_in, kh, kw)
        weight_unfold_0 = weight.reshape(C_out, -1)
        weight_unfold_1 = weight.permute(1, 0, 2, 3).reshape(C_in, -1)

        U_0, _, _ = torch.linalg.svd(weight_unfold_0, full_matrices=False)
        U_1, _, _ = torch.linalg.svd(weight_unfold_1, full_matrices=False)

        U_0_tilde = U_0[:, :r_out]  # (C_out, r_out)
        U_1_tilde = U_1[:, :r_in]   # (C_in, r_in)

        # Compute core tensor: W ×₀ U₀ᵀ ×₁ U₁ᵀ
        core = torch.einsum('oc, cihw -> oihw', U_0_tilde.T, weight)
        core = torch.einsum('ci, oihw -> ochw', U_1_tilde.T, core)


        first_1x1 = nn.Conv2d(
            in_channels=C_in,
            out_channels=r_in,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        core_conv = nn.Conv2d(
            in_channels=r_in,
            out_channels=r_out,
            kernel_size=(kh, kw),
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False
        )

        last_1x1 = nn.Conv2d(
            in_channels=r_out,
            out_channels=C_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        with torch.no_grad():
            first_1x1.weight.copy_(U_1_tilde.T.unsqueeze(-1).unsqueeze(-1))
            core_conv.weight.copy_(core)
            if conv.bias is not None:
                last_1x1.bias.copy_(conv.bias)
            last_1x1.weight.copy_(U_0_tilde.unsqueeze(-1).unsqueeze(-1))

        return nn.Sequential(first_1x1, core_conv, last_1x1)
