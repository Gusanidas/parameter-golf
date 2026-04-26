"""Time the MLP component in isolation under torch.compile.

Two-layer compiled stack of `_common.MLP` (D → 4D → D, squared-ReLU) plus residual.
Same shapes as profile_attention/profile_mamba2/profile_kda so the number is
directly subtractable from those totals.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from triton_kernels.bench._common import MLP, time_fwd_bwd


class MLPOnly(nn.Module):
    def __init__(self, dim: int, n_layers: int, mlp_mult: float = 4.0):
        super().__init__()
        self.layers = nn.ModuleList([MLP(dim, mlp_mult) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = x + l(x)
        return x


def main():
    BSZ      = int(os.environ.get("BSZ", 16))
    SEQLEN   = int(os.environ.get("SEQLEN", 1024))
    DIM      = int(os.environ.get("DIM", 512))
    N_LAYERS = int(os.environ.get("N_LAYERS", 2))
    N_ITERS  = int(os.environ.get("N_ITERS", 50))
    DTYPE    = torch.bfloat16
    NO_COMPILE = bool(int(os.environ.get("NO_COMPILE", 0)))

    print(f"MLP-only B={BSZ} T={SEQLEN} D={DIM} layers={N_LAYERS} compile={not NO_COMPILE}")

    torch.manual_seed(0)
    model = MLPOnly(DIM, N_LAYERS).to(device="cuda", dtype=DTYPE)
    if not NO_COMPILE:
        model = torch.compile(model, fullgraph=True)

    x = torch.randn(BSZ, SEQLEN, DIM, device="cuda", dtype=DTYPE, requires_grad=True)

    def step():
        # Scalar loss; .square().sum() so backward has work to do.
        return model(x).float().square().sum()

    print("\n[timing]")
    result = time_fwd_bwd(step, iters=N_ITERS, warmup=10)
    print(result.summary())


if __name__ == "__main__":
    main()
