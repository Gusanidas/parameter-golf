"""Profile Mamba2 (SSD or fused) with a compiled minimodel.

Layout: embed → N × (RMSNorm → Mamba2 → residual → RMSNorm → MLP → residual) → head.
Compiled with torch.compile(fullgraph=True), timed with CUDA events, plus a
torch.profiler trace.

Env vars: BSZ, SEQLEN, DIM, N_HEADS, HEAD_DIM, STATE_DIM, N_LAYERS, N_ITERS,
VARIANT (ssd|fused), NO_COMPILE=1.

Run:
    VARIANT=fused python -m triton_kernels.bench.profile_mamba2
    VARIANT=ssd   python -m triton_kernels.bench.profile_mamba2
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from triton_kernels.mamba2_ssd import mamba2_ssd_triton_autograd
from triton_kernels.mamba2_fused import mamba2_fused_triton_autograd
from triton_kernels.bench._common import (
    RMSNorm, MLP, time_fwd_bwd, run_profiler_trace,
)


class Mamba2Layer(nn.Module):
    """Simplified Mamba2 block matching the in_proj/out_proj layout in
    train_gpt_pr1584_mamba.py. `variant` selects the kernel.
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int,
                 state_dim: int, variant: str = "fused"):
        super().__init__()
        assert variant in ("ssd", "fused")
        self.variant = variant
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.state_dim = state_dim
        H, P, N = num_heads, head_dim, state_dim
        self.in_proj = nn.Linear(dim, 2 * H * P + 2 * H * N + H, bias=False)
        self.out_proj = nn.Linear(H * P, dim, bias=False)
        self.A_log = nn.Parameter(torch.linspace(-3.0, -1.0, H).float())
        self.dt_bias = nn.Parameter(torch.zeros(H).float())
        self.norm = RMSNorm()
        if variant == "fused":
            self.conv_w_x = nn.Parameter(torch.randn(H, P, 4) * 0.1)
            self.conv_w_b = nn.Parameter(torch.randn(H, N, 4) * 0.1)
            self.conv_w_c = nn.Parameter(torch.randn(H, N, 4) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, P, N = self.num_heads, self.head_dim, self.state_dim
        proj = self.in_proj(x)
        x_ssm = proj[..., :H * P].reshape(B, T, H, P)
        z = proj[..., H * P:2 * H * P].reshape(B, T, H, P)
        Bm = proj[..., 2 * H * P:2 * H * P + H * N].reshape(B, T, H, N)
        Cm = proj[..., 2 * H * P + H * N:2 * H * P + 2 * H * N].reshape(B, T, H, N)
        dt_raw = proj[..., 2 * H * P + 2 * H * N:]
        dt = F.softplus(dt_raw.float() + self.dt_bias)
        if self.variant == "fused":
            y = mamba2_fused_triton_autograd(
                x_ssm, self.A_log, Bm, Cm, dt,
                self.conv_w_x, self.conv_w_b, self.conv_w_c,
            )
        else:
            y = mamba2_ssd_triton_autograd(x_ssm, self.A_log, Bm, Cm, dt)
        y = self.norm(y) * F.silu(z)
        return self.out_proj(y.to(x.dtype).reshape(B, T, H * P))


class Mamba2Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int,
                 state_dim: int, variant: str, mlp_mult: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm()
        self.mamba = Mamba2Layer(dim, num_heads, head_dim, state_dim, variant)
        self.norm2 = RMSNorm()
        self.mlp = MLP(dim, mlp_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mamba(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniModel(nn.Module):
    def __init__(self, vocab: int, dim: int, num_heads: int, head_dim: int,
                 state_dim: int, n_layers: int, variant: str):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([
            Mamba2Block(dim, num_heads, head_dim, state_dim, variant)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm()
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required"); return 2

    BSZ       = int(os.environ.get("BSZ", 4))
    SEQLEN    = int(os.environ.get("SEQLEN", 1024))
    DIM       = int(os.environ.get("DIM", 512))
    N_HEADS   = int(os.environ.get("N_HEADS", 6))
    HEAD_DIM  = int(os.environ.get("HEAD_DIM", 64))
    STATE_DIM = int(os.environ.get("STATE_DIM", 16))
    N_LAYERS  = int(os.environ.get("N_LAYERS", 2))
    N_ITERS   = int(os.environ.get("N_ITERS", 50))
    VARIANT   = os.environ.get("VARIANT", "fused")
    VOCAB     = 8192
    DTYPE     = torch.bfloat16
    NO_COMPILE = bool(int(os.environ.get("NO_COMPILE", 0)))

    assert SEQLEN % 64 == 0, "Mamba2 chunk size = 64, SEQLEN must be a multiple"
    print(f"Mamba2 minimodel profile — variant={VARIANT} B={BSZ} T={SEQLEN} "
          f"D={DIM} H={N_HEADS} P={HEAD_DIM} N={STATE_DIM} "
          f"layers={N_LAYERS} dtype={DTYPE}")

    torch.manual_seed(0)
    model = MiniModel(VOCAB, DIM, N_HEADS, HEAD_DIM, STATE_DIM, N_LAYERS, VARIANT)
    model = model.to(device="cuda", dtype=DTYPE)

    if not NO_COMPILE:
        model = torch.compile(model, fullgraph=True)

    tokens = torch.randint(0, VOCAB, (BSZ, SEQLEN), device="cuda")
    targets = torch.randint(0, VOCAB, (BSZ, SEQLEN), device="cuda")
    loss_fn = nn.CrossEntropyLoss()

    def step() -> torch.Tensor:
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        logits = model(tokens)
        return loss_fn(logits.float().flatten(0, 1), targets.flatten())

    print("\n[timing]")
    result = time_fwd_bwd(step, iters=N_ITERS, warmup=10)
    print(result.summary())

    trace_path = os.environ.get("TRACE_PATH", f"mamba2_{VARIANT}_trace.json")
    print("\n[profiler trace]")
    run_profiler_trace(step, trace_path, iters=5, warmup=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
