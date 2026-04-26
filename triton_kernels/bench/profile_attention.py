"""Profile full-causal and sliding-window attention with Flash Attention 2.

Uses flash_attn.flash_attn_func (FA2) with causal=True, qkv layout (B, T, H, P).
For windowed, passes window_size=(WINDOW-1, 0) for a sliding causal window.

FA2 is what is available in this container (flash_attn 2.7.x). The train script
imports FA3 via flash_attn_interface, but that module is not installed here; FA2
has the same signature for window_size so the numbers are directly readable.

Layout mirrors profile_mamba2.py / profile_kda.py:
    embed → N × (RMSNorm → Attention → residual → RMSNorm → MLP → residual) → head.

Env vars: BSZ, SEQLEN, DIM, N_HEADS, HEAD_DIM, MLP_MULT, N_LAYERS, N_ITERS,
WINDOW (0 = full causal, else sliding-window causal of that size),
NO_COMPILE=1.

Run:
    WINDOW=0   python -m triton_kernels.bench.profile_attention
    WINDOW=512 python -m triton_kernels.bench.profile_attention
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from flash_attn import flash_attn_func as flash_attn_2_func

from triton_kernels.bench._common import (
    RMSNorm, MLP, time_fwd_bwd, run_profiler_trace,
)


class AttentionLayer(nn.Module):
    """qkv = in_proj(x); FA2; out_proj(y). window=0 → full causal."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, window: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window = window
        H, P = num_heads, head_dim
        self.in_proj = nn.Linear(dim, 3 * H * P, bias=False)
        self.out_proj = nn.Linear(H * P, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, P = self.num_heads, self.head_dim
        qkv = self.in_proj(x).reshape(B, T, 3, H, P)
        q, k, v = qkv.unbind(dim=2)                          # (B, T, H, P)
        window = (self.window - 1, 0) if self.window > 0 else (-1, -1)
        y = flash_attn_2_func(q, k, v, causal=True, window_size=window)
        y = y.reshape(B, T, H * P)
        return self.out_proj(y)


class AttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, window: int,
                 mlp_mult: float):
        super().__init__()
        self.norm1 = RMSNorm()
        self.attn = AttentionLayer(dim, num_heads, head_dim, window)
        self.norm2 = RMSNorm()
        self.mlp = MLP(dim, mlp_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with record_function("mixer"):
            h = self.attn(self.norm1(x))
        x = x + h
        with record_function("mlp"):
            h = self.mlp(self.norm2(x))
        return x + h


class MiniModel(nn.Module):
    def __init__(self, vocab: int, dim: int, num_heads: int, head_dim: int,
                 n_layers: int, window: int, mlp_mult: float):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([
            AttnBlock(dim, num_heads, head_dim, window, mlp_mult)
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
    MLP_MULT  = float(os.environ.get("MLP_MULT", 4.0))
    N_LAYERS  = int(os.environ.get("N_LAYERS", 2))
    N_ITERS   = int(os.environ.get("N_ITERS", 50))
    WINDOW    = int(os.environ.get("WINDOW", 0))
    VOCAB     = 8192
    DTYPE     = torch.bfloat16
    NO_COMPILE = bool(int(os.environ.get("NO_COMPILE", 0)))

    tag = f"FA2 window={WINDOW}" if WINDOW > 0 else "FA2 full-causal"
    print(f"Attention minimodel profile — {tag} B={BSZ} T={SEQLEN} "
          f"D={DIM} H={N_HEADS} P={HEAD_DIM} mlp_mult={MLP_MULT} "
          f"layers={N_LAYERS} dtype={DTYPE}")

    torch.manual_seed(0)
    model = MiniModel(VOCAB, DIM, N_HEADS, HEAD_DIM, N_LAYERS, WINDOW, MLP_MULT)
    model = model.to(device="cuda", dtype=DTYPE)

    if not NO_COMPILE:
        model = torch.compile(model, dynamic=False, fullgraph=True)

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

    trace_path = os.environ.get("TRACE_PATH", f"attention_w{WINDOW}_trace.json")
    print("\n[profiler trace]")
    run_profiler_trace(step, trace_path, iters=5, warmup=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
