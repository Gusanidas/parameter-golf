"""Profile KDA with a compiled minimodel (norm → KDA → residual → MLP → residual).

Builds a MiniModel = [embed → N×(KDABlock) → head], compiles it with
torch.compile(fullgraph=True), then times fwd+bwd with CUDA events and
dumps a torch.profiler trace.

Shapes default to the mamba2 training defaults (model_dim=512, H=6 analogue
head layout, T=1024). Override with env vars BSZ, SEQLEN, DIM, N_HEADS, K_DIM,
V_DIM, N_LAYERS, N_ITERS, NO_COMPILE=1.

Run:
    python -m triton_kernels.bench.profile_kda
"""
from __future__ import annotations

import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from triton_kernels.conv1d import causal_conv1d_autograd
from triton_kernels.kda import kda_triton_autograd
from triton_kernels.bench._common import (
    RMSNorm, MLP, time_fwd_bwd, run_profiler_trace,
)


class KDALayer(nn.Module):
    """FLA-shaped training KDA mixer around kda_triton_autograd."""

    def __init__(self, dim: int, num_heads: int, k_dim: int, v_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        H, K, V = num_heads, k_dim, v_dim
        self.q_proj = nn.Linear(dim, H * K, bias=False)
        self.k_proj = nn.Linear(dim, H * K, bias=False)
        self.v_proj = nn.Linear(dim, H * V, bias=False)
        self.f_proj = nn.Sequential(
            nn.Linear(dim, V, bias=False),
            nn.Linear(V, H * K, bias=False),
        )
        self.b_proj = nn.Linear(dim, H, bias=False)
        self.g_proj = nn.Sequential(
            nn.Linear(dim, V, bias=False),
            nn.Linear(V, H * V, bias=True),
        )
        self.o_norm_weight = nn.Parameter(torch.ones(V, dtype=torch.float32))
        self.out_proj = nn.Linear(num_heads * v_dim, dim, bias=False)
        self.A_log = nn.Parameter(torch.empty(H, dtype=torch.float32).uniform_(1.0, 16.0).log())
        dt = torch.exp(
            torch.rand(H * K, dtype=torch.float32) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.q_conv_w = nn.Parameter(torch.randn(H, K, 4) * 0.1)
        self.k_conv_w = nn.Parameter(torch.randn(H, K, 4) * 0.1)
        self.v_conv_w = nn.Parameter(torch.randn(H, V, 4) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, K, V = self.num_heads, self.k_dim, self.v_dim
        q = self.q_proj(x).reshape(B, T, H, K)
        k = self.k_proj(x).reshape(B, T, H, K)
        v = self.v_proj(x).reshape(B, T, H, V)
        q = F.silu(causal_conv1d_autograd(q, self.q_conv_w))
        k = F.silu(causal_conv1d_autograd(k, self.k_conv_w))
        v = F.silu(causal_conv1d_autograd(v, self.v_conv_w))
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        g_raw = self.f_proj(x).reshape(B, T, H, K)
        g = -torch.exp(self.A_log.float())[None, None, :, None] * F.softplus(
            g_raw.float() + self.dt_bias.reshape(H, K)
        )
        beta = torch.sigmoid(self.b_proj(x).float())
        y = kda_triton_autograd(q, k, v, g, beta)
        gate = self.g_proj(x).reshape(B, T, H, V)
        y = F.rms_norm(y, (V,), self.o_norm_weight.to(dtype=y.dtype)) * torch.sigmoid(gate)
        return self.out_proj(y.to(x.dtype).reshape(B, T, H * V))


class KDABlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, k_dim: int, v_dim: int,
                 mlp_mult: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm()
        self.kda = KDALayer(dim, num_heads, k_dim, v_dim)
        self.norm2 = RMSNorm()
        self.mlp = MLP(dim, mlp_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.kda(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniModel(nn.Module):
    def __init__(self, vocab: int, dim: int, num_heads: int,
                 k_dim: int, v_dim: int, n_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([
            KDABlock(dim, num_heads, k_dim, v_dim) for _ in range(n_layers)
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
    N_HEADS   = int(os.environ.get("N_HEADS", 8))
    K_DIM     = int(os.environ.get("K_DIM", 64))
    V_DIM     = int(os.environ.get("V_DIM", 64))
    N_LAYERS  = int(os.environ.get("N_LAYERS", 2))
    N_ITERS   = int(os.environ.get("N_ITERS", 50))
    VOCAB     = 8192
    DTYPE     = torch.bfloat16
    NO_COMPILE = bool(int(os.environ.get("NO_COMPILE", 0)))

    assert SEQLEN % 32 == 0, "KDA chunk size = 32, SEQLEN must be a multiple"
    print(f"KDA minimodel profile — B={BSZ} T={SEQLEN} D={DIM} H={N_HEADS} "
          f"K={K_DIM} V={V_DIM} layers={N_LAYERS} dtype={DTYPE}")

    torch.manual_seed(0)
    model = MiniModel(VOCAB, DIM, N_HEADS, K_DIM, V_DIM, N_LAYERS)
    model = model.to(device="cuda", dtype=DTYPE)
    # Keep RMSNorm eps / Linear weights in bf16; KDA kernel internally casts fp32.

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

    trace_path = os.environ.get("TRACE_PATH", "kda_trace.json")
    print("\n[profiler trace]")
    run_profiler_trace(step, trace_path, iters=5, warmup=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
