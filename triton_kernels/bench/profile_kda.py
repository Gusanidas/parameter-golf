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

import torch
import torch.nn as nn

from triton_kernels.kda import kda_triton_autograd
from triton_kernels.bench._common import (
    RMSNorm, MLP, time_fwd_bwd, run_profiler_trace,
)


class KDALayer(nn.Module):
    """Project x → q, k, v, g, beta; run kda_triton_autograd; project back."""

    def __init__(self, dim: int, num_heads: int, k_dim: int, v_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        qkv_out = num_heads * (2 * k_dim + v_dim)  # q, k, v
        gate_out = num_heads * k_dim               # g (per-key decay, log-space)
        beta_out = num_heads                        # beta (scalar per head)
        self.in_proj = nn.Linear(dim, qkv_out + gate_out + beta_out, bias=False)
        self.out_proj = nn.Linear(num_heads * v_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, K, V = self.num_heads, self.k_dim, self.v_dim
        proj = self.in_proj(x)
        ofs = 0
        q = proj[..., ofs:ofs + H * K].reshape(B, T, H, K); ofs += H * K
        k = proj[..., ofs:ofs + H * K].reshape(B, T, H, K); ofs += H * K
        v = proj[..., ofs:ofs + H * V].reshape(B, T, H, V); ofs += H * V
        g_raw = proj[..., ofs:ofs + H * K].reshape(B, T, H, K); ofs += H * K
        beta_raw = proj[..., ofs:]
        # g < 0 everywhere (decay), small magnitude.
        g = -0.1 * torch.nn.functional.softplus(g_raw.float())
        beta = torch.sigmoid(beta_raw.float())
        y = kda_triton_autograd(q, k, v, g, beta)
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
