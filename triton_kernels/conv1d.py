"""Causal depthwise Conv1d — Triton kernels (kernel_size=4).

Fused causal depthwise conv: y[t,d] = sum_{k=0}^{3} w[d,k] * x[t-k,d]
Each program handles one (batch, head) pair, loops over T.
"""
from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_fwd(
    X, W, Y,
    s_xb, s_xt, s_xh, s_xd,
    s_yb, s_yt, s_yh, s_yd,
    nheads, seqlen,
    D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // nheads
    h = pid % nheads

    d_offs = tl.arange(0, D)

    w_base = h * D * 4
    w0 = tl.load(W + w_base + d_offs * 4 + 0).to(tl.float32)
    w1 = tl.load(W + w_base + d_offs * 4 + 1).to(tl.float32)
    w2 = tl.load(W + w_base + d_offs * 4 + 2).to(tl.float32)
    w3 = tl.load(W + w_base + d_offs * 4 + 3).to(tl.float32)

    p1 = tl.zeros((D,), dtype=tl.float32)
    p2 = tl.zeros((D,), dtype=tl.float32)
    p3 = tl.zeros((D,), dtype=tl.float32)

    for t in range(seqlen):
        x_t = tl.load(X + b * s_xb + t * s_xt + h * s_xh + d_offs * s_xd).to(tl.float32)
        y_t = w0 * x_t + w1 * p1 + w2 * p2 + w3 * p3
        tl.store(Y + b * s_yb + t * s_yt + h * s_yh + d_offs * s_yd, y_t)
        p3 = p2
        p2 = p1
        p1 = x_t


@triton.jit
def _causal_conv1d_bwd(
    X, W, DY, DX, DW,
    s_xb, s_xt, s_xh, s_xd,
    s_dyb, s_dyt, s_dyh, s_dyd,
    s_dxb, s_dxt, s_dxh, s_dxd,
    nheads, seqlen,
    D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // nheads
    h = pid % nheads

    d_offs = tl.arange(0, D)

    w_base = h * D * 4
    w0 = tl.load(W + w_base + d_offs * 4 + 0).to(tl.float32)
    w1 = tl.load(W + w_base + d_offs * 4 + 1).to(tl.float32)
    w2 = tl.load(W + w_base + d_offs * 4 + 2).to(tl.float32)
    w3 = tl.load(W + w_base + d_offs * 4 + 3).to(tl.float32)

    dw0 = tl.zeros((D,), dtype=tl.float32)
    dw1 = tl.zeros((D,), dtype=tl.float32)
    dw2 = tl.zeros((D,), dtype=tl.float32)
    dw3 = tl.zeros((D,), dtype=tl.float32)

    p1 = tl.zeros((D,), dtype=tl.float32)
    p2 = tl.zeros((D,), dtype=tl.float32)
    p3 = tl.zeros((D,), dtype=tl.float32)

    f1 = tl.zeros((D,), dtype=tl.float32)
    f2 = tl.zeros((D,), dtype=tl.float32)
    f3 = tl.zeros((D,), dtype=tl.float32)

    # Forward pass: accumulate dW
    for t in range(seqlen):
        x_t = tl.load(X + b * s_xb + t * s_xt + h * s_xh + d_offs * s_xd).to(tl.float32)
        dy_t = tl.load(DY + b * s_dyb + t * s_dyt + h * s_dyh + d_offs * s_dyd).to(tl.float32)
        dw0 += dy_t * x_t
        dw1 += dy_t * p1
        dw2 += dy_t * p2
        dw3 += dy_t * p3
        p3 = p2
        p2 = p1
        p1 = x_t

    tl.atomic_add(DW + w_base + d_offs * 4 + 0, dw0)
    tl.atomic_add(DW + w_base + d_offs * 4 + 1, dw1)
    tl.atomic_add(DW + w_base + d_offs * 4 + 2, dw2)
    tl.atomic_add(DW + w_base + d_offs * 4 + 3, dw3)

    # Backward pass: compute dX
    for t_rev in range(seqlen):
        t = seqlen - 1 - t_rev
        dy_t = tl.load(DY + b * s_dyb + t * s_dyt + h * s_dyh + d_offs * s_dyd).to(tl.float32)
        dx_t = w0 * dy_t + w1 * f1 + w2 * f2 + w3 * f3
        tl.store(DX + b * s_dxb + t * s_dxt + h * s_dxh + d_offs * s_dxd, dx_t)
        f3 = f2
        f2 = f1
        f1 = dy_t


# ═══════════════════════════════════════════════════════════════════════════
# Python wrappers + torch.library custom ops
# ═══════════════════════════════════════════════════════════════════════════

def causal_conv1d_triton(x: Tensor, w: Tensor) -> Tensor:
    B, T, H, D = x.shape
    y = torch.empty_like(x)
    grid = (B * H,)
    _causal_conv1d_fwd[grid](x, w, y, *x.stride(), *y.stride(), H, T, D=D)
    return y


def causal_conv1d_triton_bwd(dy: Tensor, x: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    B, T, H, D = x.shape
    dx = torch.empty_like(x)
    dw = torch.zeros_like(w)
    grid = (B * H,)
    _causal_conv1d_bwd[grid](
        x, w, dy, dx, dw,
        *x.stride(), *dy.stride(), *dx.stride(),
        H, T, D=D,
    )
    return dx, dw


@torch.library.custom_op("mamba2t::conv1d_fwd", mutates_args=())
def _mamba2t_conv1d_fwd(x: Tensor, w: Tensor) -> Tensor:
    return causal_conv1d_triton(x, w)

@torch.library.register_fake("mamba2t::conv1d_fwd")
def _mamba2t_conv1d_fwd_fake(x, w):
    return x.new_empty(x.shape)


@torch.library.custom_op("mamba2t::conv1d_bwd", mutates_args=())
def _mamba2t_conv1d_bwd(dy: Tensor, x: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    return causal_conv1d_triton_bwd(dy, x, w)

@torch.library.register_fake("mamba2t::conv1d_bwd")
def _mamba2t_conv1d_bwd_fake(dy, x, w):
    return x.new_empty(x.shape), w.new_empty(w.shape)

def _conv1d_autograd_bwd(ctx, do):
    x, w = ctx.saved_tensors
    dx, dw = torch.ops.mamba2t.conv1d_bwd(do.contiguous(), x, w)
    return dx, dw

def _conv1d_autograd_setup(ctx, inputs, output):
    x, w = inputs
    ctx.save_for_backward(x, w)

torch.library.register_autograd(
    "mamba2t::conv1d_fwd", _conv1d_autograd_bwd, setup_context=_conv1d_autograd_setup,
)

def causal_conv1d_autograd(x: Tensor, w: Tensor) -> Tensor:
    """Causal conv1d with Triton fwd+bwd, fullgraph safe."""
    return torch.ops.mamba2t.conv1d_fwd(x, w)
