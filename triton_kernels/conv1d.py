"""Causal depthwise Conv1d — Triton kernels (kernel_size=4).

The original implementation launched one program per (batch, head) and looped
over the full sequence. This version tiles over time, which gives the GPU much
more parallel work for the short width-4 convolutions used by Mamba/KDA.
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
    BLOCK_T: tl.constexpr,
    D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)
    b = pid_bh // nheads
    h = pid_bh % nheads

    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d = tl.arange(0, D)
    mask_t = t < seqlen

    w_base = h * D * 4
    w0 = tl.load(W + w_base + d * 4 + 0).to(tl.float32)
    w1 = tl.load(W + w_base + d * 4 + 1).to(tl.float32)
    w2 = tl.load(W + w_base + d * 4 + 2).to(tl.float32)
    w3 = tl.load(W + w_base + d * 4 + 3).to(tl.float32)

    x_base = X + b * s_xb + h * s_xh
    x0 = tl.load(x_base + t[:, None] * s_xt + d[None, :] * s_xd, mask=mask_t[:, None], other=0.0).to(tl.float32)
    x1 = tl.load(
        x_base + (t[:, None] - 1) * s_xt + d[None, :] * s_xd,
        mask=((t[:, None] >= 1) & mask_t[:, None]), other=0.0,
    ).to(tl.float32)
    x2 = tl.load(
        x_base + (t[:, None] - 2) * s_xt + d[None, :] * s_xd,
        mask=((t[:, None] >= 2) & mask_t[:, None]), other=0.0,
    ).to(tl.float32)
    x3 = tl.load(
        x_base + (t[:, None] - 3) * s_xt + d[None, :] * s_xd,
        mask=((t[:, None] >= 3) & mask_t[:, None]), other=0.0,
    ).to(tl.float32)
    y = x0 * w0[None, :] + x1 * w1[None, :] + x2 * w2[None, :] + x3 * w3[None, :]
    tl.store(
        Y + b * s_yb + t[:, None] * s_yt + h * s_yh + d[None, :] * s_yd,
        y,
        mask=mask_t[:, None],
    )


@triton.jit
def _causal_conv1d_dx(
    W, DY, DX,
    s_dyb, s_dyt, s_dyh, s_dyd,
    s_dxb, s_dxt, s_dxh, s_dxd,
    nheads, seqlen,
    BLOCK_T: tl.constexpr,
    D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)
    b = pid_bh // nheads
    h = pid_bh % nheads

    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d = tl.arange(0, D)
    mask_t = t < seqlen

    w_base = h * D * 4
    w0 = tl.load(W + w_base + d * 4 + 0).to(tl.float32)
    w1 = tl.load(W + w_base + d * 4 + 1).to(tl.float32)
    w2 = tl.load(W + w_base + d * 4 + 2).to(tl.float32)
    w3 = tl.load(W + w_base + d * 4 + 3).to(tl.float32)

    dy_base = DY + b * s_dyb + h * s_dyh
    dy0 = tl.load(dy_base + t[:, None] * s_dyt + d[None, :] * s_dyd, mask=mask_t[:, None], other=0.0).to(tl.float32)
    dy1 = tl.load(
        dy_base + (t[:, None] + 1) * s_dyt + d[None, :] * s_dyd,
        mask=((t[:, None] + 1) < seqlen) & mask_t[:, None], other=0.0,
    ).to(tl.float32)
    dy2 = tl.load(
        dy_base + (t[:, None] + 2) * s_dyt + d[None, :] * s_dyd,
        mask=((t[:, None] + 2) < seqlen) & mask_t[:, None], other=0.0,
    ).to(tl.float32)
    dy3 = tl.load(
        dy_base + (t[:, None] + 3) * s_dyt + d[None, :] * s_dyd,
        mask=((t[:, None] + 3) < seqlen) & mask_t[:, None], other=0.0,
    ).to(tl.float32)
    dx = dy0 * w0[None, :] + dy1 * w1[None, :] + dy2 * w2[None, :] + dy3 * w3[None, :]
    tl.store(
        DX + b * s_dxb + t[:, None] * s_dxt + h * s_dxh + d[None, :] * s_dxd,
        dx,
        mask=mask_t[:, None],
    )


@triton.jit
def _causal_conv1d_dw_partial(
    X, DY, DW_PARTIAL,
    s_xb, s_xt, s_xh, s_xd,
    s_dyb, s_dyt, s_dyh, s_dyd,
    nheads, seqlen, num_tiles,
    BLOCK_T: tl.constexpr,
    D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)
    b = pid_bh // nheads
    h = pid_bh % nheads

    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d = tl.arange(0, D)
    mask_t = t < seqlen

    x_base = X + b * s_xb + h * s_xh
    dy = tl.load(
        DY + b * s_dyb + t[:, None] * s_dyt + h * s_dyh + d[None, :] * s_dyd,
        mask=mask_t[:, None], other=0.0,
    ).to(tl.float32)
    x0 = tl.load(x_base + t[:, None] * s_xt + d[None, :] * s_xd, mask=mask_t[:, None], other=0.0).to(tl.float32)
    x1 = tl.load(
        x_base + (t[:, None] - 1) * s_xt + d[None, :] * s_xd,
        mask=((t[:, None] >= 1) & mask_t[:, None]), other=0.0,
    ).to(tl.float32)
    x2 = tl.load(
        x_base + (t[:, None] - 2) * s_xt + d[None, :] * s_xd,
        mask=((t[:, None] >= 2) & mask_t[:, None]), other=0.0,
    ).to(tl.float32)
    x3 = tl.load(
        x_base + (t[:, None] - 3) * s_xt + d[None, :] * s_xd,
        mask=((t[:, None] >= 3) & mask_t[:, None]), other=0.0,
    ).to(tl.float32)

    base = ((pid_bh * num_tiles + pid_t) * D + d) * 4
    tl.store(DW_PARTIAL + base + 0, tl.sum(dy * x0, axis=0))
    tl.store(DW_PARTIAL + base + 1, tl.sum(dy * x1, axis=0))
    tl.store(DW_PARTIAL + base + 2, tl.sum(dy * x2, axis=0))
    tl.store(DW_PARTIAL + base + 3, tl.sum(dy * x3, axis=0))


@triton.jit
def _causal_conv1d_dw_reduce(
    DW_PARTIAL, DW,
    batch, nheads, num_tiles,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_d = tl.program_id(1)
    tap = tl.program_id(2)

    offs = tl.arange(0, BLOCK_N)
    mask = offs < (batch * num_tiles)
    b = offs // num_tiles
    tile = offs - b * num_tiles
    partial_row = (b * nheads + pid_h) * num_tiles + tile
    partial_idx = ((partial_row * D + pid_d) * 4) + tap
    acc = tl.sum(tl.load(DW_PARTIAL + partial_idx, mask=mask, other=0.0).to(tl.float32), axis=0)
    tl.store(DW + pid_h * D * 4 + pid_d * 4 + tap, acc)


# ═══════════════════════════════════════════════════════════════════════════
# Python wrappers + torch.library custom ops
# ═══════════════════════════════════════════════════════════════════════════

def _block_t(T: int) -> int:
    return 128 if T >= 128 else triton.next_power_of_2(T)


def causal_conv1d_triton(x: Tensor, w: Tensor) -> Tensor:
    B, T, H, D = x.shape
    y = torch.empty_like(x)
    block_t = _block_t(T)
    grid = (B * H, triton.cdiv(T, block_t))
    _causal_conv1d_fwd[grid](x, w, y, *x.stride(), *y.stride(), H, T, BLOCK_T=block_t, D=D)
    return y


def causal_conv1d_triton_bwd(dy: Tensor, x: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    B, T, H, D = x.shape
    dx = torch.empty_like(x)
    dw = torch.empty_like(w)
    block_t = _block_t(T)
    num_tiles = triton.cdiv(T, block_t)
    grid = (B * H, num_tiles)
    _causal_conv1d_dx[grid](
        w, dy, dx,
        *dy.stride(), *dx.stride(),
        H, T, BLOCK_T=block_t, D=D,
    )
    partial = torch.empty(B * H * num_tiles, D, 4, device=x.device, dtype=torch.float32)
    _causal_conv1d_dw_partial[grid](
        x, dy, partial,
        *x.stride(), *dy.stride(),
        H, T, num_tiles, BLOCK_T=block_t, D=D,
    )
    reduce_block = triton.next_power_of_2(B * num_tiles)
    _causal_conv1d_dw_reduce[(H, D, 4)](
        partial, dw,
        B, H, num_tiles,
        BLOCK_N=reduce_block, D=D,
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
    dx, dw = torch.ops.mamba2t.conv1d_bwd(do, x, w)
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
