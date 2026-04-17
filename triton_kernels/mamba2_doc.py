"""Doc-aware wrappers for Mamba2 SSD and fused kernels.

Stacks per-doc slices along the batch dim and runs a single kernel call,
so state (and fused conv history) reset between docs "for free" (each
batch row starts fresh). One kernel launch per fwd/bwd instead of N.

Packaged as torch.library custom ops so the Python reshuffling is opaque
to Dynamo (fullgraph=True safe). Backward recomputes the forward
(activation checkpointing) to obtain per-doc SSD states.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _doc_layout(cu_seqlens: Tensor, T: int) -> tuple[list[tuple[int, int, int]], int]:
    """Returns ([(start, end, L)], max_L_pad). Skips empty docs."""
    cu_list = cu_seqlens.detach().to(dtype=torch.int64, device="cpu").tolist()
    seen: list[int] = []
    for v in cu_list:
        iv = int(v)
        if not seen or seen[-1] != iv:
            seen.append(iv)
    if not seen or seen[0] != 0:
        seen.insert(0, 0)
    if seen[-1] != T:
        seen.append(T)
    CS = 64
    docs: list[tuple[int, int, int]] = []
    max_L_pad = 0
    for i in range(len(seen) - 1):
        s = seen[i]
        e = seen[i + 1]
        L = e - s
        if L <= 0:
            continue
        pad = (CS - L % CS) % CS
        L_pad = L + pad
        docs.append((s, e, L))
        if L_pad > max_L_pad:
            max_L_pad = L_pad
    return docs, max_L_pad


def _pack_docs(
    x: Tensor, B: Tensor, C: Tensor, dt: Tensor,
    docs: list[tuple[int, int, int]], max_L_pad: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pack [1, T, ...] → [num_docs, max_L_pad, ...] with zero padding."""
    D = len(docs)
    H = x.shape[2]
    P = x.shape[3]
    N = B.shape[3]
    xb = x.new_zeros(D, max_L_pad, H, P)
    Bb = B.new_zeros(D, max_L_pad, H, N)
    Cb = C.new_zeros(D, max_L_pad, H, N)
    dtb = dt.new_zeros(D, max_L_pad, H)
    for i, (s, e, L) in enumerate(docs):
        xb[i, :L] = x[0, s:e]
        Bb[i, :L] = B[0, s:e]
        Cb[i, :L] = C[0, s:e]
        dtb[i, :L] = dt[0, s:e]
    return xb, Bb, Cb, dtb


def _pack_dy(
    dy: Tensor, docs: list[tuple[int, int, int]], max_L_pad: int,
) -> Tensor:
    D = len(docs)
    H = dy.shape[2]
    P = dy.shape[3]
    dyb = dy.new_zeros(D, max_L_pad, H, P)
    for i, (s, e, L) in enumerate(docs):
        dyb[i, :L] = dy[0, s:e]
    return dyb


def _unpack(
    y_dst: Tensor, yb: Tensor, docs: list[tuple[int, int, int]],
) -> None:
    for i, (s, e, L) in enumerate(docs):
        y_dst[0, s:e] = yb[i, :L]


# ═══════════════════════════════════════════════════════════════════════════
# Doc-aware SSD (non-fused)
# ═══════════════════════════════════════════════════════════════════════════


@torch.library.custom_op("mamba2t::chunk_fwd_doc", mutates_args=())
def _fwd_doc(
    x: Tensor, A_log: Tensor, B: Tensor, C: Tensor, dt: Tensor,
    cu_seqlens: Tensor,
) -> Tensor:
    Bsz, T, H, P = x.shape
    assert Bsz == 1, "doc-aware expects B=1"
    docs, max_L_pad = _doc_layout(cu_seqlens, T)
    y = x.new_empty(x.shape)
    if not docs:
        return y
    xb, Bb, Cb, dtb = _pack_docs(x, B, C, dt, docs, max_L_pad)
    yb, _ = torch.ops.mamba2t.chunk_fwd(xb, A_log, Bb, Cb, dtb)
    _unpack(y, yb, docs)
    return y


@torch.library.register_fake("mamba2t::chunk_fwd_doc")
def _fwd_doc_fake(x, A_log, B, C, dt, cu_seqlens):
    return x.new_empty(x.shape)


@torch.library.custom_op("mamba2t::chunk_bwd_doc", mutates_args=())
def _bwd_doc(
    dy: Tensor, x: Tensor, A_log: Tensor, B: Tensor, C: Tensor, dt: Tensor,
    cu_seqlens: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    Bsz, T, H, P = x.shape
    dx = x.new_empty(x.shape)
    dB = B.new_empty(B.shape)
    dC = C.new_empty(C.shape)
    ddt = dt.new_empty(dt.shape)
    dA_log = torch.zeros_like(A_log)
    docs, max_L_pad = _doc_layout(cu_seqlens, T)
    if not docs:
        return dx, dA_log, dB, dC, ddt
    xb, Bb, Cb, dtb = _pack_docs(x, B, C, dt, docs, max_L_pad)
    dyb = _pack_dy(dy, docs, max_L_pad)
    # Recompute fwd to obtain SSD states (activation checkpointing).
    _yb, statesb = torch.ops.mamba2t.chunk_fwd(xb, A_log, Bb, Cb, dtb)
    dxb, dA_log_b, dBb, dCb, ddtb = torch.ops.mamba2t.chunk_bwd(
        dyb, xb, A_log, Bb, Cb, dtb, statesb,
    )
    _unpack(dx, dxb, docs)
    _unpack(dB, dBb, docs)
    _unpack(dC, dCb, docs)
    for i, (s, e, L) in enumerate(docs):
        ddt[0, s:e] = ddtb[i, :L]
    dA_log = dA_log + dA_log_b
    return dx, dA_log, dB, dC, ddt


@torch.library.register_fake("mamba2t::chunk_bwd_doc")
def _bwd_doc_fake(dy, x, A_log, B, C, dt, cu_seqlens):
    return (
        x.new_empty(x.shape),
        A_log.new_empty(A_log.shape),
        B.new_empty(B.shape),
        C.new_empty(C.shape),
        dt.new_empty(dt.shape),
    )


def _fwd_doc_setup(ctx, inputs, output):
    x, A_log, B, C, dt, cu_seqlens = inputs
    ctx.save_for_backward(x, A_log, B, C, dt, cu_seqlens)


def _fwd_doc_backward(ctx, do):
    x, A_log, B, C, dt, cu_seqlens = ctx.saved_tensors
    dx, dA_log, dB, dC, ddt = torch.ops.mamba2t.chunk_bwd_doc(
        do.contiguous(), x, A_log, B, C, dt, cu_seqlens,
    )
    return dx, dA_log, dB, dC, ddt, None


torch.library.register_autograd(
    "mamba2t::chunk_fwd_doc", _fwd_doc_backward, setup_context=_fwd_doc_setup,
)


def mamba2_ssd_doc_triton_autograd(x, A_log, B, C, dt, cu_seqlens):
    """Doc-aware chunked SSD. State resets at each cu_seqlens boundary."""
    return torch.ops.mamba2t.chunk_fwd_doc(x, A_log, B, C, dt, cu_seqlens)


# ═══════════════════════════════════════════════════════════════════════════
# Doc-aware Fused (conv1d + SiLU + SSD)
# ═══════════════════════════════════════════════════════════════════════════


@torch.library.custom_op("mamba2t::fused_fwd_doc", mutates_args=())
def _fused_fwd_doc(
    x: Tensor, A_log: Tensor, B: Tensor, C: Tensor, dt: Tensor,
    conv_w_x: Tensor, conv_w_b: Tensor, conv_w_c: Tensor,
    cu_seqlens: Tensor,
) -> Tensor:
    Bsz, T, H, P = x.shape
    assert Bsz == 1, "doc-aware expects B=1"
    docs, max_L_pad = _doc_layout(cu_seqlens, T)
    y = x.new_empty(x.shape)
    if not docs:
        return y
    xb, Bb, Cb, dtb = _pack_docs(x, B, C, dt, docs, max_L_pad)
    yb, _ = torch.ops.mamba2t.fused_fwd(
        xb, A_log, Bb, Cb, dtb, conv_w_x, conv_w_b, conv_w_c,
    )
    _unpack(y, yb, docs)
    return y


@torch.library.register_fake("mamba2t::fused_fwd_doc")
def _fused_fwd_doc_fake(
    x, A_log, B, C, dt, conv_w_x, conv_w_b, conv_w_c, cu_seqlens,
):
    return x.new_empty(x.shape)


@torch.library.custom_op("mamba2t::fused_bwd_doc", mutates_args=())
def _fused_bwd_doc(
    dy: Tensor, x: Tensor, A_log: Tensor, B: Tensor, C: Tensor, dt: Tensor,
    conv_w_x: Tensor, conv_w_b: Tensor, conv_w_c: Tensor,
    cu_seqlens: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    Bsz, T, H, P = x.shape
    dx = x.new_empty(x.shape)
    dB = B.new_empty(B.shape)
    dC = C.new_empty(C.shape)
    ddt = dt.new_empty(dt.shape)
    dA_log = torch.zeros_like(A_log)
    d_cwx = torch.zeros_like(conv_w_x)
    d_cwb = torch.zeros_like(conv_w_b)
    d_cwc = torch.zeros_like(conv_w_c)
    docs, max_L_pad = _doc_layout(cu_seqlens, T)
    if not docs:
        return dx, dA_log, dB, dC, ddt, d_cwx, d_cwb, d_cwc
    xb, Bb, Cb, dtb = _pack_docs(x, B, C, dt, docs, max_L_pad)
    dyb = _pack_dy(dy, docs, max_L_pad)
    _yb, statesb = torch.ops.mamba2t.fused_fwd(
        xb, A_log, Bb, Cb, dtb, conv_w_x, conv_w_b, conv_w_c,
    )
    (dxb, dA_log_b, dBb, dCb, ddtb,
     d_cwx_b, d_cwb_b, d_cwc_b) = torch.ops.mamba2t.fused_bwd(
        dyb, xb, A_log, Bb, Cb, dtb, statesb,
        conv_w_x, conv_w_b, conv_w_c,
    )
    _unpack(dx, dxb, docs)
    _unpack(dB, dBb, docs)
    _unpack(dC, dCb, docs)
    for i, (s, e, L) in enumerate(docs):
        ddt[0, s:e] = ddtb[i, :L]
    dA_log = dA_log + dA_log_b
    d_cwx = d_cwx + d_cwx_b
    d_cwb = d_cwb + d_cwb_b
    d_cwc = d_cwc + d_cwc_b
    return dx, dA_log, dB, dC, ddt, d_cwx, d_cwb, d_cwc


@torch.library.register_fake("mamba2t::fused_bwd_doc")
def _fused_bwd_doc_fake(
    dy, x, A_log, B, C, dt, conv_w_x, conv_w_b, conv_w_c, cu_seqlens,
):
    return (
        x.new_empty(x.shape),
        A_log.new_empty(A_log.shape),
        B.new_empty(B.shape),
        C.new_empty(C.shape),
        dt.new_empty(dt.shape),
        conv_w_x.new_empty(conv_w_x.shape),
        conv_w_b.new_empty(conv_w_b.shape),
        conv_w_c.new_empty(conv_w_c.shape),
    )


def _fused_doc_setup(ctx, inputs, output):
    x, A_log, B, C, dt, conv_w_x, conv_w_b, conv_w_c, cu_seqlens = inputs
    ctx.save_for_backward(
        x, A_log, B, C, dt, conv_w_x, conv_w_b, conv_w_c, cu_seqlens,
    )


def _fused_doc_backward(ctx, do):
    x, A_log, B, C, dt, conv_w_x, conv_w_b, conv_w_c, cu_seqlens = ctx.saved_tensors
    (dx, dA_log, dB, dC, ddt,
     d_cwx, d_cwb, d_cwc) = torch.ops.mamba2t.fused_bwd_doc(
        do.contiguous(), x, A_log, B, C, dt,
        conv_w_x, conv_w_b, conv_w_c, cu_seqlens,
    )
    return dx, dA_log, dB, dC, ddt, d_cwx, d_cwb, d_cwc, None


torch.library.register_autograd(
    "mamba2t::fused_fwd_doc", _fused_doc_backward, setup_context=_fused_doc_setup,
)


def mamba2_fused_doc_triton_autograd(
    x, A_log, B, C, dt, conv_w_x, conv_w_b, conv_w_c, cu_seqlens,
):
    """Doc-aware fused conv1d + SiLU + SSD. Resets at each boundary."""
    return torch.ops.mamba2t.fused_fwd_doc(
        x, A_log, B, C, dt, conv_w_x, conv_w_b, conv_w_c, cu_seqlens,
    )
