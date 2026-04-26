"""Kimi Delta Attention via FLA kernels with a torch.library custom op.

The implementation intentionally delegates the chunked forward/backward kernels
to ``fla.ops.kda`` and keeps this module as a fullgraph-safe adapter.
"""
from __future__ import annotations

import torch
from torch import Tensor


_FLA_FUNCS: tuple | None = None


def _require_fla():
    # Cache the function pointers on first call so the per-step fwd/bwd path
    # doesn't re-resolve `fla.ops.kda.chunk_{fwd,bwd}` through sys.modules
    # every invocation.
    global _FLA_FUNCS
    if _FLA_FUNCS is None:
        try:
            from fla.ops.kda.chunk_bwd import chunk_kda_bwd
            from fla.ops.kda.chunk_fwd import chunk_kda_fwd
        except ImportError as exc:
            raise ImportError(
                "kda_triton_autograd requires fla-core. Install dependencies from "
                "requirements.txt, or `pip install fla-core`."
            ) from exc
        _FLA_FUNCS = (chunk_kda_fwd, chunk_kda_bwd)
    return _FLA_FUNCS


@torch.library.custom_op("kdat::fla_fwd", mutates_args=())
def _kdat_fla_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run FLA's KDA forward and return the tensors needed by backward.

    Public Gus KDA is the simple ``HV == H`` path:
    q/k/v/g are [B, T, H, K/V], beta is [B, T, H], and g is already in
    log-space. FLA supports more modes; this adapter keeps the old Gus API.
    """
    chunk_kda_fwd, _ = _require_fla()
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {q.shape} and {k.shape}")
    if g.shape != q.shape:
        raise ValueError(f"g must have shape {q.shape}, got {g.shape}")
    if beta.shape != q.shape[:-1]:
        raise ValueError(f"beta must have shape {q.shape[:-1]}, got {beta.shape}")
    if v.shape[:3] != q.shape[:3]:
        raise ValueError(
            "Gus KDA currently supports HV == H; expected v[:3] to match q[:3], "
            f"got v={v.shape}, q={q.shape}"
        )

    y, _final_state, g_cumsum, Aqk, Akk, _w, _u, _qg, _kg, _v_new, _h, _h0 = chunk_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        disable_recompute=False,
    )
    return y, g_cumsum, Aqk, Akk, torch.empty(0, device=q.device, dtype=torch.float32)


@torch.library.register_fake("kdat::fla_fwd")
def _kdat_fla_fwd_fake(q, k, v, g, beta, scale: float):
    empty = torch.empty(0, device=q.device, dtype=torch.float32)
    return (
        v.new_empty(v.shape),
        g.new_empty(g.shape),
        q.new_empty(q.shape[0], q.shape[1], q.shape[2], 64),
        q.new_empty(q.shape[0], q.shape[1], q.shape[2], 64),
        empty,
    )


@torch.library.custom_op("kdat::fla_bwd", mutates_args=())
def _kdat_fla_bwd(
    dy: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g_cumsum: Tensor,
    g_input: Tensor,
    beta: Tensor,
    Aqk: Tensor,
    Akk: Tensor,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    _, chunk_kda_bwd = _require_fla()
    dq, dk, dv, dbeta, dg, _dh0, _dA, _dbias = chunk_kda_bwd(
        q=q,
        k=k,
        v=v,
        g=g_cumsum,
        beta=beta,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        initial_state=None,
        do=dy.contiguous(),
        dht=None,
        g_org=g_input,
        disable_recompute=False,
    )
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dg.to(g_input.dtype), dbeta.to(beta.dtype)


@torch.library.register_fake("kdat::fla_bwd")
def _kdat_fla_bwd_fake(dy, q, k, v, g_cumsum, g_input, beta, Aqk, Akk, scale: float):
    return (
        q.new_empty(q.shape),
        k.new_empty(k.shape),
        v.new_empty(v.shape),
        g_input.new_empty(g_input.shape),
        beta.new_empty(beta.shape),
    )


def _kdat_autograd_bwd(ctx, do, _dg_cumsum, _dAqk, _dAkk, _d_unused):
    q, k, v, g_input, beta, g_cumsum, Aqk, Akk = ctx.saved_tensors
    dq, dk, dv, dg, dbeta = torch.ops.kdat.fla_bwd(
        do.contiguous(), q, k, v, g_cumsum, g_input, beta, Aqk, Akk, ctx.scale,
    )
    return dq, dk, dv, dg, dbeta, None


def _kdat_autograd_setup(ctx, inputs, output):
    q, k, v, g, beta, scale = inputs
    _y, g_cumsum, Aqk, Akk, _unused = output
    ctx.save_for_backward(q, k, v, g, beta, g_cumsum, Aqk, Akk)
    ctx.scale = scale


torch.library.register_autograd(
    "kdat::fla_fwd", _kdat_autograd_bwd, setup_context=_kdat_autograd_setup,
)


def kda_triton_autograd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    scale: float | None = None,
) -> Tensor:
    """Fullgraph-safe KDA using FLA's performant chunk kernels.

    Args:
        q, k, g: [B, T, H, K]
        v: [B, T, H, V]
        beta: [B, T, H], already in post-sigmoid space
        scale: defaults to K**-0.5
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    y, _g_cumsum, _Aqk, _Akk, _unused = torch.ops.kdat.fla_fwd(q, k, v, g, beta, float(scale))
    return y
