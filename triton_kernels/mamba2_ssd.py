"""Mamba2 SSD — Chunked forward + backward Triton kernels.

Chunked SSD — O(T*C) instead of O(T²). Each CUDA block handles one (batch, head)
pair, loops over chunks of size C=64, maintaining state in registers.
Four small matmuls per chunk (all ≤64×64).
"""
from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════
# Forward Kernel
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _mamba2_chunk_fwd(
    X, B, C, DT, A_LOG, Y, STATES,
    s_xb, s_xt, s_xh, s_xd,
    s_bb, s_bt, s_bh, s_bn,
    s_cb, s_ct, s_ch, s_cn,
    s_db, s_dt_, s_dh,
    s_yb, s_yt, s_yh, s_yd,
    s_sb, s_sh, s_sc, s_sn, s_sd,
    nheads, num_chunks,
    CHUNK: tl.constexpr,
    HDIM: tl.constexpr,
    SDIM: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // nheads
    h = pid % nheads

    A_h = -tl.exp(tl.load(A_LOG + h).to(tl.float32))
    state = tl.zeros((SDIM, HDIM), dtype=tl.float32)

    c_offs = tl.arange(0, CHUNK)
    d_offs = tl.arange(0, HDIM)
    n_offs = tl.arange(0, SDIM)

    for ci in range(num_chunks):
        t0 = ci * CHUNK

        st_ptrs = STATES + b * s_sb + h * s_sh + ci * s_sc + n_offs[:, None] * s_sn + d_offs[None, :] * s_sd
        tl.store(st_ptrs, state)

        x_c = tl.load(X + b * s_xb + (t0 + c_offs[:, None]) * s_xt + h * s_xh + d_offs[None, :] * s_xd).to(tl.float32)
        B_c = tl.load(B + b * s_bb + (t0 + c_offs[:, None]) * s_bt + h * s_bh + n_offs[None, :] * s_bn).to(tl.float32)
        C_c = tl.load(C + b * s_cb + (t0 + c_offs[:, None]) * s_ct + h * s_ch + n_offs[None, :] * s_cn).to(tl.float32)
        dt_c = tl.load(DT + b * s_db + (t0 + c_offs) * s_dt_ + h * s_dh).to(tl.float32)

        log_a = tl.maximum(A_h * dt_c, -20.0)
        cum = tl.cumsum(log_a, axis=0)

        Cs = tl.dot(C_c, state, input_precision="tf32")
        y_state = Cs * tl.exp(cum)[:, None]

        qk = tl.dot(C_c, tl.trans(B_c), input_precision="tf32")
        decay_diff = cum[:, None] - cum[None, :]
        causal = c_offs[:, None] >= c_offs[None, :]
        decay_diff = tl.where(causal, decay_diff, -float('inf'))
        attn = qk * tl.exp(decay_diff)
        y_intra = tl.dot(attn, x_c, input_precision="tf32")

        y_chunk = y_state + y_intra
        y_ptrs = Y + b * s_yb + (t0 + c_offs[:, None]) * s_yt + h * s_yh + d_offs[None, :] * s_yd
        tl.store(y_ptrs, y_chunk)

        total_log = tl.sum(log_a, axis=0)
        decay_to_end = tl.exp(total_log - cum)
        state = state * tl.exp(total_log) + tl.dot(
            tl.trans(B_c), x_c * decay_to_end[:, None], input_precision="tf32",
        )

    st_ptrs = STATES + b * s_sb + h * s_sh + num_chunks * s_sc + n_offs[:, None] * s_sn + d_offs[None, :] * s_sd
    tl.store(st_ptrs, state)


# ═══════════════════════════════════════════════════════════════════════════
# Backward Kernel
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _mamba2_chunk_bwd(
    X, B, C, DT, A_LOG, STATES, DY,
    DX, DB, DC, DDT,
    s_xb, s_xt, s_xh, s_xd,
    s_bb, s_bt, s_bh, s_bn,
    s_cb, s_ct, s_ch, s_cn,
    s_db, s_dt_, s_dh,
    s_sb, s_sh, s_sc, s_sn, s_sd,
    s_dyb, s_dyt, s_dyh, s_dyd,
    s_dxb, s_dxt, s_dxh, s_dxd,
    s_dbb, s_dbt, s_dbh, s_dbn,
    s_dcb, s_dct, s_dch, s_dcn,
    s_ddb, s_ddt_, s_ddh,
    nheads, num_chunks,
    CHUNK: tl.constexpr,
    HDIM: tl.constexpr,
    SDIM: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // nheads
    h = pid % nheads

    A_h = -tl.exp(tl.load(A_LOG + h).to(tl.float32))

    c_offs = tl.arange(0, CHUNK)
    d_offs = tl.arange(0, HDIM)
    n_offs = tl.arange(0, SDIM)

    d_state = tl.zeros((SDIM, HDIM), dtype=tl.float32)

    for ci_rev in range(num_chunks):
        ci = num_chunks - 1 - ci_rev
        t0 = ci * CHUNK

        st_ptrs = STATES + b * s_sb + h * s_sh + ci * s_sc + n_offs[:, None] * s_sn + d_offs[None, :] * s_sd
        state = tl.load(st_ptrs).to(tl.float32)

        x_c = tl.load(X + b * s_xb + (t0 + c_offs[:, None]) * s_xt + h * s_xh + d_offs[None, :] * s_xd).to(tl.float32)
        B_c = tl.load(B + b * s_bb + (t0 + c_offs[:, None]) * s_bt + h * s_bh + n_offs[None, :] * s_bn).to(tl.float32)
        C_c = tl.load(C + b * s_cb + (t0 + c_offs[:, None]) * s_ct + h * s_ch + n_offs[None, :] * s_cn).to(tl.float32)
        dt_c = tl.load(DT + b * s_db + (t0 + c_offs) * s_dt_ + h * s_dh).to(tl.float32)

        dy = tl.load(DY + b * s_dyb + (t0 + c_offs[:, None]) * s_dyt + h * s_dyh + d_offs[None, :] * s_dyd).to(tl.float32)

        log_a = tl.maximum(A_h * dt_c, -20.0)
        cum = tl.cumsum(log_a, axis=0)
        total_log = tl.sum(log_a, axis=0)
        exp_cum = tl.exp(cum)
        decay_to_end = tl.exp(total_log - cum)

        qk = tl.dot(C_c, tl.trans(B_c), input_precision="tf32")
        decay_diff = cum[:, None] - cum[None, :]
        causal = c_offs[:, None] >= c_offs[None, :]
        decay_diff = tl.where(causal, decay_diff, -float('inf'))
        decay = tl.exp(decay_diff)
        attn = qk * decay

        Cs = tl.dot(C_c, state, input_precision="tf32")

        dCs = dy * exp_cum[:, None]
        dC_chunk = tl.dot(dCs, tl.trans(state), input_precision="tf32")
        d_state_read = tl.dot(tl.trans(C_c), dCs, input_precision="tf32")

        d_cum = tl.sum(dy * Cs * exp_cum[:, None], axis=1)

        dx_chunk = tl.dot(tl.trans(attn), dy, input_precision="tf32")
        dattn = tl.dot(dy, tl.trans(x_c), input_precision="tf32")

        dqk = dattn * decay

        d_decay = dattn * qk
        d_cum = d_cum + tl.sum(d_decay * decay, axis=1)
        d_cum = d_cum - tl.sum(d_decay * decay, axis=0)

        dC_chunk = dC_chunk + tl.dot(dqk, B_c, input_precision="tf32")
        dB_chunk = tl.dot(tl.trans(dqk), C_c, input_precision="tf32")

        weighted_x = x_c * decay_to_end[:, None]

        d_wx = tl.dot(B_c, d_state, input_precision="tf32")
        dx_chunk = dx_chunk + d_wx * decay_to_end[:, None]

        dB_state = tl.dot(weighted_x, tl.trans(d_state), input_precision="tf32")
        dB_chunk = dB_chunk + dB_state

        d_dte = tl.sum(d_wx * x_c, axis=1)

        d_total_log_dte = tl.sum(d_dte * decay_to_end, axis=0)
        d_cum = d_cum - d_dte * decay_to_end

        d_total_log_sd = tl.sum(d_state * state) * tl.exp(total_log)

        d_state = d_state * tl.exp(total_log) + d_state_read

        d_total_log = d_total_log_dte + d_total_log_sd
        total_d_cum = tl.sum(d_cum, axis=0)
        fwd_cumsum = tl.cumsum(d_cum, axis=0)
        d_log_a = total_d_cum - fwd_cumsum + d_cum + d_total_log

        not_clamped = (A_h * dt_c > -20.0).to(tl.float32)
        d_dt = d_log_a * A_h * not_clamped

        dx_ptrs = DX + b * s_dxb + (t0 + c_offs[:, None]) * s_dxt + h * s_dxh + d_offs[None, :] * s_dxd
        tl.store(dx_ptrs, dx_chunk)

        db_ptrs = DB + b * s_dbb + (t0 + c_offs[:, None]) * s_dbt + h * s_dbh + n_offs[None, :] * s_dbn
        tl.store(db_ptrs, dB_chunk)

        dc_ptrs = DC + b * s_dcb + (t0 + c_offs[:, None]) * s_dct + h * s_dch + n_offs[None, :] * s_dcn
        tl.store(dc_ptrs, dC_chunk)

        ddt_ptrs = DDT + b * s_ddb + (t0 + c_offs) * s_ddt_ + h * s_ddh
        tl.store(ddt_ptrs, d_dt)


# ═══════════════════════════════════════════════════════════════════════════
# Python wrappers + torch.library custom ops
# ═══════════════════════════════════════════════════════════════════════════

@torch.library.custom_op("mamba2t::chunk_fwd", mutates_args=())
def _mamba2t_chunk_fwd(
    x: Tensor, A_log: Tensor, B: Tensor, C: Tensor, dt: Tensor,
) -> tuple[Tensor, Tensor]:
    Bsz, T, H, P = x.shape
    N = B.shape[-1]
    CS = 64
    if T % CS != 0:
        raise ValueError("mamba2 SSD kernel requires sequence length divisible by 64")
    NC = T // CS
    y = x.new_empty(x.shape)
    states = torch.empty(Bsz, H, NC + 1, N, P, device=x.device, dtype=torch.float32)
    grid = (Bsz * H,)
    _mamba2_chunk_fwd[grid](
        x, B, C, dt, A_log, y, states,
        *x.stride(), *B.stride(), *C.stride(), *dt.stride(), *y.stride(), *states.stride(),
        H, NC, CHUNK=CS, HDIM=P, SDIM=N,
    )
    return y, states


@torch.library.register_fake("mamba2t::chunk_fwd")
def _mamba2t_chunk_fwd_fake(x, A_log, B, C, dt):
    Bsz, T, H, P = x.shape
    N = B.shape[-1]
    NC = T // 64
    y = x.new_empty(Bsz, T, H, P)
    states = torch.empty(Bsz, H, NC + 1, N, P, device=x.device, dtype=torch.float32)
    return y, states


@torch.library.custom_op("mamba2t::chunk_bwd", mutates_args=())
def _mamba2t_chunk_bwd(
    dy: Tensor, x: Tensor, A_log: Tensor, B: Tensor, C: Tensor,
    dt: Tensor, states: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    Bsz, T, H, P = x.shape
    N = B.shape[-1]
    CS = 64
    if T % CS != 0:
        raise ValueError("mamba2 SSD backward requires sequence length divisible by 64")
    NC = T // CS
    dx = x.new_empty(x.shape)
    dB = B.new_empty(B.shape)
    dC = C.new_empty(C.shape)
    ddt = dt.new_empty(dt.shape)
    grid = (Bsz * H,)
    _mamba2_chunk_bwd[grid](
        x, B, C, dt, A_log, states, dy,
        dx, dB, dC, ddt,
        *x.stride(), *B.stride(), *C.stride(), *dt.stride(),
        *states.stride(), *dy.stride(),
        *dx.stride(), *dB.stride(), *dC.stride(), *ddt.stride(),
        H, NC, CHUNK=CS, HDIM=P, SDIM=N,
    )
    dA_log = (ddt * dt).sum(dim=(0, 1))  # [H]
    return dx, dA_log, dB, dC, ddt


@torch.library.register_fake("mamba2t::chunk_bwd")
def _mamba2t_chunk_bwd_fake(dy, x, A_log, B, C, dt, states):
    return (
        x.new_empty(x.shape),
        A_log.new_empty(A_log.shape),
        B.new_empty(B.shape),
        C.new_empty(C.shape),
        dt.new_empty(dt.shape),
    )


def _mamba2t_autograd_bwd(ctx, do, d_states):
    x, A_log, B, C, dt, states = ctx.saved_tensors
    dx, dA_log, dB, dC, ddt = torch.ops.mamba2t.chunk_bwd(
        do, x, A_log, B, C, dt, states,
    )
    return dx, dA_log, dB, dC, ddt


def _mamba2t_autograd_setup(ctx, inputs, output):
    x, A_log, B, C, dt = inputs
    y, states = output
    ctx.save_for_backward(x, A_log, B, C, dt, states)


torch.library.register_autograd(
    "mamba2t::chunk_fwd", _mamba2t_autograd_bwd, setup_context=_mamba2t_autograd_setup,
)


def mamba2_ssd_triton_autograd(x, A_log, B, C, dt, chunk_size=64):
    """Triton fwd+bwd via torch.library custom ops (fullgraph=True safe)."""
    assert chunk_size == 64, "Only chunk_size=64 supported"
    y, _states = torch.ops.mamba2t.chunk_fwd(x, A_log, B, C, dt)
    return y
