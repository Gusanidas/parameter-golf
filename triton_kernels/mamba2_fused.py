"""Mamba2 Fused Conv1d + SiLU + SSD — single kernel per (batch, head).

Eliminates 3 separate conv kernel launches by fusing causal depthwise conv1d
and SiLU activation into the chunked SSD forward kernel. The conv uses shifted
global memory loads (hitting L2 cache from adjacent chunks) instead of
maintaining history registers.

Forward:  conv(x,B,C) + SiLU + SSD  — 1 kernel launch
Backward: recompute conv+SiLU, SSD_bwd, SiLU_bwd — 1 kernel, then 3x conv_bwd

Usage:
    from triton_kernels import mamba2_fused_triton_autograd
    y = mamba2_fused_triton_autograd(x, A_log, B, C, dt, conv_w_x, conv_w_b, conv_w_c)
"""
from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl

from .conv1d import causal_conv1d_triton_bwd


# ═══════════════════════════════════════════════════════════════════════════
# Fused Forward: Conv1d + SiLU + Mamba2 SSD
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _mamba2_fused_fwd(
    # Raw inputs (pre-conv)
    X, B_IN, C_IN, DT, A_LOG,
    # Conv weights: [H, D, 4] layout, contiguous
    CONV_WX, CONV_WB, CONV_WC,
    # Outputs: main (Y, STATES) + saved post-conv pre-silu activations
    Y, STATES, CONV_X_OUT, CONV_B_OUT, CONV_C_OUT,
    # X strides [B, T, H, P]
    s_xb, s_xt, s_xh, s_xd,
    # B strides [B, T, H, N]
    s_bb, s_bt, s_bh, s_bn,
    # C strides [B, T, H, N]
    s_cb, s_ct, s_ch, s_cn,
    # DT strides [B, T, H]
    s_db, s_dt_, s_dh,
    # Y strides [B, T, H, P]
    s_yb, s_yt, s_yh, s_yd,
    # STATES strides [B, H, NC+1, N, P]
    s_sb, s_sh, s_sc, s_sn, s_sd,
    nheads, num_chunks, seqlen,
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

    # Load conv weights for x: [H, P, 4]
    cwx_base = h * HDIM * 4
    wx0 = tl.load(CONV_WX + cwx_base + d_offs * 4 + 0).to(tl.float32)
    wx1 = tl.load(CONV_WX + cwx_base + d_offs * 4 + 1).to(tl.float32)
    wx2 = tl.load(CONV_WX + cwx_base + d_offs * 4 + 2).to(tl.float32)
    wx3 = tl.load(CONV_WX + cwx_base + d_offs * 4 + 3).to(tl.float32)

    # Load conv weights for B: [H, N, 4]
    cwb_base = h * SDIM * 4
    wb0 = tl.load(CONV_WB + cwb_base + n_offs * 4 + 0).to(tl.float32)
    wb1 = tl.load(CONV_WB + cwb_base + n_offs * 4 + 1).to(tl.float32)
    wb2 = tl.load(CONV_WB + cwb_base + n_offs * 4 + 2).to(tl.float32)
    wb3 = tl.load(CONV_WB + cwb_base + n_offs * 4 + 3).to(tl.float32)

    # Load conv weights for C: [H, N, 4]
    cwc_base = h * SDIM * 4
    wc0 = tl.load(CONV_WC + cwc_base + n_offs * 4 + 0).to(tl.float32)
    wc1 = tl.load(CONV_WC + cwc_base + n_offs * 4 + 1).to(tl.float32)
    wc2 = tl.load(CONV_WC + cwc_base + n_offs * 4 + 2).to(tl.float32)
    wc3 = tl.load(CONV_WC + cwc_base + n_offs * 4 + 3).to(tl.float32)

    for ci in range(num_chunks):
        t0 = ci * CHUNK

        # Save state at start of chunk
        st_ptrs = STATES + b * s_sb + h * s_sh + ci * s_sc + n_offs[:, None] * s_sn + d_offs[None, :] * s_sd
        tl.store(st_ptrs, state)

        # ── Conv1d on X: load raw + shifted, apply conv + SiLU ──
        t_abs = t0 + c_offs  # [CHUNK] absolute timestep indices
        x_ptrs_base = X + b * s_xb + h * s_xh

        x_raw = tl.load(x_ptrs_base + t_abs[:, None] * s_xt + d_offs[None, :] * s_xd).to(tl.float32)
        x_m1 = tl.load(x_ptrs_base + (t_abs[:, None] - 1) * s_xt + d_offs[None, :] * s_xd,
                        mask=(t_abs[:, None] - 1) >= 0, other=0.0).to(tl.float32)
        x_m2 = tl.load(x_ptrs_base + (t_abs[:, None] - 2) * s_xt + d_offs[None, :] * s_xd,
                        mask=(t_abs[:, None] - 2) >= 0, other=0.0).to(tl.float32)
        x_m3 = tl.load(x_ptrs_base + (t_abs[:, None] - 3) * s_xt + d_offs[None, :] * s_xd,
                        mask=(t_abs[:, None] - 3) >= 0, other=0.0).to(tl.float32)
        conv_x = wx0[None, :] * x_raw + wx1[None, :] * x_m1 + wx2[None, :] * x_m2 + wx3[None, :] * x_m3
        tl.store(CONV_X_OUT + b * s_xb + t_abs[:, None] * s_xt + h * s_xh + d_offs[None, :] * s_xd, conv_x)
        x_c = conv_x * tl.sigmoid(conv_x)  # SiLU

        # ── Conv1d on B ──
        b_ptrs_base = B_IN + b * s_bb + h * s_bh
        b_raw = tl.load(b_ptrs_base + t_abs[:, None] * s_bt + n_offs[None, :] * s_bn).to(tl.float32)
        b_m1 = tl.load(b_ptrs_base + (t_abs[:, None] - 1) * s_bt + n_offs[None, :] * s_bn,
                        mask=(t_abs[:, None] - 1) >= 0, other=0.0).to(tl.float32)
        b_m2 = tl.load(b_ptrs_base + (t_abs[:, None] - 2) * s_bt + n_offs[None, :] * s_bn,
                        mask=(t_abs[:, None] - 2) >= 0, other=0.0).to(tl.float32)
        b_m3 = tl.load(b_ptrs_base + (t_abs[:, None] - 3) * s_bt + n_offs[None, :] * s_bn,
                        mask=(t_abs[:, None] - 3) >= 0, other=0.0).to(tl.float32)
        conv_b = wb0[None, :] * b_raw + wb1[None, :] * b_m1 + wb2[None, :] * b_m2 + wb3[None, :] * b_m3
        tl.store(CONV_B_OUT + b * s_bb + t_abs[:, None] * s_bt + h * s_bh + n_offs[None, :] * s_bn, conv_b)
        B_c = conv_b * tl.sigmoid(conv_b)

        # ── Conv1d on C ──
        c_ptrs_base = C_IN + b * s_cb + h * s_ch
        c_raw = tl.load(c_ptrs_base + t_abs[:, None] * s_ct + n_offs[None, :] * s_cn).to(tl.float32)
        c_m1 = tl.load(c_ptrs_base + (t_abs[:, None] - 1) * s_ct + n_offs[None, :] * s_cn,
                        mask=(t_abs[:, None] - 1) >= 0, other=0.0).to(tl.float32)
        c_m2 = tl.load(c_ptrs_base + (t_abs[:, None] - 2) * s_ct + n_offs[None, :] * s_cn,
                        mask=(t_abs[:, None] - 2) >= 0, other=0.0).to(tl.float32)
        c_m3 = tl.load(c_ptrs_base + (t_abs[:, None] - 3) * s_ct + n_offs[None, :] * s_cn,
                        mask=(t_abs[:, None] - 3) >= 0, other=0.0).to(tl.float32)
        conv_c = wc0[None, :] * c_raw + wc1[None, :] * c_m1 + wc2[None, :] * c_m2 + wc3[None, :] * c_m3
        tl.store(CONV_C_OUT + b * s_cb + t_abs[:, None] * s_ct + h * s_ch + n_offs[None, :] * s_cn, conv_c)
        C_c = conv_c * tl.sigmoid(conv_c)

        # ── DT (no conv) ──
        dt_c = tl.load(DT + b * s_db + t_abs * s_dt_ + h * s_dh).to(tl.float32)

        # ── SSD computation (identical to non-fused) ──
        log_a = tl.maximum(A_h * dt_c, -20.0)
        cum = tl.cumsum(log_a, axis=0)

        Cs = tl.dot(C_c, state)
        y_state = Cs * tl.exp(cum)[:, None]

        qk = tl.dot(C_c, tl.trans(B_c))
        decay_diff = cum[:, None] - cum[None, :]
        causal = c_offs[:, None] >= c_offs[None, :]
        decay_diff = tl.where(causal, decay_diff, -float('inf'))
        attn = qk * tl.exp(decay_diff)
        y_intra = tl.dot(attn, x_c)

        y_chunk = y_state + y_intra
        y_ptrs = Y + b * s_yb + (t0 + c_offs[:, None]) * s_yt + h * s_yh + d_offs[None, :] * s_yd
        tl.store(y_ptrs, y_chunk)

        total_log = tl.sum(log_a, axis=0)
        decay_to_end = tl.exp(total_log - cum)
        state = state * tl.exp(total_log) + tl.dot(tl.trans(B_c), x_c * decay_to_end[:, None])

    # Save final state
    st_ptrs = STATES + b * s_sb + h * s_sh + num_chunks * s_sc + n_offs[:, None] * s_sn + d_offs[None, :] * s_sd
    tl.store(st_ptrs, state)


# ═══════════════════════════════════════════════════════════════════════════
# Fused Backward: SSD_bwd + SiLU_bwd (conv backward done separately)
# ═══════════════════════════════════════════════════════════════════════════
# Processes chunks in REVERSE order. For each chunk:
# 1. Recompute conv + SiLU from raw inputs (shifted loads)
# 2. SSD backward → d_ssd (gradients w.r.t. post-SiLU values)
# 3. SiLU backward → d_conv (gradients w.r.t. conv outputs)
# 4. Store d_conv — conv transpose + dW done by separate conv1d_bwd kernels

@triton.jit
def _mamba2_fused_bwd(
    # Raw inputs (still needed for conv_bwd elsewhere; strides reused for CONV_* too)
    X, B_IN, C_IN, DT, A_LOG,
    # Conv weights (kept for signature symmetry, unused now since we read saved conv)
    CONV_WX, CONV_WB, CONV_WC,
    # Saved states
    STATES,
    # Gradient of output
    DY,
    # Saved pre-SiLU conv outputs from forward
    CONV_X, CONV_B, CONV_C,
    # Output: gradients w.r.t. conv outputs (before conv transpose)
    D_CONV_X, D_CONV_B, D_CONV_C,
    # Output: gradient w.r.t. dt
    DDT,
    # X strides [B, T, H, P]
    s_xb, s_xt, s_xh, s_xd,
    # B strides [B, T, H, N]
    s_bb, s_bt, s_bh, s_bn,
    # C strides [B, T, H, N]
    s_cb, s_ct, s_ch, s_cn,
    # DT strides [B, T, H]
    s_db, s_dt_, s_dh,
    # States strides [B, H, NC+1, N, P]
    s_sb, s_sh, s_sc, s_sn, s_sd,
    # DY strides [B, T, H, P]
    s_dyb, s_dyt, s_dyh, s_dyd,
    # DDT strides [B, T, H]
    s_ddb, s_ddt_, s_ddh,
    nheads, num_chunks, seqlen,
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
        t_abs = t0 + c_offs

        # ══════════════════════════════════════════════════════════════
        # Step 1: Load saved pre-SiLU conv outputs, recompute only SiLU (cheap)
        # ══════════════════════════════════════════════════════════════
        conv_x = tl.load(CONV_X + b * s_xb + t_abs[:, None] * s_xt + h * s_xh + d_offs[None, :] * s_xd)
        sig_x = tl.sigmoid(conv_x)
        x_c = conv_x * sig_x

        conv_b = tl.load(CONV_B + b * s_bb + t_abs[:, None] * s_bt + h * s_bh + n_offs[None, :] * s_bn)
        sig_b = tl.sigmoid(conv_b)
        B_c = conv_b * sig_b

        conv_c = tl.load(CONV_C + b * s_cb + t_abs[:, None] * s_ct + h * s_ch + n_offs[None, :] * s_cn)
        sig_c = tl.sigmoid(conv_c)
        C_c = conv_c * sig_c

        dt_c = tl.load(DT + b * s_db + t_abs * s_dt_ + h * s_dh).to(tl.float32)

        # ══════════════════════════════════════════════════════════════
        # Step 2: SSD backward (using recomputed post-conv-SiLU values)
        # ══════════════════════════════════════════════════════════════
        st_ptrs = STATES + b * s_sb + h * s_sh + ci * s_sc + n_offs[:, None] * s_sn + d_offs[None, :] * s_sd
        state = tl.load(st_ptrs).to(tl.float32)

        dy = tl.load(DY + b * s_dyb + (t0 + c_offs[:, None]) * s_dyt + h * s_dyh + d_offs[None, :] * s_dyd).to(tl.float32)

        log_a = tl.maximum(A_h * dt_c, -20.0)
        cum = tl.cumsum(log_a, axis=0)
        total_log = tl.sum(log_a, axis=0)
        exp_cum = tl.exp(cum)
        decay_to_end = tl.exp(total_log - cum)

        qk = tl.dot(C_c, tl.trans(B_c))
        decay_diff = cum[:, None] - cum[None, :]
        causal = c_offs[:, None] >= c_offs[None, :]
        decay_diff = tl.where(causal, decay_diff, -float('inf'))
        decay = tl.exp(decay_diff)
        attn = qk * decay

        Cs = tl.dot(C_c, state)

        # d w.r.t. post-SiLU values (dx_ssm, dB_ssm, dC_ssm)
        dCs = dy * exp_cum[:, None]
        dC_ssm = tl.dot(dCs, tl.trans(state))
        d_state_read = tl.dot(tl.trans(C_c), dCs)

        d_cum = tl.sum(dy * Cs * exp_cum[:, None], axis=1)

        dx_ssm = tl.dot(tl.trans(attn), dy)
        dattn = tl.dot(dy, tl.trans(x_c))

        dqk = dattn * decay

        d_decay = dattn * qk
        d_cum = d_cum + tl.sum(d_decay * decay, axis=1)
        d_cum = d_cum - tl.sum(d_decay * decay, axis=0)

        dC_ssm = dC_ssm + tl.dot(dqk, B_c)
        dB_ssm = tl.dot(tl.trans(dqk), C_c)

        weighted_x = x_c * decay_to_end[:, None]

        d_wx = tl.dot(B_c, d_state)
        dx_ssm = dx_ssm + d_wx * decay_to_end[:, None]

        dB_state = tl.dot(weighted_x, tl.trans(d_state))
        dB_ssm = dB_ssm + dB_state

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

        # Store d_dt
        tl.store(DDT + b * s_ddb + t_abs * s_ddt_ + h * s_ddh, d_dt)

        # ══════════════════════════════════════════════════════════════
        # Step 3: SiLU backward → d_conv
        # silu(z) = z * sigmoid(z)
        # d_silu/dz = sig * (1 + z * (1 - sig))
        # ══════════════════════════════════════════════════════════════
        d_conv_x = dx_ssm * (sig_x * (1.0 + conv_x * (1.0 - sig_x)))
        d_conv_b = dB_ssm * (sig_b * (1.0 + conv_b * (1.0 - sig_b)))
        d_conv_c = dC_ssm * (sig_c * (1.0 + conv_c * (1.0 - sig_c)))

        # Store d_conv outputs (reusing input strides since same shape)
        tl.store(D_CONV_X + b * s_xb + t_abs[:, None] * s_xt + h * s_xh + d_offs[None, :] * s_xd, d_conv_x)
        tl.store(D_CONV_B + b * s_bb + t_abs[:, None] * s_bt + h * s_bh + n_offs[None, :] * s_bn, d_conv_b)
        tl.store(D_CONV_C + b * s_cb + t_abs[:, None] * s_ct + h * s_ch + n_offs[None, :] * s_cn, d_conv_c)


# ═══════════════════════════════════════════════════════════════════════════
# Python wrappers + torch.library custom ops
# ═══════════════════════════════════════════════════════════════════════════

@torch.library.custom_op("mamba2t::fused_fwd", mutates_args=())
def _mamba2t_fused_fwd(
    x: Tensor, A_log: Tensor, B: Tensor, C: Tensor, dt: Tensor,
    conv_w_x: Tensor, conv_w_b: Tensor, conv_w_c: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fused conv+SiLU+SSD forward. Returns (y, states, conv_x, conv_b, conv_c).
    conv_x/b/c are the pre-SiLU post-conv activations, saved for use in backward
    to avoid re-running the conv recomputation.
    """
    x = x.contiguous(); B = B.contiguous(); C = C.contiguous(); dt = dt.contiguous()
    Bsz, T, H, P = x.shape
    N = B.shape[-1]
    CS = 64
    NC = T // CS
    y = x.new_empty(x.shape)
    states = torch.empty(Bsz, H, NC + 1, N, P, device=x.device, dtype=torch.float32)
    conv_x_save = torch.empty_like(x, dtype=torch.float32)
    conv_b_save = torch.empty_like(B, dtype=torch.float32)
    conv_c_save = torch.empty_like(C, dtype=torch.float32)
    grid = (Bsz * H,)
    _mamba2_fused_fwd[grid](
        x, B, C, dt, A_log,
        conv_w_x, conv_w_b, conv_w_c,
        y, states, conv_x_save, conv_b_save, conv_c_save,
        *x.stride(), *B.stride(), *C.stride(), *dt.stride(),
        *y.stride(), *states.stride(),
        H, NC, T,
        CHUNK=CS, HDIM=P, SDIM=N,
    )
    return y, states, conv_x_save, conv_b_save, conv_c_save


@torch.library.register_fake("mamba2t::fused_fwd")
def _mamba2t_fused_fwd_fake(x, A_log, B, C, dt, conv_w_x, conv_w_b, conv_w_c):
    Bsz, T, H, P = x.shape
    N = B.shape[-1]
    NC = T // 64
    y = x.new_empty(Bsz, T, H, P)
    states = torch.empty(Bsz, H, NC + 1, N, P, device=x.device, dtype=torch.float32)
    conv_x_save = torch.empty_like(x, dtype=torch.float32)
    conv_b_save = torch.empty_like(B, dtype=torch.float32)
    conv_c_save = torch.empty_like(C, dtype=torch.float32)
    return y, states, conv_x_save, conv_b_save, conv_c_save


@torch.library.custom_op("mamba2t::fused_bwd", mutates_args=())
def _mamba2t_fused_bwd(
    dy: Tensor, x: Tensor, A_log: Tensor, B: Tensor, C: Tensor,
    dt: Tensor, states: Tensor,
    conv_w_x: Tensor, conv_w_b: Tensor, conv_w_c: Tensor,
    conv_x_save: Tensor, conv_b_save: Tensor, conv_c_save: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fused backward. Returns (dx, dA_log, dB, dC, ddt, d_conv_w_x, d_conv_w_b, d_conv_w_c)."""
    x = x.contiguous(); B = B.contiguous(); C = C.contiguous()
    dt = dt.contiguous(); dy = dy.contiguous()
    conv_x_save = conv_x_save.contiguous()
    conv_b_save = conv_b_save.contiguous()
    conv_c_save = conv_c_save.contiguous()
    Bsz, T, H, P = x.shape
    N = B.shape[-1]
    CS = 64
    NC = T // CS

    d_conv_x = torch.empty_like(x)
    d_conv_b = torch.empty_like(B)
    d_conv_c = torch.empty_like(C)
    ddt = dt.new_empty(dt.shape)

    grid = (Bsz * H,)
    _mamba2_fused_bwd[grid](
        x, B, C, dt, A_log,
        conv_w_x, conv_w_b, conv_w_c,
        states, dy,
        conv_x_save, conv_b_save, conv_c_save,
        d_conv_x, d_conv_b, d_conv_c, ddt,
        *x.stride(), *B.stride(), *C.stride(), *dt.stride(),
        *states.stride(), *dy.stride(), *ddt.stride(),
        H, NC, T,
        CHUNK=CS, HDIM=P, SDIM=N,
    )

    dx, d_cwx = causal_conv1d_triton_bwd(d_conv_x, x, conv_w_x)
    dB, d_cwb = causal_conv1d_triton_bwd(d_conv_b, B, conv_w_b)
    dC, d_cwc = causal_conv1d_triton_bwd(d_conv_c, C, conv_w_c)

    dA_log = (ddt * dt).sum(dim=(0, 1))  # [H]
    return dx, dA_log, dB, dC, ddt, d_cwx, d_cwb, d_cwc


@torch.library.register_fake("mamba2t::fused_bwd")
def _mamba2t_fused_bwd_fake(dy, x, A_log, B, C, dt, states, conv_w_x, conv_w_b, conv_w_c,
                             conv_x_save, conv_b_save, conv_c_save):
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


def _fused_autograd_bwd(ctx, do, d_states, d_cx, d_cb, d_cc):
    x, A_log, B, C, dt, states, conv_w_x, conv_w_b, conv_w_c, conv_x_save, conv_b_save, conv_c_save = ctx.saved_tensors
    dx, dA_log, dB, dC, ddt, d_cwx, d_cwb, d_cwc = torch.ops.mamba2t.fused_bwd(
        do.contiguous(), x, A_log, B, C, dt, states,
        conv_w_x, conv_w_b, conv_w_c,
        conv_x_save, conv_b_save, conv_c_save,
    )
    return dx, dA_log, dB, dC, ddt, d_cwx, d_cwb, d_cwc


def _fused_autograd_setup(ctx, inputs, output):
    x, A_log, B, C, dt, conv_w_x, conv_w_b, conv_w_c = inputs
    y, states, conv_x_save, conv_b_save, conv_c_save = output
    ctx.save_for_backward(x, A_log, B, C, dt, states, conv_w_x, conv_w_b, conv_w_c,
                          conv_x_save, conv_b_save, conv_c_save)


torch.library.register_autograd(
    "mamba2t::fused_fwd", _fused_autograd_bwd, setup_context=_fused_autograd_setup,
)


def mamba2_fused_triton_autograd(x, A_log, B, C, dt, conv_w_x, conv_w_b, conv_w_c, chunk_size=64):
    """Fused conv1d + SiLU + SSD via single Triton kernel (fullgraph=True safe).

    Forward: 1 kernel (conv+SiLU+SSD fused)
    Backward: 1 kernel (SSD_bwd+SiLU_bwd fused) + 3 conv1d_bwd kernels

    Args:
        x: [B, T, H, P] raw input (pre-conv)
        A_log: [H] log of decay rates
        B: [B, T, H, N] raw B matrix (pre-conv)
        C: [B, T, H, N] raw C matrix (pre-conv)
        dt: [B, T, H] timestep sizes
        conv_w_x: [H, P, 4] conv weights for x
        conv_w_b: [H, N, 4] conv weights for B
        conv_w_c: [H, N, 4] conv weights for C

    Returns:
        y: [B, T, H, P] output
    """
    assert chunk_size in (64, 128), "Only chunk_size 64 or 128 supported"
    y, _states, _cx, _cb, _cc = torch.ops.mamba2t.fused_fwd(x, A_log, B, C, dt, conv_w_x, conv_w_b, conv_w_c)
    return y
