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
    X, B_IN, C_IN, DT, A_LOG, D_SKIP,
    # Conv weights: [H, D, 4] layout, contiguous
    CONV_WX, CONV_WB, CONV_WC,
    # Outputs
    Y, STATES,
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
    D_h = tl.load(D_SKIP + h).to(tl.float32)
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
        C_c = conv_c * tl.sigmoid(conv_c)

        # ── DT (no conv) ──
        dt_c = tl.load(DT + b * s_db + t_abs * s_dt_ + h * s_dh).to(tl.float32)

        # ── SSD computation (identical to non-fused) ──
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

        y_chunk = y_state + y_intra + x_c * D_h
        y_ptrs = Y + b * s_yb + (t0 + c_offs[:, None]) * s_yt + h * s_yh + d_offs[None, :] * s_yd
        tl.store(y_ptrs, y_chunk)

        total_log = tl.sum(log_a, axis=0)
        decay_to_end = tl.exp(total_log - cum)
        state = state * tl.exp(total_log) + tl.dot(
            tl.trans(B_c), x_c * decay_to_end[:, None], input_precision="tf32",
        )

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
#
# SMEM: B200 cap is 232,448 B. With Triton's default num_stages=3 the bwd
# allocates ~304 KB and triggers OutOfResources. Pin a single low-stage config
# (num_stages=1, num_warps=4) keyed by the constexpr block dims so it fits in
# ≤227 KB SMEM on Blackwell while staying portable to H100.

@triton.autotune(
    configs=[
        # num_stages=1 avoids the multi-stage SMEM doubling that pushes us over
        # the 232,448 B Blackwell cap. The kernel's per-chunk loop body is
        # arithmetic-heavy and gains little from pipelining anyway.
        triton.Config({}, num_stages=1, num_warps=4),
    ],
    key=["CHUNK", "HDIM", "SDIM"],
)
@triton.jit
def _mamba2_fused_bwd(
    # Raw inputs (still needed for conv_bwd elsewhere; strides reused for CONV_* too)
    X, B_IN, C_IN, DT, A_LOG, D_SKIP,
    # Conv weights
    CONV_WX, CONV_WB, CONV_WC,
    # Saved states
    STATES,
    # Gradient of output
    DY,
    # Output: gradients w.r.t. conv outputs (before conv transpose)
    D_CONV_X, D_CONV_B, D_CONV_C,
    # Output: gradient w.r.t. dt
    DDT, DD_SKIP,
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
    # D_CONV_X/B/C strides
    s_dcxb, s_dcxt, s_dcxh, s_dcxd,
    s_dcbb, s_dcbt, s_dcbh, s_dcbn,
    s_dccb, s_dcct, s_dcch, s_dccn,
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
    D_h = tl.load(D_SKIP + h).to(tl.float32)

    c_offs = tl.arange(0, CHUNK)
    d_offs = tl.arange(0, HDIM)
    n_offs = tl.arange(0, SDIM)

    d_state = tl.zeros((SDIM, HDIM), dtype=tl.float32)

    cwx_base = h * HDIM * 4
    wx0 = tl.load(CONV_WX + cwx_base + d_offs * 4 + 0).to(tl.float32)
    wx1 = tl.load(CONV_WX + cwx_base + d_offs * 4 + 1).to(tl.float32)
    wx2 = tl.load(CONV_WX + cwx_base + d_offs * 4 + 2).to(tl.float32)
    wx3 = tl.load(CONV_WX + cwx_base + d_offs * 4 + 3).to(tl.float32)

    cwb_base = h * SDIM * 4
    wb0 = tl.load(CONV_WB + cwb_base + n_offs * 4 + 0).to(tl.float32)
    wb1 = tl.load(CONV_WB + cwb_base + n_offs * 4 + 1).to(tl.float32)
    wb2 = tl.load(CONV_WB + cwb_base + n_offs * 4 + 2).to(tl.float32)
    wb3 = tl.load(CONV_WB + cwb_base + n_offs * 4 + 3).to(tl.float32)

    cwc_base = h * SDIM * 4
    wc0 = tl.load(CONV_WC + cwc_base + n_offs * 4 + 0).to(tl.float32)
    wc1 = tl.load(CONV_WC + cwc_base + n_offs * 4 + 1).to(tl.float32)
    wc2 = tl.load(CONV_WC + cwc_base + n_offs * 4 + 2).to(tl.float32)
    wc3 = tl.load(CONV_WC + cwc_base + n_offs * 4 + 3).to(tl.float32)

    for ci_rev in range(num_chunks):
        ci = num_chunks - 1 - ci_rev
        t0 = ci * CHUNK
        t_abs = t0 + c_offs

        # ══════════════════════════════════════════════════════════════
        # Step 1: Recompute pre-SiLU conv outputs and SiLU.
        # ══════════════════════════════════════════════════════════════
        x_ptrs_base = X + b * s_xb + h * s_xh
        x_raw = tl.load(x_ptrs_base + t_abs[:, None] * s_xt + d_offs[None, :] * s_xd).to(tl.float32)
        x_m1 = tl.load(
            x_ptrs_base + (t_abs[:, None] - 1) * s_xt + d_offs[None, :] * s_xd,
            mask=(t_abs[:, None] - 1) >= 0, other=0.0,
        ).to(tl.float32)
        x_m2 = tl.load(
            x_ptrs_base + (t_abs[:, None] - 2) * s_xt + d_offs[None, :] * s_xd,
            mask=(t_abs[:, None] - 2) >= 0, other=0.0,
        ).to(tl.float32)
        x_m3 = tl.load(
            x_ptrs_base + (t_abs[:, None] - 3) * s_xt + d_offs[None, :] * s_xd,
            mask=(t_abs[:, None] - 3) >= 0, other=0.0,
        ).to(tl.float32)
        conv_x = wx0[None, :] * x_raw + wx1[None, :] * x_m1 + wx2[None, :] * x_m2 + wx3[None, :] * x_m3
        sig_x = tl.sigmoid(conv_x)
        x_c = conv_x * sig_x

        b_ptrs_base = B_IN + b * s_bb + h * s_bh
        b_raw = tl.load(b_ptrs_base + t_abs[:, None] * s_bt + n_offs[None, :] * s_bn).to(tl.float32)
        b_m1 = tl.load(
            b_ptrs_base + (t_abs[:, None] - 1) * s_bt + n_offs[None, :] * s_bn,
            mask=(t_abs[:, None] - 1) >= 0, other=0.0,
        ).to(tl.float32)
        b_m2 = tl.load(
            b_ptrs_base + (t_abs[:, None] - 2) * s_bt + n_offs[None, :] * s_bn,
            mask=(t_abs[:, None] - 2) >= 0, other=0.0,
        ).to(tl.float32)
        b_m3 = tl.load(
            b_ptrs_base + (t_abs[:, None] - 3) * s_bt + n_offs[None, :] * s_bn,
            mask=(t_abs[:, None] - 3) >= 0, other=0.0,
        ).to(tl.float32)
        conv_b = wb0[None, :] * b_raw + wb1[None, :] * b_m1 + wb2[None, :] * b_m2 + wb3[None, :] * b_m3
        sig_b = tl.sigmoid(conv_b)
        B_c = conv_b * sig_b

        c_ptrs_base = C_IN + b * s_cb + h * s_ch
        c_raw = tl.load(c_ptrs_base + t_abs[:, None] * s_ct + n_offs[None, :] * s_cn).to(tl.float32)
        c_m1 = tl.load(
            c_ptrs_base + (t_abs[:, None] - 1) * s_ct + n_offs[None, :] * s_cn,
            mask=(t_abs[:, None] - 1) >= 0, other=0.0,
        ).to(tl.float32)
        c_m2 = tl.load(
            c_ptrs_base + (t_abs[:, None] - 2) * s_ct + n_offs[None, :] * s_cn,
            mask=(t_abs[:, None] - 2) >= 0, other=0.0,
        ).to(tl.float32)
        c_m3 = tl.load(
            c_ptrs_base + (t_abs[:, None] - 3) * s_ct + n_offs[None, :] * s_cn,
            mask=(t_abs[:, None] - 3) >= 0, other=0.0,
        ).to(tl.float32)
        conv_c = wc0[None, :] * c_raw + wc1[None, :] * c_m1 + wc2[None, :] * c_m2 + wc3[None, :] * c_m3
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

        qk = tl.dot(C_c, tl.trans(B_c), input_precision="tf32")
        decay_diff = cum[:, None] - cum[None, :]
        causal = c_offs[:, None] >= c_offs[None, :]
        decay_diff = tl.where(causal, decay_diff, -float('inf'))
        decay = tl.exp(decay_diff)
        attn = qk * decay

        Cs = tl.dot(C_c, state, input_precision="tf32")

        # d w.r.t. post-SiLU values (dx_ssm, dB_ssm, dC_ssm)
        dCs = dy * exp_cum[:, None]
        dC_ssm = tl.dot(dCs, tl.trans(state), input_precision="tf32")
        d_state_read = tl.dot(tl.trans(C_c), dCs, input_precision="tf32")

        d_cum = tl.sum(dy * Cs * exp_cum[:, None], axis=1)

        dx_ssm = tl.dot(tl.trans(attn), dy, input_precision="tf32")
        dattn = tl.dot(dy, tl.trans(x_c), input_precision="tf32")

        dqk = dattn * decay

        d_decay = dattn * qk
        d_cum = d_cum + tl.sum(d_decay * decay, axis=1)
        d_cum = d_cum - tl.sum(d_decay * decay, axis=0)

        dC_ssm = dC_ssm + tl.dot(dqk, B_c, input_precision="tf32")
        dB_ssm = tl.dot(tl.trans(dqk), C_c, input_precision="tf32")

        weighted_x = x_c * decay_to_end[:, None]

        d_wx = tl.dot(B_c, d_state, input_precision="tf32")
        dx_ssm = dx_ssm + d_wx * decay_to_end[:, None]

        dB_state = tl.dot(weighted_x, tl.trans(d_state), input_precision="tf32")
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
        # Add gradient from the fused D skip: y += D[h] * x_c.
        dx_ssm = dx_ssm + dy * D_h
        tl.atomic_add(DD_SKIP + h, tl.sum(dy * x_c))

        # Step 3: SiLU backward → d_conv
        # silu(z) = z * sigmoid(z)
        # d_silu/dz = sig * (1 + z * (1 - sig))
        # ══════════════════════════════════════════════════════════════
        d_conv_x = dx_ssm * (sig_x * (1.0 + conv_x * (1.0 - sig_x)))
        d_conv_b = dB_ssm * (sig_b * (1.0 + conv_b * (1.0 - sig_b)))
        d_conv_c = dC_ssm * (sig_c * (1.0 + conv_c * (1.0 - sig_c)))

        # Store compact d_conv outputs for the depthwise conv transpose kernels.
        tl.store(D_CONV_X + b * s_dcxb + t_abs[:, None] * s_dcxt + h * s_dcxh + d_offs[None, :] * s_dcxd, d_conv_x)
        tl.store(D_CONV_B + b * s_dcbb + t_abs[:, None] * s_dcbt + h * s_dcbh + n_offs[None, :] * s_dcbn, d_conv_b)
        tl.store(D_CONV_C + b * s_dccb + t_abs[:, None] * s_dcct + h * s_dcch + n_offs[None, :] * s_dccn, d_conv_c)


# ═══════════════════════════════════════════════════════════════════════════
# Python wrappers + torch.library custom ops
# ═══════════════════════════════════════════════════════════════════════════

@torch.library.custom_op("mamba2t::fused_fwd", mutates_args=())
def _mamba2t_fused_fwd(
    x: Tensor, A_log: Tensor, B: Tensor, C: Tensor, dt: Tensor,
    conv_w_x: Tensor, conv_w_b: Tensor, conv_w_c: Tensor, D_skip: Tensor,
) -> tuple[Tensor, Tensor]:
    """Fused conv+SiLU+SSD forward. Returns (y, states).

    Backward recomputes the cheap width-4 convolutions instead of saving three
    fp32 full-sequence pre-SiLU activation tensors.
    """
    Bsz, T, H, P = x.shape
    N = B.shape[-1]
    CS = 64
    if T % CS != 0:
        raise ValueError("mamba2 fused kernel requires sequence length divisible by 64")
    NC = T // CS
    y = x.new_empty(x.shape)
    states = torch.empty(Bsz, H, NC + 1, N, P, device=x.device, dtype=torch.float32)
    grid = (Bsz * H,)
    _mamba2_fused_fwd[grid](
        x, B, C, dt, A_log, D_skip,
        conv_w_x, conv_w_b, conv_w_c,
        y, states,
        *x.stride(), *B.stride(), *C.stride(), *dt.stride(),
        *y.stride(), *states.stride(),
        H, NC, T,
        CHUNK=CS, HDIM=P, SDIM=N,
    )
    return y, states


@torch.library.register_fake("mamba2t::fused_fwd")
def _mamba2t_fused_fwd_fake(x, A_log, B, C, dt, conv_w_x, conv_w_b, conv_w_c, D_skip):
    Bsz, T, H, P = x.shape
    N = B.shape[-1]
    NC = T // 64
    y = x.new_empty(Bsz, T, H, P)
    states = torch.empty(Bsz, H, NC + 1, N, P, device=x.device, dtype=torch.float32)
    return y, states


@torch.library.custom_op("mamba2t::fused_bwd", mutates_args=())
def _mamba2t_fused_bwd(
    dy: Tensor, x: Tensor, A_log: Tensor, B: Tensor, C: Tensor,
    dt: Tensor, states: Tensor,
    conv_w_x: Tensor, conv_w_b: Tensor, conv_w_c: Tensor, D_skip: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fused backward. Returns (dx, dA_log, dB, dC, ddt, d_conv_w_x, d_conv_w_b, d_conv_w_c, dD)."""
    Bsz, T, H, P = x.shape
    N = B.shape[-1]
    CS = 64
    if T % CS != 0:
        raise ValueError("mamba2 fused backward requires sequence length divisible by 64")
    NC = T // CS

    d_conv_x = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    d_conv_b = torch.empty(B.shape, device=B.device, dtype=B.dtype)
    d_conv_c = torch.empty(C.shape, device=C.device, dtype=C.dtype)
    ddt = dt.new_empty(dt.shape)
    dD = torch.zeros_like(D_skip)

    grid = (Bsz * H,)
    _mamba2_fused_bwd[grid](
        x, B, C, dt, A_log, D_skip,
        conv_w_x, conv_w_b, conv_w_c,
        states, dy,
        d_conv_x, d_conv_b, d_conv_c, ddt, dD,
        *x.stride(), *B.stride(), *C.stride(), *dt.stride(),
        *states.stride(), *dy.stride(),
        *d_conv_x.stride(), *d_conv_b.stride(), *d_conv_c.stride(),
        *ddt.stride(),
        H, NC, T,
        CHUNK=CS, HDIM=P, SDIM=N,
    )

    dx, d_cwx = causal_conv1d_triton_bwd(d_conv_x, x, conv_w_x)
    dB, d_cwb = causal_conv1d_triton_bwd(d_conv_b, B, conv_w_b)
    dC, d_cwc = causal_conv1d_triton_bwd(d_conv_c, C, conv_w_c)

    dA_log = (ddt * dt).sum(dim=(0, 1))  # [H]
    return dx, dA_log, dB, dC, ddt, d_cwx, d_cwb, d_cwc, dD


@torch.library.register_fake("mamba2t::fused_bwd")
def _mamba2t_fused_bwd_fake(dy, x, A_log, B, C, dt, states, conv_w_x, conv_w_b, conv_w_c, D_skip):
    return (
        x.new_empty(x.shape),
        A_log.new_empty(A_log.shape),
        B.new_empty(B.shape),
        C.new_empty(C.shape),
        dt.new_empty(dt.shape),
        conv_w_x.new_empty(conv_w_x.shape),
        conv_w_b.new_empty(conv_w_b.shape),
        conv_w_c.new_empty(conv_w_c.shape),
        D_skip.new_empty(D_skip.shape),
    )


def _fused_autograd_bwd(ctx, do, d_states):
    x, A_log, B, C, dt, states, conv_w_x, conv_w_b, conv_w_c, D_skip = ctx.saved_tensors
    dx, dA_log, dB, dC, ddt, d_cwx, d_cwb, d_cwc, dD = torch.ops.mamba2t.fused_bwd(
        do, x, A_log, B, C, dt, states,
        conv_w_x, conv_w_b, conv_w_c, D_skip,
    )
    return dx, dA_log, dB, dC, ddt, d_cwx, d_cwb, d_cwc, dD


def _fused_autograd_setup(ctx, inputs, output):
    x, A_log, B, C, dt, conv_w_x, conv_w_b, conv_w_c, D_skip = inputs
    y, states = output
    ctx.save_for_backward(x, A_log, B, C, dt, states, conv_w_x, conv_w_b, conv_w_c, D_skip)


torch.library.register_autograd(
    "mamba2t::fused_fwd", _fused_autograd_bwd, setup_context=_fused_autograd_setup,
)


def mamba2_fused_triton_autograd(x, A_log, B, C, dt, conv_w_x, conv_w_b, conv_w_c,
                                 D_skip=None, chunk_size=64):
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
    assert chunk_size == 64, "Only chunk_size=64 supported"
    if D_skip is None:
        D_skip = A_log.new_zeros(A_log.shape)
    y, _states = torch.ops.mamba2t.fused_fwd(
        x, A_log, B, C, dt, conv_w_x, conv_w_b, conv_w_c, D_skip,
    )
    return y
