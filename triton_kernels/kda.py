"""Kimi Delta Attention (KDA) — Chunked Triton forward + backward kernels.

KDA is a gated delta-rule linear attention with per-key-dimension decay.
Per-key decay is absorbed into the keys via decay-weighted factorization:
    k_pos = k * exp(cum_g)    — "forward-decayed" keys
    k_neg = k * exp(-cum_g)   — "inverse-decayed" keys
    L[i,j] = beta[i] * k_pos[i] @ k_neg[j]^T   — standard matmul

Reference: https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda

Usage:
    from triton_kernels import kda_triton_autograd
    y = kda_triton_autograd(q, k, v, g, beta)
"""
from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════
# Triton helpers
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _tri_solve(A, rhs, c_offs, CHUNK: tl.constexpr, UPPER: tl.constexpr):
    """Solve (I + A)x = rhs via column-oriented substitution.

    A is [CHUNK, CHUNK], strictly lower-triangular (UPPER=0) or
    strictly upper-triangular (UPPER=1).
    rhs is [CHUNK, D].  Returns x = (I+A)^{-1} @ rhs.
    """
    x = rhs
    for step in range(CHUNK - 1):
        j: tl.constexpr = (CHUNK - 1 - step) if UPPER else step
        mask_j = (c_offs == j).to(tl.float32)
        x_j = tl.sum(x * mask_j[:, None], axis=0)
        A_col = tl.sum(A * mask_j[None, :], axis=1)
        x = x - A_col[:, None] * x_j[None, :]
    return x


# ═══════════════════════════════════════════════════════════════════════════
# Forward Kernel
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _kda_chunk_fwd(
    Q, K, V, G, BETA, Y, STATES,
    s_qb, s_qt, s_qh, s_qk,
    s_kb, s_kt, s_kh, s_kk,
    s_vb, s_vt, s_vh, s_vv,
    s_gb, s_gt, s_gh, s_gk,           # G is [B,T,H,K] — per-key
    s_bb, s_bt, s_bh,
    s_yb, s_yt, s_yh, s_yv,
    s_sb, s_sh, s_sc, s_sk, s_sv,
    nheads, num_chunks,
    CHUNK: tl.constexpr,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // nheads
    h = pid % nheads

    c_offs = tl.arange(0, CHUNK)
    k_offs = tl.arange(0, K_DIM)
    v_offs = tl.arange(0, V_DIM)

    state = tl.zeros((K_DIM, V_DIM), dtype=tl.float32)

    for ci in range(num_chunks):
        t0 = ci * CHUNK

        # Store beginning-of-chunk state
        st_ptrs = (STATES + b * s_sb + h * s_sh + ci * s_sc
                   + k_offs[:, None] * s_sk + v_offs[None, :] * s_sv)
        tl.store(st_ptrs, state)

        # Load chunk data  (q is pre-scaled by the caller)
        q_c = tl.load(Q + b * s_qb + (t0 + c_offs[:, None]) * s_qt
                       + h * s_qh + k_offs[None, :] * s_qk).to(tl.float32)
        k_c = tl.load(K + b * s_kb + (t0 + c_offs[:, None]) * s_kt
                       + h * s_kh + k_offs[None, :] * s_kk).to(tl.float32)
        v_c = tl.load(V + b * s_vb + (t0 + c_offs[:, None]) * s_vt
                       + h * s_vh + v_offs[None, :] * s_vv).to(tl.float32)
        # Per-key gate [C, K]
        g_c = tl.load(G + b * s_gb + (t0 + c_offs[:, None]) * s_gt
                       + h * s_gh + k_offs[None, :] * s_gk).to(tl.float32)
        beta_c = tl.load(BETA + b * s_bb + (t0 + c_offs) * s_bt
                          + h * s_bh).to(tl.float32)

        # Per-key cumulative decay [C, K]
        cum_g = tl.cumsum(g_c, axis=0)

        # Per-key midpoint shift keeps exp(±shifted) bounded (shift cancels
        # algebraically in kkt and attn dot-products).
        # shift[k] ≈ cum_g[C//2, k]; extract via masked sum.
        mid_mask = (c_offs == CHUNK // 2).to(tl.float32)    # [C]
        shift = tl.sum(cum_g * mid_mask[:, None], axis=0)   # [K]
        shifted = tl.maximum(tl.minimum(cum_g - shift[None, :], 20.0), -20.0)  # [C, K], clamped

        # Decay-weighted key factorization (bounded exponentials)
        k_pos = k_c * tl.exp(shifted)                       # [C, K]
        k_neg = k_c * tl.exp(-shifted)                      # [C, K]

        # KKT with per-key decay absorbed [C, C] (shift cancels)
        kkt = tl.dot(k_pos, tl.trans(k_neg))

        # Lower-triangular coupling
        strict_lower = c_offs[:, None] > c_offs[None, :]
        L = tl.where(strict_lower, kkt * beta_c[:, None], 0.0)

        # Solve (I+L)^{-1} applied to two RHS vectors
        u = _tri_solve(L, v_c * beta_c[:, None], c_offs, CHUNK, UPPER=0)                # [C, V]
        w = _tri_solve(L, k_pos * beta_c[:, None], c_offs, CHUNK, UPPER=0)              # [C, K]

        # w uses shifted k_pos; compensate when applying to state:
        #   w_true @ state = w_shifted @ (state * exp(shift))
        state_shifted = state * tl.exp(shift[:, None])       # [K, V]
        v_new = u - tl.dot(w, state_shifted)                 # [C, V]

        # Query with per-key decay (shifted)
        q_pos = q_c * tl.exp(shifted)                        # [C, K]

        # Inter-chunk contribution: q_pos_true @ state = q_pos_shifted @ state_shifted
        y_state = tl.dot(q_pos, state_shifted)               # [C, V]

        # Intra-chunk causal attention (shift cancels in q_pos @ k_neg^T)
        causal = c_offs[:, None] >= c_offs[None, :]
        attn = tl.where(causal, tl.dot(q_pos, tl.trans(k_neg)), 0.0)   # [C, C]
        y_intra = tl.dot(attn, v_new)                        # [C, V]

        # Store output
        y_chunk = y_state + y_intra
        y_ptrs = (Y + b * s_yb + (t0 + c_offs[:, None]) * s_yt
                  + h * s_yh + v_offs[None, :] * s_yv)
        tl.store(y_ptrs, y_chunk)

        # State update — per-key decay (use unshifted cum_g for state propagation)
        total_g = tl.sum(g_c, axis=0)                        # [K]
        dte = tl.exp(total_g[None, :] - cum_g)               # [C, K]
        state = (state * tl.exp(total_g[:, None])             # [K, V]
                 + tl.dot(tl.trans(k_c * dte), v_new))        # [K, C]@[C, V]

    # Store final state
    st_ptrs = (STATES + b * s_sb + h * s_sh + num_chunks * s_sc
               + k_offs[:, None] * s_sk + v_offs[None, :] * s_sv)
    tl.store(st_ptrs, state)


# ═══════════════════════════════════════════════════════════════════════════
# Backward Kernel
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _kda_chunk_bwd(
    Q, K, V, G, BETA, STATES, DY,
    DQ, DK, DV, DG, DBETA,
    s_qb, s_qt, s_qh, s_qk,
    s_kb, s_kt, s_kh, s_kk,
    s_vb, s_vt, s_vh, s_vv,
    s_gb, s_gt, s_gh, s_gk,
    s_bb, s_bt, s_bh,
    s_sb, s_sh, s_sc, s_sk, s_sv,
    s_dyb, s_dyt, s_dyh, s_dyv,
    s_dqb, s_dqt, s_dqh, s_dqk,
    s_dkb, s_dkt, s_dkh, s_dkk,
    s_dvb, s_dvt, s_dvh, s_dvv,
    s_dgb, s_dgt, s_dgh, s_dgk,
    s_dbb, s_dbt, s_dbh,
    nheads, num_chunks,
    CHUNK: tl.constexpr,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // nheads
    h = pid % nheads

    c_offs = tl.arange(0, CHUNK)
    k_offs = tl.arange(0, K_DIM)
    v_offs = tl.arange(0, V_DIM)

    d_state = tl.zeros((K_DIM, V_DIM), dtype=tl.float32)

    for ci_rev in range(num_chunks):
        ci = num_chunks - 1 - ci_rev
        t0 = ci * CHUNK

        # Load beginning-of-chunk state
        st_ptrs = (STATES + b * s_sb + h * s_sh + ci * s_sc
                   + k_offs[:, None] * s_sk + v_offs[None, :] * s_sv)
        state = tl.load(st_ptrs).to(tl.float32)

        # Load inputs
        q_c = tl.load(Q + b * s_qb + (t0 + c_offs[:, None]) * s_qt
                       + h * s_qh + k_offs[None, :] * s_qk).to(tl.float32)
        k_c = tl.load(K + b * s_kb + (t0 + c_offs[:, None]) * s_kt
                       + h * s_kh + k_offs[None, :] * s_kk).to(tl.float32)
        v_c = tl.load(V + b * s_vb + (t0 + c_offs[:, None]) * s_vt
                       + h * s_vh + v_offs[None, :] * s_vv).to(tl.float32)
        g_c = tl.load(G + b * s_gb + (t0 + c_offs[:, None]) * s_gt
                       + h * s_gh + k_offs[None, :] * s_gk).to(tl.float32)
        beta_c = tl.load(BETA + b * s_bb + (t0 + c_offs) * s_bt
                          + h * s_bh).to(tl.float32)
        dy = tl.load(DY + b * s_dyb + (t0 + c_offs[:, None]) * s_dyt
                      + h * s_dyh + v_offs[None, :] * s_dyv).to(tl.float32)

        # ── Recompute forward quantities (with per-key shift) ─────────────
        cum_g = tl.cumsum(g_c, axis=0)                      # [C, K]

        mid_mask = (c_offs == CHUNK // 2).to(tl.float32)
        shift = tl.sum(cum_g * mid_mask[:, None], axis=0)   # [K]
        shifted = cum_g - shift[None, :]
        exp_sh = tl.exp(shifted)
        exp_neg_sh = tl.exp(-shifted)

        k_pos = k_c * exp_sh                                # [C, K]
        k_neg = k_c * exp_neg_sh                             # [C, K]

        strict_lower = c_offs[:, None] > c_offs[None, :]
        causal = c_offs[:, None] >= c_offs[None, :]

        kkt = tl.dot(k_pos, tl.trans(k_neg))                # [C, C]
        L = tl.where(strict_lower, kkt * beta_c[:, None], 0.0)

        u = _tri_solve(L, v_c * beta_c[:, None], c_offs, CHUNK, UPPER=0)
        w = _tri_solve(L, k_pos * beta_c[:, None], c_offs, CHUNK, UPPER=0)

        state_shifted = state * tl.exp(shift[:, None])       # [K, V]
        v_new = u - tl.dot(w, state_shifted)
        q_pos = q_c * exp_sh

        qk_raw = tl.dot(q_pos, tl.trans(k_neg))
        attn = tl.where(causal, qk_raw, 0.0)

        total_g = tl.sum(g_c, axis=0)                       # [K]
        dte = tl.exp(total_g[None, :] - cum_g)              # [C, K]

        # ── Backward ──────────────────────────────────────────────────────

        # y_state = q_pos @ state_shifted
        d_q_pos = tl.dot(dy, tl.trans(state_shifted))       # [C, K]
        d_state_sh_y = tl.dot(tl.trans(q_pos), dy)          # [K, V]

        # y_intra = attn @ v_new
        d_attn = tl.dot(dy, tl.trans(v_new))                # [C, C]
        d_v_new = tl.dot(tl.trans(attn), dy)                # [C, V]

        # state_new = state*exp(total_g[:,None]) + (k*dte)^T @ v_new
        d_v_new = d_v_new + tl.dot(k_c * dte, d_state)     # [C, V]
        d_k_dte_mat = tl.dot(v_new, tl.trans(d_state))     # [C, K]
        d_dte = d_k_dte_mat * k_c                           # [C, K]
        d_k_state = d_k_dte_mat * dte                       # [C, K]

        d_total_g = tl.sum(d_state * state, axis=1) * tl.exp(total_g)
        d_total_g = d_total_g + tl.sum(d_dte * dte, axis=0)
        d_cum_g = -d_dte * dte                               # [C, K]

        # v_new = u - w @ state_shifted
        d_u = d_v_new                                        # [C, V]
        d_w = -tl.dot(d_v_new, tl.trans(state_shifted))     # [C, K]
        d_state_sh_w = -tl.dot(tl.trans(w), d_v_new)        # [K, V]

        # Convert d_state_shifted → d_state, d_shift
        d_state_sh = d_state_sh_y + d_state_sh_w
        d_state_from_sh = d_state_sh * tl.exp(shift[:, None])   # d_state contribution

        # Propagate d_state
        d_state = d_state * tl.exp(total_g[:, None]) + d_state_from_sh

        # u = (I+L)^{-1} u_rhs,  w = (I+L)^{-1} w_rhs
        LT = tl.trans(L)
        d_u_rhs = _tri_solve(LT, d_u, c_offs, CHUNK, UPPER=1)
        d_w_rhs = _tri_solve(LT, d_w, c_offs, CHUNK, UPPER=1)

        d_L = -(tl.dot(d_u_rhs, tl.trans(u)) + tl.dot(d_w_rhs, tl.trans(w)))
        d_L = tl.where(strict_lower, d_L, 0.0)

        d_kkt = tl.where(strict_lower, d_L * beta_c[:, None], 0.0)
        d_beta_L = tl.sum(tl.where(strict_lower, d_L * kkt, 0.0), axis=1)

        # kkt = k_pos @ k_neg^T  (d_kkt is strict_lower → 0*inf safe)
        d_k_pos_kkt = tl.dot(d_kkt, k_neg)                  # [C, K]
        d_k_neg_kkt = tl.dot(tl.trans(d_kkt), k_pos)        # [C, K]

        # attn = where(causal, q_pos @ k_neg^T, 0)
        d_qk_raw = tl.where(causal, d_attn, 0.0)
        d_q_pos = d_q_pos + tl.dot(d_qk_raw, k_neg)
        d_k_neg_attn = tl.dot(tl.trans(d_qk_raw), q_pos)

        d_k_neg = d_k_neg_kkt + d_k_neg_attn

        # w_rhs = k_pos * beta
        d_k_pos_w = d_w_rhs * beta_c[:, None]
        d_beta_w = tl.sum(d_w_rhs * k_pos, axis=1)
        d_k_pos = d_k_pos_kkt + d_k_pos_w

        # Gradients through shifted exponentials → d_shifted, d_k
        # k_pos = k * exp(shifted), k_neg = k * exp(-shifted), q_pos = q * exp(shifted)
        d_q_c = d_q_pos * exp_sh                             # [C, K]
        d_k_from_pos = d_k_pos * exp_sh
        d_k_from_neg = d_k_neg * exp_neg_sh

        # d_shifted accumulates from q_pos, k_pos, k_neg
        d_shifted = d_q_pos * q_pos + d_k_pos * k_pos - d_k_neg * k_neg

        # shifted = cum_g - shift(cum_g).  Treat shift as constant for
        # gradient purposes — the midpoint correction is a second-order
        # effect that doesn't materially affect training.
        d_cum_g = d_cum_g + d_shifted

        # u_rhs = v * beta  →  d_v, d_beta
        d_v_c = d_u_rhs * beta_c[:, None]                    # [C, V]
        d_beta_u = tl.sum(d_u_rhs * v_c, axis=1)            # [C]

        # Totals
        d_k_c = d_k_state + d_k_from_pos + d_k_from_neg     # [C, K]
        d_beta_c = d_beta_L + d_beta_w + d_beta_u            # [C]

        # cum_g = cumsum(g, axis=0) [C, K] → reverse cumsum per key
        # total_g = sum(g, axis=0) [K] → broadcast
        total_d_cum = tl.sum(d_cum_g, axis=0)                # [K]
        fwd_cumsum_d = tl.cumsum(d_cum_g, axis=0)            # [C, K]
        d_g_c = total_d_cum[None, :] - fwd_cumsum_d + d_cum_g + d_total_g[None, :]   # [C, K]

        # ── Store gradients ───────────────────────────────────────────────
        tl.store(DQ + b * s_dqb + (t0 + c_offs[:, None]) * s_dqt
                 + h * s_dqh + k_offs[None, :] * s_dqk, d_q_c)
        tl.store(DK + b * s_dkb + (t0 + c_offs[:, None]) * s_dkt
                 + h * s_dkh + k_offs[None, :] * s_dkk, d_k_c)
        tl.store(DV + b * s_dvb + (t0 + c_offs[:, None]) * s_dvt
                 + h * s_dvh + v_offs[None, :] * s_dvv, d_v_c)
        tl.store(DG + b * s_dgb + (t0 + c_offs[:, None]) * s_dgt
                 + h * s_dgh + k_offs[None, :] * s_dgk, d_g_c)
        tl.store(DBETA + b * s_dbb + (t0 + c_offs) * s_dbt
                 + h * s_dbh, d_beta_c)


# ═══════════════════════════════════════════════════════════════════════════
# Python wrappers + torch.library custom ops
# ═══════════════════════════════════════════════════════════════════════════

CS = 32  # chunk size (smaller = faster forward substitution, more chunks)


@torch.library.custom_op("kdat::fwd", mutates_args=())
def _kdat_fwd(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
) -> tuple[Tensor, Tensor]:
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert T % CS == 0, f"seq_len must be divisible by {CS}, got {T}"
    NC = T // CS
    y = torch.empty(B, T, H, V, device=q.device, dtype=torch.float32)
    states = torch.empty(B, H, NC + 1, K, V, device=q.device, dtype=torch.float32)
    grid = (B * H,)
    _kda_chunk_fwd[grid](
        q, k, v, g, beta, y, states,
        *q.stride(), *k.stride(), *v.stride(), *g.stride(), *beta.stride(),
        *y.stride(), *states.stride(),
        H, NC, CHUNK=CS, K_DIM=K, V_DIM=V,
    )
    return y, states


@torch.library.register_fake("kdat::fwd")
def _kdat_fwd_fake(q, k, v, g, beta):
    B, T, H, K = q.shape
    V = v.shape[-1]
    NC = T // CS
    y = torch.empty(B, T, H, V, device=q.device, dtype=torch.float32)
    states = torch.empty(B, H, NC + 1, K, V, device=q.device, dtype=torch.float32)
    return y, states


@torch.library.custom_op("kdat::bwd", mutates_args=())
def _kdat_bwd(
    dy: Tensor, q: Tensor, k: Tensor, v: Tensor,
    g: Tensor, beta: Tensor, states: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    B, T, H, K = q.shape
    V = v.shape[-1]
    NC = T // CS
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.empty_like(k, dtype=torch.float32)
    dv = torch.empty_like(v, dtype=torch.float32)
    dg = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
    dbeta = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
    grid = (B * H,)
    _kda_chunk_bwd[grid](
        q, k, v, g, beta, states, dy,
        dq, dk, dv, dg, dbeta,
        *q.stride(), *k.stride(), *v.stride(), *g.stride(), *beta.stride(),
        *states.stride(), *dy.stride(),
        *dq.stride(), *dk.stride(), *dv.stride(), *dg.stride(), *dbeta.stride(),
        H, NC, CHUNK=CS, K_DIM=K, V_DIM=V,
    )
    return dq, dk, dv, dg, dbeta


@torch.library.register_fake("kdat::bwd")
def _kdat_bwd_fake(dy, q, k, v, g, beta, states):
    B, T, H, K = q.shape
    V = v.shape[-1]
    return (
        torch.empty(B, T, H, K, device=q.device, dtype=torch.float32),
        torch.empty(B, T, H, K, device=q.device, dtype=torch.float32),
        torch.empty(B, T, H, V, device=q.device, dtype=torch.float32),
        torch.empty(B, T, H, K, device=q.device, dtype=torch.float32),
        torch.empty(B, T, H, device=q.device, dtype=torch.float32),
    )


def _kdat_autograd_bwd(ctx, do, d_states):
    q, k, v, g, beta, states = ctx.saved_tensors
    dq, dk, dv, dg, dbeta = torch.ops.kdat.bwd(
        do.contiguous(), q, k, v, g, beta, states,
    )
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dg.to(g.dtype), dbeta.to(beta.dtype)


def _kdat_autograd_setup(ctx, inputs, output):
    q, k, v, g, beta = inputs
    y, states = output
    ctx.save_for_backward(q, k, v, g, beta, states)


torch.library.register_autograd(
    "kdat::fwd", _kdat_autograd_bwd, setup_context=_kdat_autograd_setup,
)


def kda_triton_autograd(q, k, v, g, beta, scale=None):
    """KDA with chunked Triton fwd+bwd via torch.library custom ops.

    Args:
        q: [B, T, H, K] queries
        k: [B, T, H, K] keys
        v: [B, T, H, V] values
        g: [B, T, H, K] per-key gating in log space (negative values = decay)
        beta: [B, T, H] scalar mixing coefficient
        scale: float, default K^(-0.5)

    Returns:
        y: [B, T, H, V] output
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    y, _states = torch.ops.kdat.fwd(q * scale, k, v, g, beta)
    return y
