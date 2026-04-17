"""Gated Delta Rule (GDR) — Chunked Triton forward + backward kernels.

Chunked algorithm: O(T/C) sequential steps instead of O(T) for recurrent.
Within each chunk of C=64 timesteps, computations are parallel matrix ops:

  1. Compute beta * K @ K^T with causal decay mask → lower-triangular L
  2. Solve (I + L)^{-1} via column-oriented forward substitution
  3. Compute effective v, w via matmul with the solved matrix
  4. Propagate state across chunks (sequential, T/C steps)
  5. Compute output via intra-chunk attention + state readout

Reference: https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule

Usage:
    from triton_kernels import gdr_triton_autograd
    y = gdr_triton_autograd(q, k, v, g, beta)
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
        mask_j = (c_offs == j).to(tl.float32)          # [C]
        x_j = tl.sum(x * mask_j[:, None], axis=0)      # [D]
        A_col = tl.sum(A * mask_j[None, :], axis=1)     # [C]
        x = x - A_col[:, None] * x_j[None, :]
    return x


# ═══════════════════════════════════════════════════════════════════════════
# Forward Kernel
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _gdr_chunk_fwd(
    Q, K, V, G, BETA, Y, STATES,
    s_qb, s_qt, s_qh, s_qk,
    s_kb, s_kt, s_kh, s_kk,
    s_vb, s_vt, s_vh, s_vv,
    s_gb, s_gt, s_gh,
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
        g_c = tl.load(G + b * s_gb + (t0 + c_offs) * s_gt
                       + h * s_gh).to(tl.float32)
        beta_c = tl.load(BETA + b * s_bb + (t0 + c_offs) * s_bt
                          + h * s_bh).to(tl.float32)

        # Cumulative decay  ─  g is in log-space (negative), clamp for stability
        cum_g = tl.cumsum(g_c, axis=0)                     # [C]

        exp_cum = tl.exp(cum_g)                             # [C]

        # Decay matrix  [C, C] — mask before exp to avoid inf in upper triangle
        causal = c_offs[:, None] >= c_offs[None, :]
        strict_lower = c_offs[:, None] > c_offs[None, :]
        decay_diff = cum_g[:, None] - cum_g[None, :]
        decay_diff = tl.where(causal, decay_diff, -float('inf'))
        exp_decay = tl.exp(decay_diff)                      # upper tri = 0, not inf

        # Lower-triangular coupling:  L[i,j] = kkt[i,j] * decay * beta  (i>j)
        kkt = tl.dot(k_c, tl.trans(k_c))                   # [C, C]
        L = tl.where(strict_lower, kkt * exp_decay * beta_c[:, None], 0.0)

        # Solve (I+L)^{-1} applied to two RHS vectors
        u = _tri_solve(L, v_c * beta_c[:, None], c_offs, CHUNK, UPPER=0)             # [C, V]
        w = _tri_solve(L, k_c * beta_c[:, None] * exp_cum[:, None], c_offs, CHUNK, UPPER=0)  # [C, K]

        # Effective delta values using state from previous chunks
        v_new = u - tl.dot(w, state)                        # [C, V]

        # Query with per-position decay
        q_decay = q_c * exp_cum[:, None]                    # [C, K]

        # Inter-chunk contribution
        y_state = tl.dot(q_decay, state)                    # [C, V]

        # Intra-chunk causal attention
        qk = tl.dot(q_c, tl.trans(k_c))                    # [C, C]
        attn = qk * exp_decay                               # already 0 in upper tri
        y_intra = tl.dot(attn, v_new)                       # [C, V]

        # Store output
        y_chunk = y_state + y_intra
        y_ptrs = (Y + b * s_yb + (t0 + c_offs[:, None]) * s_yt
                  + h * s_yh + v_offs[None, :] * s_yv)
        tl.store(y_ptrs, y_chunk)

        # State update
        total_g = tl.sum(g_c, axis=0)
        dte = tl.exp(total_g - cum_g)                       # [C]
        state = (state * tl.exp(total_g)
                 + tl.dot(tl.trans(k_c * dte[:, None]), v_new))

    # Store final state
    st_ptrs = (STATES + b * s_sb + h * s_sh + num_chunks * s_sc
               + k_offs[:, None] * s_sk + v_offs[None, :] * s_sv)
    tl.store(st_ptrs, state)


# ═══════════════════════════════════════════════════════════════════════════
# Backward Kernel
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _gdr_chunk_bwd(
    Q, K, V, G, BETA, STATES, DY,
    DQ, DK, DV, DG, DBETA,
    s_qb, s_qt, s_qh, s_qk,
    s_kb, s_kt, s_kh, s_kk,
    s_vb, s_vt, s_vh, s_vv,
    s_gb, s_gt, s_gh,
    s_bb, s_bt, s_bh,
    s_sb, s_sh, s_sc, s_sk, s_sv,
    s_dyb, s_dyt, s_dyh, s_dyv,
    s_dqb, s_dqt, s_dqh, s_dqk,
    s_dkb, s_dkt, s_dkh, s_dkk,
    s_dvb, s_dvt, s_dvh, s_dvv,
    s_dgb, s_dgt, s_dgh,
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
        g_c = tl.load(G + b * s_gb + (t0 + c_offs) * s_gt
                       + h * s_gh).to(tl.float32)
        beta_c = tl.load(BETA + b * s_bb + (t0 + c_offs) * s_bt
                          + h * s_bh).to(tl.float32)
        dy = tl.load(DY + b * s_dyb + (t0 + c_offs[:, None]) * s_dyt
                      + h * s_dyh + v_offs[None, :] * s_dyv).to(tl.float32)

        # ── Recompute forward quantities ──────────────────────────────────
        cum_g = tl.cumsum(g_c, axis=0)
        exp_cum = tl.exp(cum_g)

        causal = c_offs[:, None] >= c_offs[None, :]
        strict_lower = c_offs[:, None] > c_offs[None, :]
        decay_diff = cum_g[:, None] - cum_g[None, :]
        decay_diff = tl.where(causal, decay_diff, -float('inf'))
        exp_decay = tl.exp(decay_diff)                      # upper tri = 0

        kkt = tl.dot(k_c, tl.trans(k_c))
        L = tl.where(strict_lower, kkt * exp_decay * beta_c[:, None], 0.0)

        u = _tri_solve(L, v_c * beta_c[:, None], c_offs, CHUNK, UPPER=0)
        w = _tri_solve(L, k_c * beta_c[:, None] * exp_cum[:, None], c_offs, CHUNK, UPPER=0)

        v_new = u - tl.dot(w, state)
        q_decay = q_c * exp_cum[:, None]

        qk = tl.dot(q_c, tl.trans(k_c))
        attn = qk * exp_decay                               # already 0 in upper tri

        total_g = tl.sum(g_c, axis=0)
        dte = tl.exp(total_g - cum_g)

        # ── Backward ──────────────────────────────────────────────────────

        # y = y_state + y_intra
        # y_state = q_decay @ state  [C,V]
        d_q_decay = tl.dot(dy, tl.trans(state))             # [C, K]
        d_state_y = tl.dot(tl.trans(q_decay), dy)           # [K, V]

        # y_intra = attn @ v_new  [C,V]
        d_attn = tl.dot(dy, tl.trans(v_new))                # [C, C]
        d_v_new = tl.dot(tl.trans(attn), dy)                # [C, V]

        # state_new = state*exp(total_g) + (k*dte)^T @ v_new
        d_v_new = d_v_new + tl.dot(k_c * dte[:, None], d_state)   # [C, V]
        d_k_dte_mat = tl.dot(v_new, tl.trans(d_state))            # [C, K]
        d_dte = tl.sum(d_k_dte_mat * k_c, axis=1)                 # [C]
        d_k_state = d_k_dte_mat * dte[:, None]                    # [C, K]
        d_total_g_s = tl.sum(d_state * state) * tl.exp(total_g)
        d_total_g_dte = tl.sum(d_dte * dte)
        d_cum_g = -d_dte * dte                                     # [C]

        # v_new = u - w @ state
        d_u = d_v_new                                              # [C, V]
        d_w = -tl.dot(d_v_new, tl.trans(state))                   # [C, K]
        d_state_w = -tl.dot(tl.trans(w), d_v_new)                 # [K, V]

        # Propagate d_state to previous chunk
        d_state = d_state * tl.exp(total_g) + d_state_y + d_state_w

        # u = (I+L)^{-1} u_rhs,  w = (I+L)^{-1} w_rhs
        # Grad through inverse:  d_rhs = (I+L)^{-T} d_x
        LT = tl.trans(L)
        d_u_rhs = _tri_solve(LT, d_u, c_offs, CHUNK, UPPER=1)    # [C, V]
        d_w_rhs = _tri_solve(LT, d_w, c_offs, CHUNK, UPPER=1)    # [C, K]

        # d_L = -(d_u_rhs @ u^T + d_w_rhs @ w^T), strict lower
        d_L = -(tl.dot(d_u_rhs, tl.trans(u)) + tl.dot(d_w_rhs, tl.trans(w)))
        d_L = tl.where(strict_lower, d_L, 0.0)

        # L = kkt * exp_decay * beta[:, None]  (strict lower)
        d_kkt = tl.where(strict_lower, d_L * exp_decay * beta_c[:, None], 0.0)
        d_exp_decay_L = tl.where(strict_lower, d_L * kkt * beta_c[:, None], 0.0)
        d_beta_L = tl.sum(tl.where(strict_lower, d_L * kkt * exp_decay, 0.0), axis=1)

        # kkt = k @ k^T
        d_k_kkt = tl.dot(d_kkt + tl.trans(d_kkt), k_c)           # [C, K]

        # attn = qk * exp_decay (causal)
        d_qk = tl.where(causal, d_attn * exp_decay, 0.0)
        d_exp_decay_a = tl.where(causal, d_attn * qk, 0.0)

        # exp_decay = exp(decay_diff)  →  d_decay_diff
        d_decay_diff = (d_exp_decay_L + d_exp_decay_a) * exp_decay
        d_cum_g = d_cum_g + tl.sum(d_decay_diff, axis=1) - tl.sum(d_decay_diff, axis=0)

        # q_decay = q_c * exp_cum[:, None]
        d_q_c = d_q_decay * exp_cum[:, None]
        d_cum_g = d_cum_g + tl.sum(d_q_decay * q_c * exp_cum[:, None], axis=1)

        # w_rhs = k_c * beta * exp_cum
        d_k_w = d_w_rhs * beta_c[:, None] * exp_cum[:, None]
        d_beta_w = tl.sum(d_w_rhs * k_c * exp_cum[:, None], axis=1)
        d_cum_g = d_cum_g + tl.sum(d_w_rhs * k_c * beta_c[:, None] * exp_cum[:, None], axis=1)

        # u_rhs = v_c * beta
        d_v_c = d_u_rhs * beta_c[:, None]
        d_beta_u = tl.sum(d_u_rhs * v_c, axis=1)

        # qk = q_c @ k_c^T
        d_q_c = d_q_c + tl.dot(d_qk, k_c)
        d_k_qk = tl.dot(tl.trans(d_qk), q_c)

        # Accumulate per-input gradients
        d_k_c = d_k_state + d_k_kkt + d_k_w + d_k_qk
        d_beta_c = d_beta_L + d_beta_w + d_beta_u

        # cum_g = cumsum(g)  →  reverse cumsum
        # total_g = sum(g)   →  broadcast
        d_total_g = d_total_g_s + d_total_g_dte
        total_d_cum = tl.sum(d_cum_g, axis=0)
        fwd_cumsum_d = tl.cumsum(d_cum_g, axis=0)
        d_g_c = total_d_cum - fwd_cumsum_d + d_cum_g + d_total_g

        # ── Store gradients ───────────────────────────────────────────────
        tl.store(DQ + b * s_dqb + (t0 + c_offs[:, None]) * s_dqt
                 + h * s_dqh + k_offs[None, :] * s_dqk, d_q_c)
        tl.store(DK + b * s_dkb + (t0 + c_offs[:, None]) * s_dkt
                 + h * s_dkh + k_offs[None, :] * s_dkk, d_k_c)
        tl.store(DV + b * s_dvb + (t0 + c_offs[:, None]) * s_dvt
                 + h * s_dvh + v_offs[None, :] * s_dvv, d_v_c)
        tl.store(DG + b * s_dgb + (t0 + c_offs) * s_dgt
                 + h * s_dgh, d_g_c)
        tl.store(DBETA + b * s_dbb + (t0 + c_offs) * s_dbt
                 + h * s_dbh, d_beta_c)


# ═══════════════════════════════════════════════════════════════════════════
# Python wrappers + torch.library custom ops
# ═══════════════════════════════════════════════════════════════════════════

CS = 32  # chunk size (smaller = faster forward substitution, more chunks)


@torch.library.custom_op("gdrt::fwd", mutates_args=())
def _gdrt_fwd(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
) -> tuple[Tensor, Tensor]:
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert T % CS == 0, f"seq_len must be divisible by {CS}, got {T}"
    NC = T // CS
    y = torch.empty(B, T, H, V, device=q.device, dtype=torch.float32)
    states = torch.empty(B, H, NC + 1, K, V, device=q.device, dtype=torch.float32)
    grid = (B * H,)
    _gdr_chunk_fwd[grid](
        q, k, v, g, beta, y, states,
        *q.stride(), *k.stride(), *v.stride(), *g.stride(), *beta.stride(),
        *y.stride(), *states.stride(),
        H, NC, CHUNK=CS, K_DIM=K, V_DIM=V,
    )
    return y, states


@torch.library.register_fake("gdrt::fwd")
def _gdrt_fwd_fake(q, k, v, g, beta):
    B, T, H, K = q.shape
    V = v.shape[-1]
    NC = T // CS
    y = torch.empty(B, T, H, V, device=q.device, dtype=torch.float32)
    states = torch.empty(B, H, NC + 1, K, V, device=q.device, dtype=torch.float32)
    return y, states


@torch.library.custom_op("gdrt::bwd", mutates_args=())
def _gdrt_bwd(
    dy: Tensor, q: Tensor, k: Tensor, v: Tensor,
    g: Tensor, beta: Tensor, states: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    B, T, H, K = q.shape
    V = v.shape[-1]
    NC = T // CS
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.empty_like(k, dtype=torch.float32)
    dv = torch.empty_like(v, dtype=torch.float32)
    dg = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
    dbeta = torch.empty(B, T, H, device=q.device, dtype=torch.float32)
    grid = (B * H,)
    _gdr_chunk_bwd[grid](
        q, k, v, g, beta, states, dy,
        dq, dk, dv, dg, dbeta,
        *q.stride(), *k.stride(), *v.stride(), *g.stride(), *beta.stride(),
        *states.stride(), *dy.stride(),
        *dq.stride(), *dk.stride(), *dv.stride(), *dg.stride(), *dbeta.stride(),
        H, NC, CHUNK=CS, K_DIM=K, V_DIM=V,
    )
    return dq, dk, dv, dg, dbeta


@torch.library.register_fake("gdrt::bwd")
def _gdrt_bwd_fake(dy, q, k, v, g, beta, states):
    B, T, H, K = q.shape
    V = v.shape[-1]
    return (
        torch.empty(B, T, H, K, device=q.device, dtype=torch.float32),
        torch.empty(B, T, H, K, device=q.device, dtype=torch.float32),
        torch.empty(B, T, H, V, device=q.device, dtype=torch.float32),
        torch.empty(B, T, H, device=q.device, dtype=torch.float32),
        torch.empty(B, T, H, device=q.device, dtype=torch.float32),
    )


def _gdrt_autograd_bwd(ctx, do, d_states):
    q, k, v, g, beta, states = ctx.saved_tensors
    dq, dk, dv, dg, dbeta = torch.ops.gdrt.bwd(
        do.contiguous(), q, k, v, g, beta, states,
    )
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dg.to(g.dtype), dbeta.to(beta.dtype)


def _gdrt_autograd_setup(ctx, inputs, output):
    q, k, v, g, beta = inputs
    y, states = output
    ctx.save_for_backward(q, k, v, g, beta, states)


torch.library.register_autograd(
    "gdrt::fwd", _gdrt_autograd_bwd, setup_context=_gdrt_autograd_setup,
)


def gdr_triton_autograd(q, k, v, g, beta, scale=None):
    """GDR with chunked Triton fwd+bwd via torch.library custom ops.

    Args:
        q: [B, T, H, K] queries
        k: [B, T, H, K] keys
        v: [B, T, H, V] values
        g: [B, T, H] scalar gating in log space (negative values = decay)
        beta: [B, T, H] scalar mixing coefficient
        scale: float, default K^(-0.5)

    Returns:
        y: [B, T, H, V] output
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    # Pre-scale q; autograd handles the chain rule through the multiply.
    y, _states = torch.ops.gdrt.fwd(q * scale, k, v, g, beta)
    return y
