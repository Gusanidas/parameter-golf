"""GDN — Recurrent gated delta rule Triton kernels.

One (batch, head) per block, sequential over T. State is a [Dk, Dv] matrix
updated via rank-1 delta corrections.
"""
from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl


@triton.jit
def _gdn_recurrent_fwd(
    Q, K, V, G, BETA, Y, ALL_STATES,
    s_qb, s_qt, s_qh, s_qd,
    s_kb, s_kt, s_kh, s_kd,
    s_vb, s_vt, s_vh, s_vd,
    s_gb, s_gt, s_gh,
    s_bb_, s_bt_, s_bh_,
    s_yb, s_yt, s_yh, s_yd,
    s_asb, s_ash, s_ast, s_ask, s_asv,
    nheads, seqlen,
    DK: tl.constexpr,
    DV: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // nheads
    h = pid % nheads

    dk_offs = tl.arange(0, DK)
    dv_offs = tl.arange(0, DV)

    state = tl.zeros((DK, DV), dtype=tl.float32)

    for t in range(seqlen):
        as_ptrs = ALL_STATES + b * s_asb + h * s_ash + t * s_ast + dk_offs[:, None] * s_ask + dv_offs[None, :] * s_asv
        tl.store(as_ptrs, state)

        q_t = tl.load(Q + b * s_qb + t * s_qt + h * s_qh + dk_offs * s_qd).to(tl.float32)
        k_t = tl.load(K + b * s_kb + t * s_kt + h * s_kh + dk_offs * s_kd).to(tl.float32)
        v_t = tl.load(V + b * s_vb + t * s_vt + h * s_vh + dv_offs * s_vd).to(tl.float32)
        g_t = tl.load(G + b * s_gb + t * s_gt + h * s_gh).to(tl.float32)
        beta_t = tl.load(BETA + b * s_bb_ + t * s_bt_ + h * s_bh_).to(tl.float32)

        state = state * tl.exp(g_t)
        retrieved = tl.sum(state * k_t[:, None], axis=0)
        delta = beta_t * (v_t - retrieved)
        state = state + k_t[:, None] * delta[None, :]
        y_t = tl.sum(state * q_t[:, None], axis=0)

        y_ptrs = Y + b * s_yb + t * s_yt + h * s_yh + dv_offs * s_yd
        tl.store(y_ptrs, y_t)

    as_ptrs = ALL_STATES + b * s_asb + h * s_ash + seqlen * s_ast + dk_offs[:, None] * s_ask + dv_offs[None, :] * s_asv
    tl.store(as_ptrs, state)


@triton.jit
def _gdn_recurrent_bwd(
    Q, K, V, G, BETA, DY, ALL_STATES,
    DQ, DK_OUT, DV_OUT, DG, DBETA,
    s_qb, s_qt, s_qh, s_qd,
    s_kb, s_kt, s_kh, s_kd,
    s_vb, s_vt, s_vh, s_vd,
    s_gb, s_gt, s_gh,
    s_bb_, s_bt_, s_bh_,
    s_dyb, s_dyt, s_dyh, s_dyd,
    s_asb, s_ash, s_ast, s_ask, s_asv,
    s_dqb, s_dqt, s_dqh, s_dqd,
    s_dkb, s_dkt, s_dkh, s_dkd,
    s_dvb, s_dvt, s_dvh, s_dvd,
    s_dgb, s_dgt, s_dgh,
    s_dbb, s_dbt, s_dbh,
    nheads, seqlen,
    DK: tl.constexpr,
    DV: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // nheads
    h = pid % nheads

    dk_offs = tl.arange(0, DK)
    dv_offs = tl.arange(0, DV)

    d_state = tl.zeros((DK, DV), dtype=tl.float32)

    for t_rev in range(seqlen):
        t = seqlen - 1 - t_rev

        as_ptrs = ALL_STATES + b * s_asb + h * s_ash + t * s_ast + dk_offs[:, None] * s_ask + dv_offs[None, :] * s_asv
        state_pre = tl.load(as_ptrs).to(tl.float32)

        q_t = tl.load(Q + b * s_qb + t * s_qt + h * s_qh + dk_offs * s_qd).to(tl.float32)
        k_t = tl.load(K + b * s_kb + t * s_kt + h * s_kh + dk_offs * s_kd).to(tl.float32)
        v_t = tl.load(V + b * s_vb + t * s_vt + h * s_vh + dv_offs * s_vd).to(tl.float32)
        g_t = tl.load(G + b * s_gb + t * s_gt + h * s_gh).to(tl.float32)
        beta_t = tl.load(BETA + b * s_bb_ + t * s_bt_ + h * s_bh_).to(tl.float32)
        dy_t = tl.load(DY + b * s_dyb + t * s_dyt + h * s_dyh + dv_offs * s_dyd).to(tl.float32)

        state_decayed = state_pre * tl.exp(g_t)
        retrieved = tl.sum(state_decayed * k_t[:, None], axis=0)
        delta = beta_t * (v_t - retrieved)
        state_after = state_decayed + k_t[:, None] * delta[None, :]

        d_state = d_state + q_t[:, None] * dy_t[None, :]
        dq_t = tl.sum(state_after * dy_t[None, :], axis=1)

        dk_t = tl.sum(d_state * delta[None, :], axis=1)
        d_delta = tl.sum(d_state * k_t[:, None], axis=0)

        d_v_minus_ret = d_delta * beta_t
        d_beta_t = tl.sum(d_delta * (v_t - retrieved))
        dv_t = d_v_minus_ret
        d_retrieved = -d_v_minus_ret

        d_state = d_state + k_t[:, None] * d_retrieved[None, :]
        dk_t = dk_t + tl.sum(state_decayed * d_retrieved[None, :], axis=1)

        d_g_t = tl.sum(d_state * state_pre) * tl.exp(g_t)
        d_state = d_state * tl.exp(g_t)

        tl.store(DQ + b * s_dqb + t * s_dqt + h * s_dqh + dk_offs * s_dqd, dq_t)
        tl.store(DK_OUT + b * s_dkb + t * s_dkt + h * s_dkh + dk_offs * s_dkd, dk_t)
        tl.store(DV_OUT + b * s_dvb + t * s_dvt + h * s_dvh + dv_offs * s_dvd, dv_t)
        tl.store(DG + b * s_dgb + t * s_dgt + h * s_dgh, d_g_t)
        tl.store(DBETA + b * s_dbb + t * s_dbt + h * s_dbh, d_beta_t)


# ═══════════════════════════════════════════════════════════════════════════
# Python wrappers
# ═══════════════════════════════════════════════════════════════════════════

def gdn_recurrent_triton(q, k, v, g, beta):
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    y = torch.empty(B, T, H, Dv, device=q.device, dtype=torch.float32)
    all_states = torch.empty(B, H, T + 1, Dk, Dv, device=q.device, dtype=torch.float32)

    grid = (B * H,)
    _gdn_recurrent_fwd[grid](
        q, k, v, g, beta, y, all_states,
        *q.stride(), *k.stride(), *v.stride(), *g.stride(), *beta.stride(),
        *y.stride(), *all_states.stride(),
        H, T, DK=Dk, DV=Dv,
    )
    return y.to(q.dtype), all_states


def gdn_recurrent_triton_bwd(dy, q, k, v, g, beta, all_states):
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty(B, T, H, Dv, device=q.device, dtype=q.dtype)
    dg = torch.empty_like(g)
    dbeta = torch.empty_like(beta)

    grid = (B * H,)
    _gdn_recurrent_bwd[grid](
        q, k, v, g, beta, dy, all_states,
        dq, dk, dv, dg, dbeta,
        *q.stride(), *k.stride(), *v.stride(), *g.stride(), *beta.stride(),
        *dy.stride(), *all_states.stride(),
        *dq.stride(), *dk.stride(), *dv.stride(), *dg.stride(), *dbeta.stride(),
        H, T, DK=Dk, DV=Dv,
    )
    return dq, dk, dv, dg, dbeta


class GDNRecurrentTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        y, all_states = gdn_recurrent_triton(q, k, v, g, beta)
        ctx.save_for_backward(q, k, v, g, beta, all_states)
        return y

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, beta, all_states = ctx.saved_tensors
        return gdn_recurrent_triton_bwd(do.contiguous(), q, k, v, g, beta, all_states)


def gdn_recurrent_triton_autograd(q, k, v, g, beta):
    """Triton forward + Triton backward for GDN."""
    return GDNRecurrentTriton.apply(q, k, v, g, beta)
