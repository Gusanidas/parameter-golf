"""PyTorch reference implementations for the Triton kernels under review.

Slow (per-timestep loop) but transparently correct. Used by test_kda.py and
test_mamba2.py to check forward outputs and gradients against the kernels.

Recurrences implemented:

- causal_conv1d_ref:  y[t] = sum_{k=0..3} w[:, :, k] * x[t - k]  (zero-padded)
- mamba2_ssd_ref:     state_t = exp(-exp(A_log) * dt_t) * state_{t-1} + B_t * x_t
                      y_t     = C_t @ state_t
- mamba2_fused_ref:   mamba2_ssd on (SiLU(conv(x)), SiLU(conv(B)), SiLU(conv(C)))
- kda_ref:            D_t      = exp(g_t)                        [K]
                      pre_t    = D_t[:, None] * state_{t-1}
                      v_new_t  = v_t - k_t @ pre_t                [V]
                      state_t  = pre_t + beta_t * outer(k_t, v_new_t)
                      y_t      = (q_t * scale) @ state_t
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def causal_conv1d_ref(x: Tensor, w: Tensor) -> Tensor:
    """Depthwise causal conv1d, kernel_size=4, zero-padded on the left.

    x: [B, T, H, D]   w: [H, D, 4]   returns [B, T, H, D] fp32.
    w[:, :, 0] is the tap at t; w[:, :, k] is the tap at t-k.
    """
    B_, T_, H_, D_ = x.shape
    xf = x.float()
    out = torch.zeros_like(xf)
    for k in range(4):
        if k == 0:
            shifted = xf
        else:
            shifted = F.pad(xf, (0, 0, 0, 0, k, 0))[:, :T_]
        out = out + shifted * w[None, None, :, :, k]
    return out


def conv_silu_ref(x: Tensor, w: Tensor) -> Tensor:
    c = causal_conv1d_ref(x, w)
    return c * torch.sigmoid(c)


def mamba2_ssd_ref(
    x: Tensor, A_log: Tensor, B: Tensor, C: Tensor, dt: Tensor,
) -> Tensor:
    """Recurrent Mamba2 SSD reference, matching kernel's A*dt ≥ -20 clamp.

    x: [Bsz, T, H, P]   A_log: [H]
    B, C: [Bsz, T, H, N]   dt: [Bsz, T, H]
    Returns y: [Bsz, T, H, P] fp32.
    """
    Bsz, T, H, P = x.shape
    N = B.shape[-1]
    A = -torch.exp(A_log.float())
    xf, Bf, Cf, dtf = x.float(), B.float(), C.float(), dt.float()
    y = torch.zeros_like(xf)
    for bi in range(Bsz):
        for h in range(H):
            state = torch.zeros(N, P, device=x.device, dtype=torch.float32)
            for t in range(T):
                log_a = torch.clamp(A[h] * dtf[bi, t, h], min=-20.0)
                state = torch.exp(log_a) * state \
                    + torch.outer(Bf[bi, t, h], xf[bi, t, h])
                y[bi, t, h] = Cf[bi, t, h] @ state
    return y


def mamba2_fused_ref(
    x: Tensor, A_log: Tensor, B: Tensor, C: Tensor, dt: Tensor,
    conv_w_x: Tensor, conv_w_b: Tensor, conv_w_c: Tensor,
    D_skip: Tensor | None = None,
) -> Tensor:
    x_c = conv_silu_ref(x, conv_w_x)
    B_c = conv_silu_ref(B, conv_w_b)
    C_c = conv_silu_ref(C, conv_w_c)
    y = mamba2_ssd_ref(x_c, A_log, B_c, C_c, dt)
    if D_skip is not None:
        y = y + x_c * D_skip.float()[None, None, :, None]
    return y


def kda_ref(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
    scale: float | None = None,
) -> Tensor:
    """Recurrent KDA reference (gated delta rule with per-key decay).

    q, k, g: [B, T, H, K]   v: [B, T, H, V]   beta: [B, T, H]
    Returns y: [B, T, H, V] fp32.
    """
    B_, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    qf = q.float() * scale
    kf, vf, gf, bf = k.float(), v.float(), g.float(), beta.float()
    y = torch.zeros(B_, T, H, V, device=q.device, dtype=torch.float32)
    for bi in range(B_):
        for h in range(H):
            state = torch.zeros(K, V, device=q.device, dtype=torch.float32)
            for t in range(T):
                state = torch.exp(gf[bi, t, h])[:, None] * state
                k_t, v_t = kf[bi, t, h], vf[bi, t, h]
                v_new = v_t - k_t @ state
                state = state + bf[bi, t, h] * torch.outer(k_t, v_new)
                y[bi, t, h] = qf[bi, t, h] @ state
    return y
