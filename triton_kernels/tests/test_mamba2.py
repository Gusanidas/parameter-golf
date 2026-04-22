"""Correctness checks for mamba2_ssd_triton_autograd and mamba2_fused_triton_autograd.

Compares forward + all gradients (dx, dA_log, dB, dC, ddt, and conv weight
gradients for the fused variant) against the recurrent PyTorch reference.
Runs on CUDA; fp32 inputs.

Run:
    python -m triton_kernels.tests.test_mamba2
"""
from __future__ import annotations

import sys

import torch

from triton_kernels.mamba2_ssd import mamba2_ssd_triton_autograd
from triton_kernels.mamba2_fused import mamba2_fused_triton_autograd
from triton_kernels.tests.reference import mamba2_ssd_ref, mamba2_fused_ref


def _max_err(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = (a.float() - b.float()).abs()
    rel = d / b.float().abs().clamp(min=1e-6)
    return d.max().item(), rel.max().item()


def _check(name: str, got: torch.Tensor, ref: torch.Tensor,
           atol: float, rtol: float) -> bool:
    abs_err, rel_err = _max_err(got, ref)
    ok = torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol)
    flag = "PASS" if ok else "FAIL"
    print(f"  {flag} {name:<10s} max_abs={abs_err:.3e}  max_rel={rel_err:.3e}"
          f"   (atol={atol:.0e}, rtol={rtol:.0e})")
    return ok


def _random_inputs(B, T, H, P, N, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    def mkp(*shape, requires_grad=True, scale=1.0):
        t = torch.randn(*shape, device=device, dtype=torch.float32,
                        generator=gen) * scale
        if requires_grad: t = t.detach().requires_grad_()
        return t
    x = mkp(B, T, H, P)
    Bm = mkp(B, T, H, N)
    Cm = mkp(B, T, H, N)
    # dt stays positive (softplus-like). Keep small so A*dt doesn't hit the
    # -20 clamp, which would otherwise zero some gradients.
    dt = (torch.rand(B, T, H, device=device, dtype=torch.float32,
                     generator=gen) * 0.5 + 0.1).detach().requires_grad_()
    # A_log: log of decay magnitude. Range matches training script.
    A_log = (torch.linspace(-3.0, -1.0, H, device=device, dtype=torch.float32)
             .detach().requires_grad_())
    return x, Bm, Cm, dt, A_log


def run_ssd(B=2, T=64, H=2, P=32, N=16, seed=0,
            atol=1e-3, rtol=1e-3) -> bool:
    device = "cuda"
    x, Bm, Cm, dt, A_log = _random_inputs(B, T, H, P, N, seed, device)

    y = mamba2_ssd_triton_autograd(x, A_log, Bm, Cm, dt)
    y.float().sum().backward()

    x2, Bm2, Cm2, dt2, A_log2 = (t.detach().clone().requires_grad_()
                                  for t in (x, Bm, Cm, dt, A_log))
    y_ref = mamba2_ssd_ref(x2, A_log2, Bm2, Cm2, dt2)
    y_ref.sum().backward()

    print(f"[SSD] B={B} T={T} H={H} P={P} N={N} seed={seed}")
    ok = True
    ok &= _check("y",     y,          y_ref,        atol, rtol)
    ok &= _check("dx",    x.grad,     x2.grad,      atol, rtol)
    ok &= _check("dB",    Bm.grad,    Bm2.grad,     atol, rtol)
    ok &= _check("dC",    Cm.grad,    Cm2.grad,     atol, rtol)
    ok &= _check("ddt",   dt.grad,    dt2.grad,     atol, rtol)
    ok &= _check("dA_log", A_log.grad, A_log2.grad, atol, rtol)
    return ok


def run_fused(B=2, T=64, H=2, P=32, N=16, seed=0,
              atol=1e-3, rtol=1e-3) -> bool:
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed + 1000)
    x, Bm, Cm, dt, A_log = _random_inputs(B, T, H, P, N, seed, device)
    wx = (torch.randn(H, P, 4, device=device, dtype=torch.float32,
                      generator=gen) * 0.1).detach().requires_grad_()
    wb = (torch.randn(H, N, 4, device=device, dtype=torch.float32,
                      generator=gen) * 0.1).detach().requires_grad_()
    wc = (torch.randn(H, N, 4, device=device, dtype=torch.float32,
                      generator=gen) * 0.1).detach().requires_grad_()

    y = mamba2_fused_triton_autograd(x, A_log, Bm, Cm, dt, wx, wb, wc)
    y.float().sum().backward()

    x2, Bm2, Cm2, dt2, A_log2, wx2, wb2, wc2 = (
        t.detach().clone().requires_grad_()
        for t in (x, Bm, Cm, dt, A_log, wx, wb, wc))
    y_ref = mamba2_fused_ref(x2, A_log2, Bm2, Cm2, dt2, wx2, wb2, wc2)
    y_ref.sum().backward()

    print(f"[FUSED] B={B} T={T} H={H} P={P} N={N} seed={seed}")
    ok = True
    ok &= _check("y",      y,          y_ref,        atol, rtol)
    ok &= _check("dx",     x.grad,     x2.grad,      atol, rtol)
    ok &= _check("dB",     Bm.grad,    Bm2.grad,     atol, rtol)
    ok &= _check("dC",     Cm.grad,    Cm2.grad,     atol, rtol)
    ok &= _check("ddt",    dt.grad,    dt2.grad,     atol, rtol)
    ok &= _check("dA_log", A_log.grad, A_log2.grad,  atol, rtol)
    ok &= _check("dconv_x", wx.grad,   wx2.grad,     atol, rtol)
    ok &= _check("dconv_b", wb.grad,   wb2.grad,     atol, rtol)
    ok &= _check("dconv_c", wc.grad,   wc2.grad,     atol, rtol)
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required"); return 2
    all_ok = True
    all_ok &= run_ssd(B=1, T=64,  H=1, P=32, N=16, seed=0)
    all_ok &= run_ssd(B=2, T=128, H=2, P=64, N=16, seed=1)
    all_ok &= run_fused(B=1, T=64,  H=1, P=32, N=16, seed=0)
    all_ok &= run_fused(B=2, T=128, H=2, P=64, N=16, seed=1)
    print("\n", "ALL PASS" if all_ok else "FAILURES", sep="")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
