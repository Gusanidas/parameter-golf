"""Correctness checks for mamba2_ssd_triton_autograd and mamba2_fused_triton_autograd.

Compares forward + all gradients (dx, dA_log, dB, dC, ddt, and conv weight
gradients for the fused variant) against the recurrent PyTorch reference.
Runs on CUDA with random upstream gradients and fullgraph compile coverage.

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


def _random_inputs(B, T, H, P, N, seed, device, dtype=torch.float32):
    gen = torch.Generator(device=device).manual_seed(seed)
    def mkp(*shape, requires_grad=True, scale=1.0):
        t = torch.randn(*shape, device=device, dtype=dtype, generator=gen) * scale
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


def _random_like(t: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=t.device).manual_seed(seed)
    return torch.randn(t.shape, device=t.device, dtype=t.dtype, generator=gen)


def run_ssd(B=2, T=64, H=2, P=32, N=16, seed=0, dtype=torch.float32,
            compiled=False, atol=2e-2, rtol=2e-2) -> bool:
    device = "cuda"
    x, Bm, Cm, dt, A_log = _random_inputs(B, T, H, P, N, seed, device, dtype=dtype)

    fn = mamba2_ssd_triton_autograd
    if compiled:
        fn = torch.compile(fn, fullgraph=True)
    y = fn(x, A_log, Bm, Cm, dt)
    dy = _random_like(y, seed + 10_000)
    y.backward(dy)

    x2, Bm2, Cm2, dt2, A_log2 = (t.detach().clone().requires_grad_()
                                  for t in (x, Bm, Cm, dt, A_log))
    y_ref = mamba2_ssd_ref(x2, A_log2, Bm2, Cm2, dt2)
    y_ref.backward(dy.float())

    mode = "compiled" if compiled else "eager"
    print(f"[SSD] {mode} dtype={dtype} B={B} T={T} H={H} P={P} N={N} seed={seed}")
    ok = True
    ok &= _check("y",     y,          y_ref,        atol, rtol)
    ok &= _check("dx",    x.grad,     x2.grad,      atol, rtol)
    ok &= _check("dB",    Bm.grad,    Bm2.grad,     atol, rtol)
    ok &= _check("dC",    Cm.grad,    Cm2.grad,     atol, rtol)
    ok &= _check("ddt",   dt.grad,    dt2.grad,     atol, rtol)
    ok &= _check("dA_log", A_log.grad, A_log2.grad, atol, rtol)
    return ok


def run_fused(B=2, T=64, H=2, P=32, N=16, seed=0, dtype=torch.float32,
              compiled=False, atol=2e-2, rtol=2e-2) -> bool:
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed + 1000)
    x, Bm, Cm, dt, A_log = _random_inputs(B, T, H, P, N, seed, device, dtype=dtype)
    wx = (torch.randn(H, P, 4, device=device, dtype=dtype,
                      generator=gen) * 0.1).detach().requires_grad_()
    wb = (torch.randn(H, N, 4, device=device, dtype=dtype,
                      generator=gen) * 0.1).detach().requires_grad_()
    wc = (torch.randn(H, N, 4, device=device, dtype=dtype,
                      generator=gen) * 0.1).detach().requires_grad_()

    fn = mamba2_fused_triton_autograd
    if compiled:
        fn = torch.compile(fn, fullgraph=True)
    y = fn(x, A_log, Bm, Cm, dt, wx, wb, wc)
    dy = _random_like(y, seed + 20_000)
    y.backward(dy)

    x2, Bm2, Cm2, dt2, A_log2, wx2, wb2, wc2 = (
        t.detach().clone().requires_grad_()
        for t in (x, Bm, Cm, dt, A_log, wx, wb, wc))
    y_ref = mamba2_fused_ref(x2, A_log2, Bm2, Cm2, dt2, wx2, wb2, wc2)
    y_ref.backward(dy.float())

    mode = "compiled" if compiled else "eager"
    print(f"[FUSED] {mode} dtype={dtype} B={B} T={T} H={H} P={P} N={N} seed={seed}")
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


def run_contract_checks() -> bool:
    device = "cuda"
    x, Bm, Cm, dt, A_log = _random_inputs(1, 65, 1, 32, 16, 123, device)
    ok = True
    for name, fn in (
        ("ssd tail", lambda: mamba2_ssd_triton_autograd(x, A_log, Bm, Cm, dt)),
        ("fused tail", lambda: mamba2_fused_triton_autograd(
            x, A_log, Bm, Cm, dt,
            torch.randn(1, 32, 4, device=device),
            torch.randn(1, 16, 4, device=device),
            torch.randn(1, 16, 4, device=device),
        )),
    ):
        try:
            fn()
        except ValueError:
            print(f"  PASS {name:<10s} rejected unsupported sequence length")
        else:
            print(f"  FAIL {name:<10s} accepted unsupported sequence length")
            ok = False
    try:
        mamba2_fused_triton_autograd(
            x[:, :64], A_log, Bm[:, :64], Cm[:, :64], dt[:, :64],
            torch.randn(1, 32, 4, device=device),
            torch.randn(1, 16, 4, device=device),
            torch.randn(1, 16, 4, device=device),
            chunk_size=128,
        )
    except AssertionError:
        print("  PASS chunk_size rejected unsupported 128")
    else:
        print("  FAIL chunk_size accepted unsupported 128")
        ok = False
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required"); return 2
    all_ok = True
    all_ok &= run_ssd(B=1, T=64,  H=1, P=32, N=16, seed=0)
    all_ok &= run_ssd(B=2, T=128, H=2, P=64, N=16, seed=1, compiled=True)
    all_ok &= run_fused(B=1, T=64,  H=1, P=32, N=16, seed=0)
    all_ok &= run_fused(B=2, T=128, H=2, P=64, N=16, seed=1, compiled=True)
    all_ok &= run_fused(B=2, T=128, H=2, P=64, N=16, seed=2, dtype=torch.bfloat16,
                        atol=3e-2, rtol=3e-2)
    all_ok &= run_contract_checks()
    print("\n", "ALL PASS" if all_ok else "FAILURES", sep="")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
