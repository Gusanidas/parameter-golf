"""Correctness check for kda_triton_autograd.

Compares forward output and all gradients (dq, dk, dv, dg, dbeta) against
the recurrent PyTorch reference in reference.py. Runs on CUDA; fp32 inputs
to keep the signal well above accumulator noise.

Run:
    python -m triton_kernels.tests.test_kda
"""
from __future__ import annotations

import sys

import torch

from triton_kernels.kda import kda_triton_autograd
from triton_kernels.tests.reference import kda_ref


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


def run_case(B=2, T=64, H=2, K=32, V=32, seed=0,
             atol=1e-3, rtol=1e-3) -> bool:
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)

    def mkp(*shape):
        return torch.randn(*shape, device=device, dtype=torch.float32,
                           generator=gen, requires_grad=True)

    q = mkp(B, T, H, K)
    k = mkp(B, T, H, K)
    v = mkp(B, T, H, V)
    # g is a log-space decay; keep it small (|g| <= 0.1) to match realistic
    # training and keep the reference's cumulative exponentials well-scaled.
    g = (torch.randn(B, T, H, K, device=device, dtype=torch.float32,
                     generator=gen) * 0.1).requires_grad_()
    beta = (torch.sigmoid(torch.randn(B, T, H, device=device,
                                      dtype=torch.float32, generator=gen))
            .detach().requires_grad_())

    # --- kernel ---
    y = kda_triton_autograd(q, k, v, g, beta)
    loss = y.float().sum()  # arbitrary scalar loss
    loss.backward()
    dq, dk, dv, dg, dbeta = q.grad, k.grad, v.grad, g.grad, beta.grad

    # --- reference ---
    q2, k2, v2, g2, beta2 = (t.detach().clone().requires_grad_()
                              for t in (q, k, v, g, beta))
    y_ref = kda_ref(q2, k2, v2, g2, beta2)
    loss_ref = y_ref.sum()
    loss_ref.backward()

    print(f"Case B={B} T={T} H={H} K={K} V={V} seed={seed}")
    ok = True
    ok &= _check("y",     y,     y_ref,    atol, rtol)
    ok &= _check("dq",    dq,    q2.grad,  atol, rtol)
    ok &= _check("dk",    dk,    k2.grad,  atol, rtol)
    ok &= _check("dv",    dv,    v2.grad,  atol, rtol)
    ok &= _check("dg",    dg,    g2.grad,  atol, rtol)
    ok &= _check("dbeta", dbeta, beta2.grad, atol, rtol)
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required"); return 2
    all_ok = True
    all_ok &= run_case(B=1, T=32,  H=1, K=32, V=32, seed=0)
    all_ok &= run_case(B=2, T=64,  H=2, K=32, V=32, seed=1)
    all_ok &= run_case(B=2, T=128, H=4, K=64, V=64, seed=2)
    print("\n", "ALL PASS" if all_ok else "FAILURES", sep="")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
