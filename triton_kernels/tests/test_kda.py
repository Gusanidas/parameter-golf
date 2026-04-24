"""Correctness checks for the FLA-backed KDA custom op.

Compares forward output and gradients against FLA's naive recurrent reference.

Run:
    python -m triton_kernels.tests.test_kda
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from triton_kernels.kda import kda_triton_autograd

try:
    from fla.ops.kda.naive import naive_recurrent_kda
except ImportError:
    naive_recurrent_kda = None


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


def run_case(B=2, T=63, H=2, K=64, V=32, seed=0,
             compile_fullgraph=False, atol=8e-3, rtol=8e-3) -> bool:
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)

    def mkp(*shape):
        return torch.randn(*shape, device=device, dtype=torch.float32,
                           generator=gen, requires_grad=True)

    q = F.normalize(torch.randn(B, T, H, K, device=device,
                                dtype=torch.float32, generator=gen),
                    p=2, dim=-1).detach().requires_grad_()
    k = F.normalize(torch.randn(B, T, H, K, device=device,
                                dtype=torch.float32, generator=gen),
                    p=2, dim=-1).detach().requires_grad_()
    v = mkp(B, T, H, V)
    # FLA KDA expects g in log-space. Keep most cases in a realistic negative
    # range, while still allowing non-64 sequence lengths.
    g = F.logsigmoid(torch.randn(B, T, H, K, device=device,
                                 dtype=torch.float32, generator=gen))
    g = g.detach().requires_grad_()
    beta = torch.sigmoid(torch.randn(B, T, H, device=device,
                                     dtype=torch.float32, generator=gen))
    beta = beta.detach().requires_grad_()
    do = torch.randn(B, T, H, V, device=device, dtype=torch.float32,
                     generator=gen)

    fn = kda_triton_autograd
    if compile_fullgraph:
        fn = torch.compile(fn, fullgraph=True)

    y = fn(q, k, v, g, beta)
    (y * do).sum().backward()
    dq, dk, dv, dg, dbeta = q.grad, k.grad, v.grad, g.grad, beta.grad

    q2, k2, v2, g2, beta2 = (t.detach().clone().requires_grad_()
                              for t in (q, k, v, g, beta))
    y_ref, _ = naive_recurrent_kda(q2, k2, v2, g2, beta2)
    (y_ref * do).sum().backward()

    label = "compiled" if compile_fullgraph else "eager"
    print(f"Case {label} B={B} T={T} H={H} K={K} V={V} seed={seed}")
    ok = True
    ok &= _check("y",     y,      y_ref,       atol, rtol)
    ok &= _check("dq",    dq,     q2.grad,     atol, rtol)
    ok &= _check("dk",    dk,     k2.grad,     atol, rtol)
    ok &= _check("dv",    dv,     v2.grad,     atol, rtol)
    ok &= _check("dg",    dg,     g2.grad,     2e-2, 2e-2)
    ok &= _check("dbeta", dbeta,  beta2.grad,  2e-2, 2e-2)
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required"); return 2
    if naive_recurrent_kda is None:
        print("fla-core required"); return 2

    all_ok = True
    all_ok &= run_case(B=1, T=63,  H=1, K=32, V=32, seed=0)
    all_ok &= run_case(B=2, T=128, H=2, K=64, V=64, seed=1)
    all_ok &= run_case(B=1, T=65,  H=2, K=64, V=32, seed=2,
                       compile_fullgraph=True)
    print("\n", "ALL PASS" if all_ok else "FAILURES", sep="")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
