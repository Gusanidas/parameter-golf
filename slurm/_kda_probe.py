"""Probe: does kda_triton_autograd match fla's naive_recurrent_kda?

Avoids T=63/65 which trigger an FLA autotune config that fails Triton
compilation on Blackwell (sm_100). Uses chunk-aligned T values that were
fine in the earlier profile run.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from triton_kernels.kda import kda_triton_autograd

try:
    from fla.ops.kda.naive import naive_recurrent_kda
except ImportError as e:
    print(f"fla.ops.kda.naive import failed: {e}")
    sys.exit(2)


def _max_err(a, b):
    d = (a.float() - b.float()).abs()
    rel = d / b.float().abs().clamp(min=1e-6)
    return d.max().item(), rel.max().item()


def _check(name, got, ref, atol, rtol):
    abs_err, rel_err = _max_err(got, ref)
    ok = torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol)
    flag = "PASS" if ok else "FAIL"
    print(f"  {flag} {name:<7s} max_abs={abs_err:.3e}  max_rel={rel_err:.3e}")
    return ok


def run(B, T, H, K, V, seed, compile_mode=False, atol=5e-2, rtol=5e-2,
        dtype=torch.bfloat16):
    """Default bf16 — matches the profile scripts and the FLA autotune path
    that works on Blackwell (the fp32 path hits a Triton compile bug)."""
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)

    q = F.normalize(torch.randn(B, T, H, K, device=device, dtype=dtype, generator=gen), p=2, dim=-1).detach().requires_grad_()
    k = F.normalize(torch.randn(B, T, H, K, device=device, dtype=dtype, generator=gen), p=2, dim=-1).detach().requires_grad_()
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, generator=gen, requires_grad=True)
    # g stays fp32 (log-space decay, avoid precision loss in cumsum inside kernel)
    g = F.logsigmoid(torch.randn(B, T, H, K, device=device, dtype=torch.float32, generator=gen)).detach().requires_grad_()
    beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32, generator=gen)).detach().requires_grad_()
    do = torch.randn(B, T, H, V, device=device, dtype=dtype, generator=gen)

    fn = kda_triton_autograd
    if compile_mode:
        fn = torch.compile(fn, fullgraph=True)

    y = fn(q, k, v, g, beta)
    (y * do).sum().backward()
    dq, dk, dv, dg, dbeta = q.grad, k.grad, v.grad, g.grad, beta.grad

    q2, k2, v2, g2, beta2 = (t.detach().clone().requires_grad_() for t in (q, k, v, g, beta))
    y_ref, _ = naive_recurrent_kda(q2, k2, v2, g2, beta2)
    (y_ref * do).sum().backward()

    mode = "compiled" if compile_mode else "eager"
    print(f"\n[{mode}] B={B} T={T} H={H} K={K} V={V} seed={seed}")
    ok = True
    ok &= _check("y",     y,     y_ref,      atol, rtol)
    ok &= _check("dq",    dq,    q2.grad,    atol, rtol)
    ok &= _check("dk",    dk,    k2.grad,    atol, rtol)
    ok &= _check("dv",    dv,    v2.grad,    atol, rtol)
    ok &= _check("dg",    dg,    g2.grad,    2e-2, 2e-2)
    ok &= _check("dbeta", dbeta, beta2.grad, 2e-2, 2e-2)
    return ok


def safe_run(**kwargs):
    try:
        return run(**kwargs)
    except Exception as e:
        tag = f"B={kwargs.get('B')} T={kwargs.get('T')} H={kwargs.get('H')} K={kwargs.get('K')} V={kwargs.get('V')}"
        print(f"\n[{tag}] CRASH  {type(e).__name__}: {str(e)[:200]}")
        return False


def main():
    if not torch.cuda.is_available():
        print("CUDA required"); return 2
    results = []
    # Run profile-matching shape first — we know this one works.
    results.append(("B=16 T=1024 H=6 K=64 V=64",
                    safe_run(B=16, T=1024, H=6, K=64, V=64, seed=0)))
    results.append(("B=2 T=2048 H=6 K=64 V=64",
                    safe_run(B=2, T=2048, H=6, K=64, V=64, seed=1)))
    results.append(("B=2 T=1024 H=6 K=64 V=64 [compiled]",
                    safe_run(B=2, T=1024, H=6, K=64, V=64, seed=2, compile_mode=True)))
    # Smaller shapes — may hit the Blackwell autotune issue.
    results.append(("B=2 T=128 H=2 K=64 V=64",
                    safe_run(B=2, T=128, H=2, K=64, V=64, seed=3)))
    results.append(("B=1 T=64 H=1 K=32 V=32",
                    safe_run(B=1, T=64,   H=1, K=32, V=32, seed=4)))

    print("\n--- summary ---")
    for label, ok in results:
        print(f"  {'PASS' if ok else 'FAIL/CRASH'}  {label}")
    all_ok = all(ok for _, ok in results)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
