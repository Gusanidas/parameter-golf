"""Shared benchmarking utilities for kernel minimodel profiling."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),))


class MLP(nn.Module):
    """Simple squared-ReLU MLP, matching the style in the training script."""

    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        h = int(dim * mult)
        self.up = nn.Linear(dim, h, bias=False)
        self.down = nn.Linear(h, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.relu(self.up(x)).square())


@dataclass
class BenchResult:
    fwd_ms: list[float]
    bwd_ms: list[float]
    peak_mem_mb: float

    def summary(self) -> str:
        f_med, f_mean = median(self.fwd_ms), sum(self.fwd_ms) / len(self.fwd_ms)
        b_med, b_mean = median(self.bwd_ms), sum(self.bwd_ms) / len(self.bwd_ms)
        tot = f_med + b_med
        return (
            f"  fwd  median={f_med:7.3f} ms  mean={f_mean:7.3f} ms\n"
            f"  bwd  median={b_med:7.3f} ms  mean={b_mean:7.3f} ms\n"
            f"  fwd+bwd median={tot:7.3f} ms\n"
            f"  peak memory: {self.peak_mem_mb:.1f} MB"
        )


def time_fwd_bwd(
    step: Callable[[], torch.Tensor],
    iters: int = 50,
    warmup: int = 10,
) -> BenchResult:
    """Time the given (fwd → scalar) closure + backward. `step` must return a
    scalar loss. Uses CUDA events for ms-level precision.
    """
    assert torch.cuda.is_available()
    # Warmup (also triggers torch.compile recompiles / kernel autotune).
    for _ in range(warmup):
        loss = step()
        loss.backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    fwd_ms, bwd_ms = [], []
    for _ in range(iters):
        f0 = torch.cuda.Event(enable_timing=True)
        f1 = torch.cuda.Event(enable_timing=True)
        b1 = torch.cuda.Event(enable_timing=True)
        f0.record()
        loss = step()
        f1.record()
        loss.backward()
        b1.record()
        torch.cuda.synchronize()
        fwd_ms.append(f0.elapsed_time(f1))
        bwd_ms.append(f1.elapsed_time(b1))
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return BenchResult(fwd_ms, bwd_ms, peak_mb)


def run_profiler_trace(
    step: Callable[[], torch.Tensor],
    out_path: str,
    iters: int = 5,
    warmup: int = 3,
) -> None:
    """Capture a torch.profiler trace to `out_path` (chrome trace json)."""
    from torch.profiler import profile, ProfilerActivity, schedule

    for _ in range(warmup):
        step().backward()
    torch.cuda.synchronize()

    sched = schedule(wait=0, warmup=1, active=iters, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(iters + 1):
            step().backward()
            prof.step()
    prof.export_chrome_trace(out_path)
    print(f"  trace written to {out_path}")
    print("  top 10 CUDA ops by self time:")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=10))
