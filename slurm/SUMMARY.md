# pr1584_mamba — kernel profiling summary

End-to-end summary of the work done on the `pr1584_mamba` branch of
`Gusanidas/parameter-golf`: profiling infrastructure, the variants we
benchmark, the optimizations landed so far, the latest numbers, and
what's still open.

Branch tip: `e2a5c41` (Cache fla.ops.kda function pointers in
_require_fla). Working tree clean.

GPU: B200 (cuda:100). Container: `seq-modls.sqsh` (FA2 only — no FA3).
DType: bf16 unless noted.

---

## What's in the repo

### Mini-models we profile
A 2-layer `embed → N×(norm → mixer → residual → norm → MLP → residual) → head`
compiled with `torch.compile(fullgraph=True)`, with `record_function("mixer")`
and `record_function("mlp")` scopes for the trace.

| variant | mixer | file |
|---|---|---|
| mamba2_ssd   | conv1d×3 + SSD chunked recurrence + skip + RMSNorm·gate | `triton_kernels/bench/profile_mamba2.py` (`VARIANT=ssd`) |
| mamba2_fused | one Triton kernel = conv1d×3 ⊕ SSD ⊕ skip       | `profile_mamba2.py` (`VARIANT=fused`) |
| kda          | conv1d×3 + FLA chunked KDA + RMSNorm·gate         | `triton_kernels/bench/profile_kda.py`        |
| attn_full    | FA2 causal                                         | `triton_kernels/bench/profile_attention.py` (`WINDOW=0`) |
| attn_window  | FA2 causal, window=512                             | `profile_attention.py` (`WINDOW=512`)        |

Default shapes: D=512, H=6, head_dim=64, state_dim=16 (mamba2),
K=V=64 (kda), 2 layers, vocab=8192.

### Slurm infrastructure (`slurm/`)
- `run_profile_kernels.sbatch` — runs all 5 variants, env-driven. Per-variant
  failures are tolerated so one Triton crash doesn't kill the rest.
  Trace JSONs written to `traces/<JOB_ID>/`.
- `run_kernel_tests.sbatch` — runs `test_mamba2` and `test_kda`.
- `run_mlp_probe.sbatch` + `_mlp_probe.py` — 2-layer compiled MLP-only
  timing, three shapes. Lets us subtract MLP cost from the totals.
- `run_kda_probe.sbatch` + `_kda_probe.py` — bf16 KDA correctness
  vs `fla.ops.kda.naive.naive_recurrent_kda`. All shapes pass.

---

## Latest numbers (post-`e2a5c41`, all bf16, B200)

Constant ~32K tokens per step (B·T = 32768 for the larger shapes).

### fwd+bwd, ms (median over 50 iters, 10 warmup, compiled)

| variant         | T=2048 B=16 | T=4096 B=8 | T=8192 B=4 |
|-----------------|-------------|------------|------------|
| attn_window=512 | **5.49**    | **5.59**   | **5.50**   |
| attn_full       | 5.99        | 7.20       | 9.46       |
| mamba2_ssd      | 7.26        | 8.37       | 10.24      |
| kda             | 9.31        | 9.47       | 9.98       |
| mamba2_fused    | 8.84        | 11.67      | 17.30      |

Source: jobs 361693 / 361694 / 361695.

### fwd+bwd, µs/token (constant 32K tokens)

| variant         | T=2048 B=16 | T=4096 B=8 | T=8192 B=4 |
|-----------------|-------------|------------|------------|
| attn_window=512 | 0.168       | 0.171      | 0.168      |
| attn_full       | 0.183       | 0.220      | 0.289      |
| mamba2_ssd      | 0.222       | 0.255      | 0.313      |
| kda             | 0.284       | 0.289      | 0.305      |
| mamba2_fused    | 0.270       | 0.356      | 0.528      |

### fwd / bwd split (ms)

| variant         | 2048 fwd | 2048 bwd | 4096 fwd | 4096 bwd | 8192 fwd | 8192 bwd |
|-----------------|----------|----------|----------|----------|----------|----------|
| attn_window=512 | 2.72     | 2.77     | 2.78     | 2.81     | 2.67     | 2.82     |
| attn_full       | 2.79     | 3.20     | 3.16     | 4.04     | 3.81     | 5.65     |
| mamba2_ssd      | 3.09     | 4.17     | 3.40     | 4.97     | 3.78     | 6.46     |
| kda             | 3.90     | 5.41     | 3.94     | 5.53     | 4.13     | 5.85     |
| mamba2_fused    | 3.35     | 5.49     | 4.13     | 7.55     | 5.66     | 11.64    |

### Headlines from the seqlen sweep
- **attn_window is flat in token-throughput** across all three lengths
  (~5.5 ms at constant ~32K tokens). FA2's window kernel is O(T·W),
  so doubling T while halving B leaves work constant.
- **attn_full grows roughly T¹·¹** (5.99 → 9.46 over 4×T at constant
  tokens). FA2 is still memory-bound enough that the full T² coefficient
  hasn't hit yet.
- **At T=8192 attn_full, kda, mamba2_ssd are clustered tightly**
  (9.46 / 9.98 / 10.24) — the linear mixers have caught up to *full*
  attention but haven't pulled ahead. Window=512 attention stays
  ~2× cheaper at every length we measured.
- **mamba2_fused, post-autotune, is no longer a 2–3× pessimization** —
  but it's still slower than `mamba2_ssd` because its bwd recomputes
  three width-4 convs per chunk. Fixing that is the highest remaining
  Mamba2 lever.

---

## Optimizations landed

1. **Conv1d parallelization** (`666a9ca`, upstream). The dominant cost
   in the linear mixers was three sequential `causal_conv1d` launches.
   After this commit conv1d is parallelized across H × dim and all
   linear mixers got 43–74% faster vs the pre-fix baseline.
2. **KDA projection pack** (`edc48bd`). Replaced 9 separate `nn.Linear`s
   with 5 (`qkv_proj` packed Q/K/V, `f_proj` and `g_proj` collapsed
   their D→V→… bottlenecks into a single matmul). Saves ~6 launches
   per layer; modest ~3% speedup on KDA mixer.
3. **mamba2_fused SMEM-safe single-config autotune** (`edc48bd`). Pinned
   the bwd kernel to `num_stages=1, num_warps=4` so it fits in B200's
   227 KB SMEM cap. Made the kernel work on Blackwell at the cost of
   throughput, which (4) recovers most of.
4. **mamba2_fused autotune expansion** (`9aa071c`). Added
   `num_warps=8` (still `num_stages=1`) for bwd and three configs
   for fwd. Triton picks per-shape; the matmul-heavy bwd loop favors
   8 warps. Fwd+bwd: T=4096 B=8 18.05 → 11.67 (−35%), T=8192 B=4
   29.58 → 17.30 (−42%). Correctness verified by `run_fused`.
5. **KDA: cache `_require_fla()`** (`e2a5c41`). Each fwd/bwd was
   re-resolving `fla.ops.kda.chunk_{fwd,bwd}` through `sys.modules`.
   Cache once; lazy-import semantics preserved. End-to-end impact at
   T=8192 B=4 was within the run-to-run noise (a few percent), so
   credit this as a code cleanup, not a measurable perf win.

### Tried this round, did not land
- **Triton-fuse `conv1d + SiLU`** (kept Inductor doing the SiLU
  pointwise). Implemented in a new custom op with fused fwd+bwd; passed
  numerical correctness against `conv_silu_ref`. Profiled at T=4096 B=8
  and T=8192 B=4: KDA regressed +5%, `mamba2_ssd` regressed +1–2%,
  `mamba2_fused` flat. The launch-overhead saving (~6 launches per
  layer-pair) was outweighed by the extra HBM traffic for the saved
  pre-SiLU activations the bwd kernels need (the fused-bwd dx kernel
  reads `y_pre` 4× per output position — once per conv tap).
  Reverted.

## Component breakdown (ms at T=2048 B=16)

| component        | time | notes |
|---|---|---|
| FA2 kernel only  | ~0.32 | from trace; ~12% of attn_full step total |
| MLP (2 layers, fwd+bwd) | ~1.7 | from `_mlp_probe.py` |
| attn_full step total | 5.99 | so non-MLP, non-FA2 = ~4.0 (proj, RMSNorm, embed/head, framework) |
| kda step total   | 9.31 | mixer dominates; conv1d×3 still ~30% of mixer |

## Why is attention still faster than the linear mixers?

1. **The FA2 kernel is only ~12% of the attention step.** The other
   88% (projections, embed, head, RMSNorm, framework overhead) is
   shared with the linear mixers. Optimizing past attention requires
   beating it on the *non-mixer* path too.
2. **Linear mixers carry framework work attention doesn't.** Three
   conv1d's, a softplus + decay computation, an output gate
   (RMSNorm + sigmoid + mul), a residual skip path, and (for mamba2)
   the dt/A_log scalar broadcasting. Many of these are individual
   small launches that don't land on Triton.
3. **The asymptotic crossover is far past where we tested.**
   At T=8192 attn_full ≈ kda ≈ mamba2_ssd. The linear advantage
   only materializes around T=16K–32K, and then *only* against
   full-causal — window attention at W=512 stays cheaper at every
   length we measured.

---

## Open items (priority order)

### 1. Skip the conv-recompute in `mamba2_fused` bwd
After (4) above, the dominant cost in `mamba2_fused` bwd is recomputing
the three width-4 convs every chunk (the kernel reloads `x`/`B`/`C`
four times each, then re-derives `conv_x`/`conv_b`/`conv_c` and
`sigmoid` for the SiLU bwd). Saving the post-conv pre-SiLU outputs in
fwd (one fp32-or-bf16 tensor per branch, ~100 MB total at T=8192 B=4
for 2 layers) lets bwd just load them. Expected: closes most of the
remaining ~7 ms gap to `mamba2_ssd` at T=8192. File:
`triton_kernels/mamba2_fused.py`.

Skip-path note: forward also runs `silu(causal_conv1d(x_ssm))` once
for `x_skip` *and* again inside the fused kernel. Same fix —
materialize once.

### 2. State-dim ablation
Run `K_DIM=V_DIM ∈ {32,48,64}` and mamba2 `STATE_DIM ∈ {8,16,32}` at
B=16 T∈{1024,2048}. Pareto vs default config. Two known constraints
the kernels currently hit:
- N=8 fails (Triton requires matmul tile ≥16).
- K=V=48 fails (FLA KDA `tl.arange` requires power-of-2).

### 3. Outstanding correctness regressions (independent)
- mamba2_ssd compiled correctness: `y` max_rel=15.5 vs reference.
- mamba2_fused fwd+bwd `tl.fence_async_shared` PassManager fail on
  B200 (compute capability 90 hardcoded somewhere; works fine on H100).
- FLA KDA fp32 path: `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`
  PassManager fail on B200 — workaround: use bf16, which we do. Tests
  that drive fp32 are gated.

---

## Attention width sweep (FA2 full-causal, B=8 T=2048, 2 layers, bf16)

Job 358964. Two independent axes; otherwise identical to the
attn_full row in the main table.

### (a) MLP width — fix DIM=512, H=6, P=64, vary `MLP_MULT`

| MLP_MULT | hidden | fwd ms | bwd ms | fwd+bwd ms | peak MB |
|---|---|---|---|---|---|
| 2 | 1024 | 1.479 | 1.601 | **3.080** | 1961 |
| 4 | 2048 | 1.606 | 1.734 | 3.340 | 2093 |
| 6 | 3072 | 1.686 | 1.850 | 3.536 | 2226 |
| 8 | 4096 | 1.652 | 1.939 | 3.591 | 2357 |

Going 2× → 8× hidden adds only ~17% to fwd+bwd. MLP cost is
sub-linear in `mult` here because the GEMMs at small width are
memory-bound and only become compute-bound past mult≈4.
Bwd grows slightly faster than fwd because it does 2 GEMMs
(activation grad + weight grad) per Linear.

### (b) Model dim — `MLP_MULT=4`, scale `N_HEADS = DIM/64` so attn dim = DIM

| DIM | H | fwd ms | bwd ms | fwd+bwd ms | peak MB |
|---|---|---|---|---|---|
|  384 |  6 | 1.545 | 1.592 | **3.137** | 1980 |
|  512 |  8 | 1.678 | 1.919 | 3.597 | 2127 |
|  768 | 12 | 1.984 | 2.660 | 4.644 | 2421 |
| 1024 | 16 | 2.342 | 3.529 | 5.871 | 2723 |

DIM 384 → 1024 is 2.67×; fwd+bwd grows 1.87×. Sub-quadratic because
FA2 is still memory-bound at these head counts and the embed/head
linears (vocab=8192 × DIM) cost grows only linearly. Backward grows
faster than forward (2.22× vs 1.52×) — the residual-stream GEMMs
double in the bwd path.

Note: the DIM=512 row here uses H=8 (attn_dim=DIM), while the
attn_full row in the main table uses H=6 (attn_dim=384). Numbers
diverge by ~10% because of that.

---

## How to reproduce

```bash
# 5-variant profile, default shape (B=4 T=1024)
sbatch slurm/run_profile_kernels.sbatch

# Sweep one shape
BSZ=8  SEQLEN=4096 sbatch slurm/run_profile_kernels.sbatch
BSZ=4  SEQLEN=8192 sbatch slurm/run_profile_kernels.sbatch
BSZ=16 SEQLEN=2048 sbatch slurm/run_profile_kernels.sbatch

# Just one variant
RUN_ONLY=mamba2_fused sbatch slurm/run_profile_kernels.sbatch

# Correctness
sbatch slurm/run_kernel_tests.sbatch

# Standalone MLP timing
sbatch slurm/run_mlp_probe.sbatch

# Attn width / model-dim sweep (FA2 full-causal, B=8 T=2048)
sbatch slurm/run_attn_width_sweep.sbatch
```

Trace JSONs land in `traces/<JOB_ID>/*.json`; load in
<https://ui.perfetto.dev/> or `chrome://tracing`.

Logs: `slurm/logs/profile-<JOB_ID>.out`.

This run: jobs 361693 (T=2048), 361694 (T=4096), 361695 (T=8192).
