# pr1584_mamba — kernel profiling summary

End-to-end summary of the work done on the `pr1584_mamba` branch of
`Gusanidas/parameter-golf`: profiling infrastructure, the variants we
benchmark, the optimizations landed so far, the latest numbers (now
including T=4096 and T=8192), and what's still open.

Branch tip: `666a9ca` (Parallelize causal conv kernels). Local working
tree carries packed-projection edits for `profile_kda.py`,
`record_function` scopes for `profile_mamba2.py`, and an autotune-pin
in `triton_kernels/mamba2_fused.py`.

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

## Latest numbers (post-`666a9ca`, all bf16, B200)

### fwd+bwd, ms (median over 50 iters, 10 warmup, compiled)

| variant         | T=1024 B=16 | T=2048 B=16 | T=4096 B=8 | T=8192 B=4 |
|-----------------|-------------|-------------|------------|------------|
| attn_window=512 | 3.03        | 5.54        | 5.58       | **5.60**   |
| attn_full       | 3.02        | 5.94        | 7.16       | 9.42       |
| mamba2_ssd      | 4.34        | 7.31        | 8.33       | 10.24      |
| kda (packed)    | 6.13        | 9.44        | 9.48       | 9.85       |
| mamba2_fused ⚠  | 5.40        | 9.63        | 18.12      | 29.69      |

⚠ `mamba2_fused` regressed after the SMEM-safe autotune pin (see Open
items §2 below). Pre-pin numbers at T=4096 were ~13 ms.

### fwd+bwd, µs/token (32K tokens at T=4096 B=8 and T=8192 B=4)

| variant         | T=4096 B=8 | T=8192 B=4 |
|-----------------|------------|------------|
| attn_window=512 | 0.170      | 0.171      |
| attn_full       | 0.218      | 0.287      |
| mamba2_ssd      | 0.254      | 0.313      |
| kda             | 0.289      | 0.301      |
| mamba2_fused    | 0.553      | 0.906      |

### fwd / bwd split at the new shapes (ms)

| variant         | T=4096 fwd | T=4096 bwd | T=8192 fwd | T=8192 bwd |
|-----------------|------------|------------|------------|------------|
| attn_window=512 | 2.77       | 2.81       | 2.77       | 2.83       |
| attn_full       | 3.12       | 4.04       | 3.76       | 5.66       |
| mamba2_ssd      | 3.36       | 4.97       | 3.78       | 6.46       |
| kda             | 3.94       | 5.54       | 4.00       | 5.85       |
| mamba2_fused    | 4.69       | 13.43      | 6.66       | 23.04      |

### Headlines from the seqlen sweep
- **attn_window is essentially flat** in token-throughput from T=2048→T=8192
  (5.54 → 5.60 ms at constant ~32K tokens). FA2's window kernel is
  doing exactly the work expected: O(T·W).
- **attn_full is ~T¹·²** here (5.94 → 9.42 over 4×T at constant tokens).
  At T=8192 the QK² term starts visible; FA2 is still memory-bound
  enough that the full T² coefficient hasn't hit yet.
- **mamba2_ssd and KDA scale linearly with T** — their µs/tok rises
  only because batch shrank and bf16 GEMMs were already saturated at
  B=8.
- **The "linear vs attention" crossover hasn't happened.** At T=8192
  attn_full (9.42) and kda (9.85) and mamba2_ssd (10.24) are within
  ~10% of each other. We'd need T≈16K to see attention pull ahead, and
  even then window=512 stays well below all of them.

---

## Optimizations landed

1. **Conv1d parallelization** (`666a9ca`, upstream). The dominant cost
   in the linear mixers was three sequential `causal_conv1d` launches.
   After this commit conv1d is parallelized across H × dim and all
   linear mixers got 43–74% faster vs the pre-fix baseline.
2. **KDA projection pack** (local). Replaced 9 separate `nn.Linear`s
   with 5 (`qkv_proj` packed Q/K/V, `f_proj` and `g_proj` collapsed
   their D→V→… bottlenecks into a single matmul). Saves ~6 launches
   per layer; modest ~3% speedup on KDA mixer.
3. **mamba2_fused SMEM fix** (local, regressing). Pinned the bwd
   kernel to a single small autotune config so it fits in B200's 227
   KB SMEM cap. The Triton kernel no longer OOMs on Blackwell, but
   throughput dropped — the prior aggressive configs were faster
   wherever they fit. Net effect at the profile shapes: slower than
   ssd. **This is the single biggest open perf bug.**

## Component breakdown (ms at T=2048 B=16)

| component        | time | notes |
|---|---|---|
| FA2 kernel only  | ~0.32 | from trace; ~12% of attn_full step total |
| MLP (2 layers, fwd+bwd) | ~1.7 | from `_mlp_probe.py` |
| attn_full step total | 5.94 | so non-MLP, non-FA2 = ~3.9 (proj, RMSNorm, embed/head, framework) |
| kda step total   | 9.44 | mixer dominates; conv1d×3 still ~30% of mixer |

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

## Open items (priority order, also in `slurm/PLAN.md`)

### 1. Fix `mamba2_fused` (highest impact)
- (a) Replace the pinned-single-config autotune with 2–3 less
  conservative configs that still fit in 227 KB SMEM. File:
  `triton_kernels/mamba2_fused.py`.
- (b) Skip-path duplication: forward currently runs
  `silu(causal_conv1d(x_ssm))` once for `x_skip` *and* again inside
  the fused kernel. Either save the conv output once and reuse, or
  read the kernel-internal pre-SiLU activation back out (the bwd
  already saves it).
- Expected: drop from 18 ms → ~7 ms at T=4096 B=8.

### 2. Triton-fuse `conv1d + SiLU` (item 3 in `PLAN.md`)
Currently `silu(causal_conv1d(x))` is two kernel launches per branch
× 3 branches per layer × 2 layers = 12 launches per step that can be
4. Likely new file `triton_kernels/conv1d_silu.py`. Expected: shaves
the conv1d portion of every linear mixer roughly in half.

### 3. State-dim ablation
Run `K_DIM=V_DIM ∈ {32,48,64}` and mamba2 `STATE_DIM ∈ {8,16,32}` at
B=16 T∈{1024,2048}. Pareto vs default config. Two known constraints
the kernels currently hit:
- N=8 fails (Triton requires matmul tile ≥16).
- K=V=48 fails (FLA KDA `tl.arange` requires power-of-2).

### 4. Outstanding correctness regressions (independent)
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

This run: jobs 358922 (T=4096) and 358923 (T=8192).
