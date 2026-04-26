# pr1584_mamba — kernel/MLP optimization plan

Branch state: `pr1584_mamba` at `666a9ca` (Parallelize causal conv kernels).

Outstanding bugs (independent of these items):
- mamba2_ssd compiled correctness regression (`y` max_rel=15.5)
- mamba2_fused Triton smem OOM on B200 (304 KB needed, 227 KB cap) — also fails on H100
- FLA KDA fp32 PassManager fail on B200 (Blackwell-specific, not on H100)

## Action items (priority order)

### 1. Pack KDA projections — `triton_kernels/bench/profile_kda.py`
Replace 9 separate `nn.Linear` calls with 3 packed:
- `qkv_proj` : `D → 2·H·K + H·V` (split into q, k, v after)
- `f_proj`   : collapse 2-linear bottleneck `D→V→H·K` to one `D → H·K`
- `g_proj`   : collapse 2-linear bottleneck `D→V→H·V` to one `D → H·V` (with bias)
Keep `b_proj` and `out_proj` as is.
Verify: profile at BSZ=16, T=1024,2048,4096 — expect ~15% faster KDA from fewer launches.

### 2. Fix `mamba2_fused`
Two issues:
- (a) backward smem OOM on Blackwell: add a Triton autotune config with smaller block sizes / fewer stages so something fits in ≤227 KB SMEM. File: `triton_kernels/mamba2_fused.py`.
- (b) skip-path duplication: forward currently runs `silu(causal_conv1d(x_ssm))` once for `x_skip` AND again inside the fused kernel. Either save the conv output once and reuse, or drop the external `x_skip` conv and read the kernel-internal pre-SiLU activation back out (the fused kernel's bwd already saves it).
Verify: `slurm/run_kernel_tests.sbatch` (test_mamba2 fused compiled) PASSes, profile shows fused < ssd.

### 3. Triton-fuse conv1d + SiLU into the SSD/KDA kernel front-end
Currently `silu(causal_conv1d(x))` is two kernel launches. Merge into one Triton kernel exposed as a custom_op. New file likely `triton_kernels/conv1d_silu.py`. Wire into `profile_mamba2.py` (ssd variant) and `profile_kda.py`.
Verify: per-call latency drops; profile total drops by ~half the conv1d cost.

### 4. State-dim ablation sweep
Run profile_kernels.sbatch across:
- mamba2: `STATE_DIM ∈ {8, 16, 32}`
- kda:    `K_DIM=V_DIM ∈ {32, 48, 64}`
At BSZ=16, T=1024 and T=2048. Tabulate fwd+bwd ms and per-token cost — Pareto vs default config.

### 5. (Stretch) Low-rank state factorization
Mamba2 state `[H,P,N]` and KDA state `[H,K,V]` factorized as `[H,P,r]+[r,N]`. Research-grade; out of scope for this round.

## Cross-cutting infra (already in repo)

- `slurm/run_profile_kernels.sbatch` — 5-variant profile, env vars `BSZ SEQLEN N_HEADS HEAD_DIM STATE_DIM K_DIM V_DIM N_LAYERS N_ITERS WINDOW NO_COMPILE RUN_ONLY`
- `slurm/run_kernel_tests.sbatch` — test_mamba2 + test_kda
- `slurm/run_mlp_probe.sbatch` — MLP-only timing
- Logs: `slurm/logs/`. Traces: `traces/<JOB_ID>/`.
- Container: `seq-modls.sqsh` (FA2, no FA3, fla-core 0.4.2 installed at job start)
- GPU: B200 (also need to keep H100 portability — separate FA3 build infra exists in cousin branch).

## What's been verified working at `666a9ca`

| variant | T=1024 | T=2048 | T=4096(B=8) | µs/tok at T=2048 |
|---|---|---|---|---|
| attn_full (FA2) | 3.02 | 5.94 | 7.15 | 0.181 |
| attn_window=512 | 3.03 | 5.54 | 5.52 | 0.169 |
| mamba2_ssd | 4.34 | 7.31 | 8.31 | 0.223 |
| mamba2_fused | 5.40 | 9.63 | 13.10 | 0.294 |
| kda (FLA) | 6.13 | 9.44 | 9.51 | 0.288 |

KDA correctness (bf16) verified vs `naive_recurrent_kda` reference at all profile shapes.
