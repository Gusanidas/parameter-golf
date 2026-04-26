#!/bin/bash
#SBATCH --job-name=pgolf-profile-kernels
#SBATCH --partition=llm
#SBATCH --qos=llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=128G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/profile-%j.out

# Profile the Triton kernels on pr1584_mamba:
#   - triton_kernels.bench.profile_mamba2 (VARIANT=ssd, VARIANT=fused)
#   - triton_kernels.bench.profile_kda
#
# Each run emits a Chrome trace (.json) into traces/<SLURM_JOB_ID>/.
# Open traces in chrome://tracing or https://ui.perfetto.dev/
#
# Override any env var before sbatch, e.g.:
#   BSZ=8 SEQLEN=2048 sbatch slurm/run_profile_kernels.sbatch
#   RUN_ONLY=mamba2_fused sbatch slurm/run_profile_kernels.sbatch

set -e

export XDG_RUNTIME_DIR="/tmp/enroot-runtime-$(id -u)"
mkdir -p "$XDG_RUNTIME_DIR"

CACHE_DIR="/sw/alejandro/.cache"
CONTAINER_IMAGE="${CACHE_DIR}/enroot/seq-modls.sqsh"
CONTAINER_NAME="seq-modls"
CONTAINER_MOUNTS="$PWD:/workspace,$CACHE_DIR/huggingface:/root/.cache/huggingface,$CACHE_DIR/uv:/root/.cache/uv,$HOME/.local/bin/uv:/usr/local/bin/uv:ro,/sw/alejandro:/sw/alejandro,/sw/public:/sw/public"

BSZ="${BSZ:-4}"
SEQLEN="${SEQLEN:-1024}"
DIM="${DIM:-512}"
N_HEADS="${N_HEADS:-6}"
HEAD_DIM="${HEAD_DIM:-64}"
STATE_DIM="${STATE_DIM:-16}"
K_DIM="${K_DIM:-64}"
V_DIM="${V_DIM:-64}"
N_LAYERS="${N_LAYERS:-2}"
N_ITERS="${N_ITERS:-50}"
WINDOW="${WINDOW:-512}"
NO_COMPILE="${NO_COMPILE:-0}"
RUN_ONLY="${RUN_ONLY:-all}"   # all | mamba2_ssd | mamba2_fused | kda | attn_full | attn_window

TRACE_DIR="traces/${SLURM_JOB_ID}"
mkdir -p "$TRACE_DIR" slurm/logs

echo "=============================================="
echo "Parameter Golf - kernel profiling"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Branch:  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
echo "Commit:  $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "Trace:   $TRACE_DIR"
echo "Shapes:  B=$BSZ T=$SEQLEN D=$DIM H=$N_HEADS P=$HEAD_DIM N=$STATE_DIM layers=$N_LAYERS iters=$N_ITERS"
echo "Run:     $RUN_ONLY"
echo "Start:   $(date)"
echo "=============================================="

# Pre-create container (lock so multiple jobs don't race).
LOCK_FILE="/tmp/enroot-seq-modls-lock-$(id -u)"
(
    flock -w 600 9 || { echo ">>> Failed to acquire container lock after 600s"; exit 1; }
    if srun --overlap \
         --container-name="$CONTAINER_NAME" \
         --container-mounts="$CONTAINER_MOUNTS" \
         true 2>/dev/null; then
        echo ">>> Container already exists, reusing."
    else
        echo ">>> Extracting fresh container..."
        rm -rf "/tmp/enroot-data/user-$(id -u)/pyxis_${CONTAINER_NAME}" 2>/dev/null || true
        srun --overlap \
             --container-image="$CONTAINER_IMAGE" \
             --container-name="$CONTAINER_NAME" \
             --container-mounts="$CONTAINER_MOUNTS" \
             true
        echo ">>> Container extracted."
    fi
) 9>"$LOCK_FILE"

run_one() {
    local tag="$1"; shift
    local mod="$1"; shift
    local extra_env="$*"

    if [ "$RUN_ONLY" != "all" ] && [ "$RUN_ONLY" != "$tag" ]; then
        return 0
    fi

    echo ""
    echo "--- [$tag] ${mod} ---"
    local trace_path="${TRACE_DIR}/${tag}_trace.json"

    set +e
    srun --overlap \
         --container-image="$CONTAINER_IMAGE" \
         --container-name="${CONTAINER_NAME}-${SLURM_JOB_ID}-${tag}" \
         --container-mounts="$CONTAINER_MOUNTS" \
         --container-workdir=/workspace \
         bash -c "set -e; \
                  export NVIDIA_VISIBLE_DEVICES=all; \
                  export BSZ=$BSZ SEQLEN=$SEQLEN DIM=$DIM N_HEADS=$N_HEADS \
                         HEAD_DIM=$HEAD_DIM STATE_DIM=$STATE_DIM \
                         K_DIM=$K_DIM V_DIM=$V_DIM \
                         N_LAYERS=$N_LAYERS N_ITERS=$N_ITERS \
                         NO_COMPILE=$NO_COMPILE TRACE_PATH='$trace_path' \
                         $extra_env; \
                  pip install sentencepiece fla-core==0.4.2 2>/dev/null; \
                  python -m $mod"
    local rc=$?
    set -e
    if [ $rc -ne 0 ]; then
        echo "!!! [$tag] exited with code $rc — continuing with remaining variants"
    fi
    return 0
}

run_one mamba2_ssd   triton_kernels.bench.profile_mamba2    "VARIANT=ssd"
run_one mamba2_fused triton_kernels.bench.profile_mamba2    "VARIANT=fused"
run_one kda          triton_kernels.bench.profile_kda       ""
run_one attn_full    triton_kernels.bench.profile_attention "WINDOW=0"
run_one attn_window  triton_kernels.bench.profile_attention "WINDOW=$WINDOW"

echo ""
echo "=============================================="
echo "Traces in: $TRACE_DIR"
ls -lh "$TRACE_DIR" 2>/dev/null || true
echo "Done: $(date)"
echo "=============================================="
