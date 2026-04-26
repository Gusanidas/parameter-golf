#!/bin/bash
#SBATCH --job-name=pgolf-attn-width
#SBATCH --partition=llm
#SBATCH --qos=llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=128G
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/attn-width-%j.out

# Sweep two axes for FA2 full-causal:
#   (a) MLP width — vary MLP_MULT at fixed DIM=512, H*P=384.
#   (b) Model dim — vary DIM (residual + attn + MLP all scale)
#       with N_HEADS = DIM/64 so attn dim = DIM, mlp_mult fixed at 4.
# Common shape: B=8 T=2048, FA2 full-causal, 2 layers.

set -e
export XDG_RUNTIME_DIR="/tmp/enroot-runtime-$(id -u)"
mkdir -p "$XDG_RUNTIME_DIR"

CACHE_DIR="/sw/alejandro/.cache"
CONTAINER_IMAGE="${CACHE_DIR}/enroot/seq-modls.sqsh"
CONTAINER_NAME="seq-modls"
CONTAINER_MOUNTS="$PWD:/workspace,$CACHE_DIR/huggingface:/root/.cache/huggingface,$CACHE_DIR/uv:/root/.cache/uv,$HOME/.local/bin/uv:/usr/local/bin/uv:ro,/sw/alejandro:/sw/alejandro,/sw/public:/sw/public"

BSZ="${BSZ:-8}"
SEQLEN="${SEQLEN:-2048}"
N_LAYERS="${N_LAYERS:-2}"
N_ITERS="${N_ITERS:-50}"

TRACE_DIR="traces/${SLURM_JOB_ID}"
mkdir -p "$TRACE_DIR" slurm/logs

echo "=============================================="
echo "Parameter Golf - attn width sweep"
echo "Job ID:  $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Branch:  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
echo "Commit:  $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "Common:  B=$BSZ T=$SEQLEN layers=$N_LAYERS iters=$N_ITERS"
echo "Start:   $(date)"
echo "=============================================="

# Pre-create container with a lock so concurrent jobs don't race.
LOCK_FILE="/tmp/enroot-seq-modls-lock-$(id -u)"
(
    flock -w 600 9 || { echo ">>> Failed to acquire container lock"; exit 1; }
    if srun --overlap --container-name="$CONTAINER_NAME" \
            --container-mounts="$CONTAINER_MOUNTS" true 2>/dev/null; then
        echo ">>> Container already exists, reusing."
    else
        echo ">>> Extracting fresh container..."
        rm -rf "/tmp/enroot-data/user-$(id -u)/pyxis_${CONTAINER_NAME}" 2>/dev/null || true
        srun --overlap --container-image="$CONTAINER_IMAGE" \
             --container-name="$CONTAINER_NAME" \
             --container-mounts="$CONTAINER_MOUNTS" true
    fi
) 9>"$LOCK_FILE"

run_one() {
    local tag="$1"; shift
    local extra_env="$*"
    local trace_path="${TRACE_DIR}/${tag}_trace.json"

    echo ""
    echo "--- [$tag] ---"
    set +e
    srun --overlap \
         --container-image="$CONTAINER_IMAGE" \
         --container-name="${CONTAINER_NAME}-${SLURM_JOB_ID}-${tag}" \
         --container-mounts="$CONTAINER_MOUNTS" \
         --container-workdir=/workspace \
         bash -c "set -e; export NVIDIA_VISIBLE_DEVICES=all; \
                  export BSZ=$BSZ SEQLEN=$SEQLEN N_LAYERS=$N_LAYERS \
                         N_ITERS=$N_ITERS WINDOW=0 \
                         TRACE_PATH='$trace_path' $extra_env; \
                  pip install sentencepiece flash-attn 2>/dev/null; \
                  python -m triton_kernels.bench.profile_attention"
    local rc=$?
    set -e
    [ $rc -ne 0 ] && echo "!!! [$tag] exited rc=$rc — continuing"
    return 0
}

# (a) MLP width sweep — DIM=512, H=6, P=64 (attn dim=384).
for M in 2 4 6 8; do
    run_one "mlp_mult${M}" \
        "DIM=512 N_HEADS=6 HEAD_DIM=64 MLP_MULT=$M"
done

# (b) Model-dim sweep — N_HEADS=DIM/64 so attn dim = DIM, mlp_mult=4.
for D in 384 512 768 1024; do
    H=$((D / 64))
    run_one "dim${D}" \
        "DIM=$D N_HEADS=$H HEAD_DIM=64 MLP_MULT=4"
done

echo ""
echo "=============================================="
echo "Traces: $TRACE_DIR"
ls -lh "$TRACE_DIR" 2>/dev/null || true
echo "Done: $(date)"
echo "=============================================="
