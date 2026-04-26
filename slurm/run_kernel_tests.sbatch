#!/bin/bash
#SBATCH --job-name=pgolf-kernel-tests
#SBATCH --partition=llm
#SBATCH --qos=llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/kernel-tests-%j.out

# Run the Triton-kernel correctness tests in the same container as the
# profilers. Compares Triton fwd+bwd against the slow PyTorch reference in
# triton_kernels/tests/reference.py.

set -e

export XDG_RUNTIME_DIR="/tmp/enroot-runtime-$(id -u)"
mkdir -p "$XDG_RUNTIME_DIR"

CACHE_DIR="/sw/alejandro/.cache"
CONTAINER_IMAGE="${CACHE_DIR}/enroot/seq-modls.sqsh"
CONTAINER_NAME="seq-modls"
CONTAINER_MOUNTS="$PWD:/workspace,$CACHE_DIR/huggingface:/root/.cache/huggingface,$CACHE_DIR/uv:/root/.cache/uv,$HOME/.local/bin/uv:/usr/local/bin/uv:ro,/sw/alejandro:/sw/alejandro,/sw/public:/sw/public"

echo "=============================================="
echo "Kernel correctness tests"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Branch:  $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
echo "Commit:  $(git rev-parse --short HEAD 2>/dev/null)"
echo "Start:   $(date)"
echo "=============================================="

run_test() {
    local mod="$1"
    echo ""
    echo "--- $mod ---"
    set +e
    srun --overlap \
         --container-image="$CONTAINER_IMAGE" \
         --container-name="${CONTAINER_NAME}-tests-${SLURM_JOB_ID}" \
         --container-mounts="$CONTAINER_MOUNTS" \
         --container-workdir=/workspace \
         bash -c "set -e; export NVIDIA_VISIBLE_DEVICES=all; \
                  pip install sentencepiece fla-core==0.4.2 2>/dev/null; \
                  python -m $mod"
    local rc=$?
    set -e
    if [ $rc -ne 0 ]; then
        echo "!!! $mod exited with code $rc — continuing"
    fi
    return 0
}

run_test triton_kernels.tests.test_mamba2
run_test triton_kernels.tests.test_kda

echo ""
echo "=============================================="
echo "Done: $(date)"
echo "=============================================="
