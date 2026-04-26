#!/bin/bash
#SBATCH --job-name=pgolf-mlp-probe
#SBATCH --partition=llm
#SBATCH --qos=llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/mlp-probe-%j.out

export XDG_RUNTIME_DIR="/tmp/enroot-runtime-$(id -u)"
mkdir -p "$XDG_RUNTIME_DIR"
CACHE_DIR="/sw/alejandro/.cache"
CONTAINER_IMAGE="${CACHE_DIR}/enroot/seq-modls.sqsh"
CONTAINER_NAME="seq-modls"
CONTAINER_MOUNTS="$PWD:/workspace,$CACHE_DIR/huggingface:/root/.cache/huggingface,$CACHE_DIR/uv:/root/.cache/uv,$HOME/.local/bin/uv:/usr/local/bin/uv:ro,/sw/alejandro:/sw/alejandro,/sw/public:/sw/public"

run_one() {
    local label="$1"; shift
    local extra="$*"
    echo ""
    echo "--- $label ---"
    srun --overlap \
         --container-image="$CONTAINER_IMAGE" \
         --container-name="${CONTAINER_NAME}-mlp-${SLURM_JOB_ID}-$RANDOM" \
         --container-mounts="$CONTAINER_MOUNTS" \
         --container-workdir=/workspace \
         bash -c "set -e; export NVIDIA_VISIBLE_DEVICES=all; \
                  pip install sentencepiece fla-core==0.4.2 2>/dev/null; \
                  $extra python slurm/_mlp_probe.py"
}

run_one "B=16 T=1024" "BSZ=16 SEQLEN=1024"
run_one "B=16 T=2048" "BSZ=16 SEQLEN=2048"
run_one "B=8  T=4096" "BSZ=8  SEQLEN=4096"
