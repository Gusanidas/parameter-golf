#!/bin/bash
#SBATCH --job-name=pgolf-kda-probe
#SBATCH --partition=llm
#SBATCH --qos=llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=00:08:00
#SBATCH --output=slurm/logs/kda-probe-%j.out

export XDG_RUNTIME_DIR="/tmp/enroot-runtime-$(id -u)"
mkdir -p "$XDG_RUNTIME_DIR"

CACHE_DIR="/sw/alejandro/.cache"
CONTAINER_IMAGE="${CACHE_DIR}/enroot/seq-modls.sqsh"
CONTAINER_NAME="seq-modls"
CONTAINER_MOUNTS="$PWD:/workspace,$CACHE_DIR/huggingface:/root/.cache/huggingface,$CACHE_DIR/uv:/root/.cache/uv,$HOME/.local/bin/uv:/usr/local/bin/uv:ro,/sw/alejandro:/sw/alejandro,/sw/public:/sw/public"

srun --overlap \
     --container-image="$CONTAINER_IMAGE" \
     --container-name="kda-probe-${SLURM_JOB_ID}" \
     --container-mounts="$CONTAINER_MOUNTS" \
     --container-workdir=/workspace \
     bash -c "set -e; export NVIDIA_VISIBLE_DEVICES=all; \
              pip install sentencepiece fla-core==0.4.2 2>/dev/null; \
              python slurm/_kda_probe.py"
