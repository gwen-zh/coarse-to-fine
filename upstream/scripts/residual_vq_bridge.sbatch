#!/bin/bash
#SBATCH --job-name=hcg_rvq_bridge
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/rvq-bridge-%j.out
set -euo pipefail
TASK_ROOT=/net/galaxy/home/koes/zhangwen/coarsetofine
cd "$TASK_ROOT"
export HF_HOME="$TASK_ROOT/artifacts/hf_cache"
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$TASK_ROOT/artifacts/matplotlib"
export XDG_CACHE_HOME="$TASK_ROOT/artifacts/cache"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
.venv/bin/python upstream/experiments/evaluate_residual_vq.py \
  --manifest artifacts/development/manifest.json --cache artifacts/development/encoder_broad \
  --control artifacts/development/quantization_control --weights artifacts/tokenizer \
  --run artifacts/development/residual_vq --threads 8 --bridge fine_codebook
