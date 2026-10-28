#!/bin/bash
#SBATCH --job-name=hcg_rvq_replicate
#SBATCH --partition=dept_cpu
#SBATCH --array=1-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rvq-replicate-%A_%a.out
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
RVQ_OUTPUT="artifacts/development/residual_vq_seed${SLURM_ARRAY_TASK_ID}"
.venv/bin/python upstream/experiments/train_residual_vq.py \
  --manifest artifacts/development/manifest.json --cache artifacts/development/encoder_broad \
  --control artifacts/development/quantization_control --weights artifacts/tokenizer \
  --output "$RVQ_OUTPUT" --epochs 30 --threads 8 --seed "$SLURM_ARRAY_TASK_ID"
.venv/bin/python upstream/experiments/evaluate_residual_vq.py \
  --manifest artifacts/development/manifest.json --cache artifacts/development/encoder_broad \
  --control artifacts/development/quantization_control --weights artifacts/tokenizer \
  --run "$RVQ_OUTPUT" --threads 8 --bridge fine_codebook
