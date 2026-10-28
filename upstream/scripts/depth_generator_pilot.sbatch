#!/bin/bash
#SBATCH --job-name=hcg_depth_gen
#SBATCH --partition=dept_cpu
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/depth-gen-%A_%a.out
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
MODES=(residual parallel direct)
GEN_MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"
.venv/bin/python upstream/experiments/train_depth_generator.py \
  --tokens artifacts/development/residual_tokens --features artifacts/development/generator_features \
  --output artifacts/development/depth_generator --mode "$GEN_MODE" --epochs 20 --threads 8
.venv/bin/python upstream/experiments/evaluate_depth_generator.py \
  --tokens artifacts/development/residual_tokens --features artifacts/development/generator_features \
  --manifest artifacts/development/manifest.json --weights artifacts/tokenizer \
  --run artifacts/development/depth_generator --mode "$GEN_MODE"
