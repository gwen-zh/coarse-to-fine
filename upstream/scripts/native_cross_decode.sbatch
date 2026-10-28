#!/bin/bash
#SBATCH --job-name=hcg_cross_decode
#SBATCH --partition=dept_cpu
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/native-cross-decode-%A_%a.out
set -euo pipefail
TASK_ROOT=/net/galaxy/home/koes/zhangwen/coarsetofine
cd "$TASK_ROOT"
export HF_HOME="$TASK_ROOT/artifacts/hf_cache"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$TASK_ROOT/artifacts/matplotlib" XDG_CACHE_HOME="$TASK_ROOT/artifacts/cache"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CROSS_VARIANTS=(base_base adapt_base base_adapt adapt_adapt)
.venv/bin/python upstream/experiments/evaluate_depth_generator.py \
  --tokens artifacts/development/native_tokens --features artifacts/development/generator_features \
  --manifest artifacts/development/manifest.json --weights artifacts/tokenizer \
  --run artifacts/development/native_cross --mode direct --label "${CROSS_VARIANTS[$SLURM_ARRAY_TASK_ID]}"
