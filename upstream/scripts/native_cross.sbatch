#!/bin/bash
#SBATCH --job-name=hcg_cross_infer
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/native-cross-%j.out
set -euo pipefail
TASK_ROOT=/net/galaxy/home/koes/zhangwen/coarsetofine
cd "$TASK_ROOT"
export HF_HOME="$TASK_ROOT/artifacts/hf_cache"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$TASK_ROOT/artifacts/matplotlib" XDG_CACHE_HOME="$TASK_ROOT/artifacts/cache"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
.venv/bin/python upstream/experiments/native_stage_cross.py \
  --source artifacts/development/native_generator --tokens artifacts/development/native_tokens \
  --checkpoint artifacts/mp_rank_00_model_states.pt \
  --output artifacts/development/native_cross --device cpu --threads 16
