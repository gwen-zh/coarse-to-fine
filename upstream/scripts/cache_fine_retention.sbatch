#!/bin/bash
#SBATCH --job-name=hcg_retention_cache
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/retention-cache-%j.out
set -euo pipefail
TASK_ROOT=/net/galaxy/home/koes/zhangwen/coarsetofine
cd "$TASK_ROOT"
export HF_HOME="$TASK_ROOT/artifacts/hf_cache"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$TASK_ROOT/artifacts/matplotlib" XDG_CACHE_HOME="$TASK_ROOT/artifacts/cache"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
.venv/bin/python upstream/experiments/cache_fine_retention.py \
  --tokens artifacts/pdb_training_v2/native_tokens --diagnostic-tokens artifacts/development/native_tokens \
  --source artifacts/development/native_generator --checkpoint artifacts/mp_rank_00_model_states.pt \
  --output artifacts/pdb_training_v2/fine_retention_features
