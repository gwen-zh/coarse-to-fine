#!/bin/bash
#SBATCH --job-name=hcg_residual_cache
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/residual-condition-cache-%j.out
set -euo pipefail
TASK_ROOT=/net/galaxy/home/koes/zhangwen/coarsetofine
cd "$TASK_ROOT"
export HF_HOME="$TASK_ROOT/artifacts/hf_cache"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$TASK_ROOT/artifacts/matplotlib" XDG_CACHE_HOME="$TASK_ROOT/artifacts/cache"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
.venv/bin/python upstream/experiments/cache_residual_condition.py \
  --features artifacts/pdb_training_v2/fine_retention_features --source artifacts/development/native_generator \
  --checkpoint artifacts/mp_rank_00_model_states.pt --encoder artifacts/tokenizer/encoder_checkpoint_110.pth \
  --output artifacts/pdb_training_v2/residual_condition_features
