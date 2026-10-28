#!/bin/bash
#SBATCH --job-name=hcg_residual_cond
#SBATCH --partition=dept_cpu
#SBATCH --array=0-1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/residual-condition-%A_%a.out
set -euo pipefail
TASK_ROOT=/net/galaxy/home/koes/zhangwen/coarsetofine
cd "$TASK_ROOT"
export HF_HOME="$TASK_ROOT/artifacts/hf_cache"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$TASK_ROOT/artifacts/matplotlib" XDG_CACHE_HOME="$TASK_ROOT/artifacts/cache"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
MID_MODES=(residual parallel)
MID_MODE="${MID_MODES[$SLURM_ARRAY_TASK_ID]}"
.venv/bin/python upstream/experiments/fit_residual_mid.py \
  --cache artifacts/pdb_training_v2/residual_condition_features --latents artifacts/pdb_training_v2/tokens \
  --output artifacts/development/residual_condition --mode "$MID_MODE"
.venv/bin/python upstream/experiments/train_residual_condition.py \
  --cache artifacts/pdb_training_v2/residual_condition_features --run artifacts/development/residual_condition --mode "$MID_MODE"
.venv/bin/python upstream/experiments/select_residual_condition.py \
  --run artifacts/development/residual_condition --weights artifacts/tokenizer --mode "$MID_MODE"
for MID_LABEL in "$MID_MODE" "${MID_MODE}_shuffled"; do
  .venv/bin/python upstream/experiments/evaluate_depth_generator.py \
    --tokens artifacts/development/native_tokens --features artifacts/development/generator_features \
    --manifest artifacts/development/manifest.json --weights artifacts/tokenizer \
    --run artifacts/development/residual_condition --mode direct --label "$MID_LABEL"
done
