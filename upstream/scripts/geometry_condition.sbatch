#!/bin/bash
#SBATCH --job-name=hcg_geom_condition
#SBATCH --partition=dept_cpu
#SBATCH --array=0-1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/geometry-condition-%A_%a.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
CONDITION_MODES=(no_geometry geometry)
CONDITION_MODE="${CONDITION_MODES[$SLURM_ARRAY_TASK_ID]}"
.venv/bin/python upstream/experiments/train_geometry_condition.py \
  --cache artifacts/pdb_training_v2/residual_condition_features \
  --mid-source artifacts/development/residual_condition/residual \
  --run artifacts/development/geometry_condition --weights artifacts/tokenizer --mode "$CONDITION_MODE"
.venv/bin/python upstream/experiments/select_residual_condition.py \
  --run artifacts/development/geometry_condition --weights artifacts/tokenizer --mode "$CONDITION_MODE"
for CONDITION_LABEL in "$CONDITION_MODE" "${CONDITION_MODE}_shuffled"; do
  .venv/bin/python upstream/experiments/evaluate_depth_generator.py \
    --tokens artifacts/development/native_tokens --features artifacts/development/generator_features \
    --manifest artifacts/development/manifest.json --weights artifacts/tokenizer \
    --run artifacts/development/geometry_condition --mode direct --label "$CONDITION_LABEL"
done
