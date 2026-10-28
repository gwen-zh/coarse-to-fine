#!/bin/bash
#SBATCH --job-name=hcg_fine_retention
#SBATCH --partition=dept_cpu
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fine-retention-%A_%a.out
set -euo pipefail
TASK_ROOT=/net/galaxy/home/koes/zhangwen/coarsetofine
cd "$TASK_ROOT"
export HF_HOME="$TASK_ROOT/artifacts/hf_cache"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$TASK_ROOT/artifacts/matplotlib" XDG_CACHE_HOME="$TASK_ROOT/artifacts/cache"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
RETENTION_VARIANTS=(plain retained)
RETENTION_VARIANT="${RETENTION_VARIANTS[$SLURM_ARRAY_TASK_ID]}"
.venv/bin/python upstream/experiments/train_fine_retention.py \
  --cache artifacts/pdb_training_v2/fine_retention_features \
  --anchors artifacts/development/generator_features --anchor-manifest artifacts/development/residual_tokens/manifest.json \
  --output artifacts/development/fine_retention --variant "$RETENTION_VARIANT"
.venv/bin/python upstream/experiments/select_fine_retention.py \
  --cache artifacts/pdb_training_v2/fine_retention_features --weights artifacts/tokenizer \
  --run artifacts/development/fine_retention --variant "$RETENTION_VARIANT"
.venv/bin/python upstream/experiments/evaluate_depth_generator.py \
  --tokens artifacts/development/native_tokens --features artifacts/development/generator_features \
  --manifest artifacts/development/manifest.json --weights artifacts/tokenizer \
  --run artifacts/development/fine_retention --mode direct --label "$RETENTION_VARIANT"
