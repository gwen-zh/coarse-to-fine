#!/bin/bash
#SBATCH --job-name=hcg_prefix_refine
#SBATCH --partition=dept_cpu
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prefix-refine-%A_%a.out
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
VARIANTS=(teacher fixed adaptive)
REFINE_VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
.venv/bin/python upstream/experiments/train_prefix_refinement.py \
  --tokens artifacts/development/residual_tokens --features artifacts/development/generator_features \
  --cache artifacts/development/encoder_broad --prefixes artifacts/development/refinement_prefixes \
  --source artifacts/development/depth_generator/residual --output artifacts/development/prefix_refinement \
  --variant "$REFINE_VARIANT" --epochs 20
.venv/bin/python upstream/experiments/evaluate_depth_generator.py \
  --tokens artifacts/development/residual_tokens --features artifacts/development/generator_features \
  --manifest artifacts/development/manifest.json --weights artifacts/tokenizer \
  --run artifacts/development/prefix_refinement --mode residual --label "$REFINE_VARIANT"
