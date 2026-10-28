#!/bin/bash
#SBATCH --job-name=hcg_coarse_decode
#SBATCH --partition=dept_cpu
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/coarse-decode-%A_%a.out
set -euo pipefail
TASK_ROOT=/net/galaxy/home/koes/zhangwen/coarsetofine
cd "$TASK_ROOT"
export HF_HOME="$TASK_ROOT/artifacts/hf_cache"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$TASK_ROOT/artifacts/matplotlib"
export XDG_CACHE_HOME="$TASK_ROOT/artifacts/cache"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
LABELS=(initial adaptive)
CHECKPOINTS=(artifacts/development/depth_generator/residual/best.pt artifacts/development/prefix_refinement/adaptive/best.pt)
RUN_DIR="${CONTINUATION_RUN_DIR:-artifacts/development/pretrained_coarse_continuation}"
COARSE_DIR="${CONTINUATION_COARSE_DIR:-artifacts/development/pretrained_coarse_lora}"
CONT_LABEL="${LABELS[$SLURM_ARRAY_TASK_ID]}"
.venv/bin/python upstream/experiments/continue_pretrained_coarse.py \
  --tokens artifacts/development/residual_tokens --features artifacts/development/generator_features \
  --coarse "$COARSE_DIR" \
  --refiner "${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}" --output "$RUN_DIR/$CONT_LABEL"
.venv/bin/python upstream/experiments/evaluate_depth_generator.py \
  --tokens artifacts/development/residual_tokens --features artifacts/development/generator_features \
  --manifest artifacts/development/manifest.json --weights artifacts/tokenizer \
  --run "$RUN_DIR" --mode residual --label "$CONT_LABEL"
