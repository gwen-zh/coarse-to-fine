#!/bin/bash
#SBATCH --job-name=hcg_medium_adapt
#SBATCH --partition=dept_gpu
#SBATCH --constraint=A4500|A100|L40|A4000|RTX4000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/medium-adapt-%j.out
set -euo pipefail
TASK_ROOT=/net/galaxy/home/koes/zhangwen/coarsetofine
cd "$TASK_ROOT"
export HF_HOME="$TASK_ROOT/artifacts/hf_cache"
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$TASK_ROOT/artifacts/matplotlib"
export XDG_CACHE_HOME="$TASK_ROOT/artifacts/cache"
export OMP_NUM_THREADS=1
.venv/bin/python upstream/experiments/cache_generator_features.py \
  --manifest artifacts/development/manifest.json --cache artifacts/development/encoder_broad \
  --checkpoint artifacts/mp_rank_00_model_states.pt --output artifacts/development/generator_features
.venv/bin/python upstream/experiments/train_medium_adapter.py \
  --manifest artifacts/development/manifest.json --features artifacts/development/generator_features \
  --control artifacts/development/quantization_control --output artifacts/development/medium_adaptation
.venv/bin/python upstream/experiments/evaluate_medium_adapter.py \
  --manifest artifacts/development/manifest.json --features artifacts/development/generator_features \
  --control artifacts/development/quantization_control --adapters artifacts/development/medium_adaptation
