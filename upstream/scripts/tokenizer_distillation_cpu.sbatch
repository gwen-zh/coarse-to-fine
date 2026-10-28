#!/bin/bash
#SBATCH --job-name=hcg_vqdistillcpu
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/vqdistillcpu-%j.out
set -euo pipefail
TASK_ROOT=/net/galaxy/home/koes/zhangwen/coarsetofine
cd "$TASK_ROOT"
export HF_HOME="$TASK_ROOT/artifacts/hf_cache"
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
.venv/bin/python upstream/experiments/tokenizer_distillation_control.py \
    --manifest artifacts/development/manifest.json --cache artifacts/development/encoder_broad \
    --weights artifacts/tokenizer --control artifacts/development/quantization_control \
    --output artifacts/development/tokenizer_distillation --device cpu
