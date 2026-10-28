#!/bin/bash
#SBATCH --job-name=hcg_vqdistill
#SBATCH --partition=dept_gpu
#SBATCH --constraint=A4500|A100|L40|2080Ti|Xp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/vqdistill-%j.out
set -euo pipefail
TASK_ROOT=/net/galaxy/home/koes/zhangwen/coarsetofine
cd "$TASK_ROOT"
export HF_HOME="$TASK_ROOT/artifacts/hf_cache"
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
.venv/bin/python upstream/experiments/tokenizer_distillation_control.py \
    --manifest artifacts/development/manifest.json --cache artifacts/development/encoder_broad \
    --weights artifacts/tokenizer --control artifacts/development/quantization_control \
    --output artifacts/development/tokenizer_distillation
