#!/bin/bash
#SBATCH --job-name=hcg_rvq_tokens
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/rvq-tokens-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
.venv/bin/python upstream/experiments/prepare_residual_tokens.py \
  --manifest artifacts/development/manifest.json --cache artifacts/development/encoder_broad \
  --control artifacts/development/quantization_control --weights artifacts/tokenizer \
  --run artifacts/development/residual_vq --output artifacts/development/residual_tokens
