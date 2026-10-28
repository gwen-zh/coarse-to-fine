#!/bin/bash
#SBATCH --job-name=hcg_vqcontrol
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/vqcontrol-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
.venv/bin/python upstream/experiments/quantization_control.py \
    --manifest artifacts/development/manifest.json --encoder-cache artifacts/development/encoder_broad \
    --encoder-checkpoint artifacts/tokenizer/encoder_checkpoint_110.pth \
    --output artifacts/development/quantization_control --threads 2
