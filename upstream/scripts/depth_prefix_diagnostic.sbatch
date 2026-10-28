#!/bin/bash
#SBATCH --job-name=hcg_prefix_diag
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/depth-prefix-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
.venv/bin/python upstream/experiments/diagnose_depth_prefix.py \
  --tokens artifacts/development/residual_tokens --cache artifacts/development/encoder_broad \
  --run artifacts/development/depth_generator
