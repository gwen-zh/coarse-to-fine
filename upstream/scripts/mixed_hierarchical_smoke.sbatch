#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=hcg_mixed_smoke
#SBATCH --output=logs/hcg_mixed_smoke-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python upstream/experiments/train_mixed_hierarchical_generator.py --stage smoke --index 0 --device cpu
