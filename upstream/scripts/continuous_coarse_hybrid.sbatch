#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=hcg_coarse_hybrid
#SBATCH --output=logs/hcg_coarse_hybrid-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python upstream/experiments/evaluate_continuous_coarse_hybrid.py
