#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=hcg_coarse_geometry
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python upstream/experiments/run_coarse_geometry.py --stage "$1" --index "${SLURM_ARRAY_TASK_ID:-0}"
