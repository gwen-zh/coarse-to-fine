#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=0-5%6
#SBATCH --job-name=hcg_ae_boundary
#SBATCH --output=logs/hcg_ae_boundary-%A_%a.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python upstream/experiments/train_two_level_ae_boundary.py --stage train --index "$SLURM_ARRAY_TASK_ID"
