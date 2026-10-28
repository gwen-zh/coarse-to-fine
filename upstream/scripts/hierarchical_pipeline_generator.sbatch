#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --job-name=hcg_full_generator
#SBATCH --output=logs/hcg_full_generator-%A_%a.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python upstream/experiments/train_hierarchical_generator.py --index "$SLURM_ARRAY_TASK_ID" --device cpu
