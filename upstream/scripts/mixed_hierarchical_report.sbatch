#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=hcg_mixed_report
#SBATCH --output=logs/hcg_mixed_report-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python upstream/experiments/report_mixed_hierarchical_generator.py
