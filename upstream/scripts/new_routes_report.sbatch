#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=hcg_new_routes_report
#SBATCH --output=logs/hcg_new_routes_report-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
if [ "$1" = multiscale ]; then
 .venv/bin/python upstream/experiments/report_multiscale_discrete.py
else
 .venv/bin/python upstream/experiments/score_md_expansion.py
fi
