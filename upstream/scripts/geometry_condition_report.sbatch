#!/bin/bash
#SBATCH --job-name=hcg_geom_cond_report
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/geometry-condition-report-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
.venv/bin/python upstream/experiments/report_residual_condition.py \
  --run artifacts/development/geometry_condition --baseline artifacts/development/native_generator/baseline --variant geometry
