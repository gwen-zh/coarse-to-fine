#!/bin/bash
#SBATCH --job-name=hcg_prefix_report
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/prefix-report-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
.venv/bin/python upstream/experiments/report_prefix_refinement.py \
  --run artifacts/development/prefix_refinement --source artifacts/development/depth_generator/residual
