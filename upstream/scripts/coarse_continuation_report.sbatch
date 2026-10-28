#!/bin/bash
#SBATCH --job-name=hcg_coarse_report
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/coarse-report-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
.venv/bin/python upstream/experiments/report_coarse_continuation.py \
  --run artifacts/development/pretrained_coarse_continuation \
  --initial-control artifacts/development/depth_generator/residual \
  --adaptive-control artifacts/development/prefix_refinement/adaptive
