#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=hcg_continuous_report
#SBATCH --output=logs/hcg_continuous_report-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python upstream/experiments/report_continuous_backbone_probe.py
