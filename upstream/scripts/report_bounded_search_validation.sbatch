#!/bin/bash
#SBATCH --job-name=hcg_search_report
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/search-validation-report-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python upstream/experiments/report_bounded_fine_search.py --run artifacts/development/bounded_fine_search_validation_v1
