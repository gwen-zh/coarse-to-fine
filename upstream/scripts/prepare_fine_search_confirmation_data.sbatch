#!/bin/bash
#SBATCH --job-name=hcg_new_test_data
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fine-search-new-data-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python upstream/experiments/prepare_fine_search_confirmation_data.py --output artifacts/fine_search_confirmation_v1
