#!/bin/bash
#SBATCH --job-name=hcg_search_val
#SBATCH --partition=dept_cpu
#SBATCH --array=0-23%8
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bounded-search-val-%A_%a.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
.venv/bin/python upstream/experiments/bounded_fine_search_validation.py --run artifacts/development/bounded_fine_search_validation_v1 --index "$SLURM_ARRAY_TASK_ID"
