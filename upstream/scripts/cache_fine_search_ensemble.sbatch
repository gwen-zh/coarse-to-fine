#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=hcg_ensemble_cache
#SBATCH --output=logs/hcg_ensemble_cache-%A_%a.out
#SBATCH --array=0-3
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
.venv/bin/python upstream/experiments/cache_fine_search_ensemble.py --run artifacts/development/bounded_fine_search_ensemble_v1 --chunk "$SLURM_ARRAY_TASK_ID"
