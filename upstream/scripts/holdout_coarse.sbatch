#!/bin/bash
#SBATCH --job-name=hcg_holdout_coarse
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/holdout-coarse-%A_%a.out
#SBATCH --array=0-3
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
.venv/bin/python upstream/experiments/sample_holdout_coarse.py --run artifacts/holdout_confirmation_v1 --index "$SLURM_ARRAY_TASK_ID"
.venv/bin/python upstream/experiments/decode_holdout_coarse.py --run artifacts/holdout_confirmation_v1 --index "$SLURM_ARRAY_TASK_ID"
