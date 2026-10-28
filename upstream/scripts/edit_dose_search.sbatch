#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --job-name=hcg_edit_dose
#SBATCH --output=logs/hcg_edit_dose-%A_%a.out
#SBATCH --array=0-15%12
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
.venv/bin/python upstream/experiments/run_edit_dose.py --run artifacts/development/edit_dose_v1 --chunk "$SLURM_ARRAY_TASK_ID"
