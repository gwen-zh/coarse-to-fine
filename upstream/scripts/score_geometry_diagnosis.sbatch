#!/bin/bash
#SBATCH --job-name=hcg_score_geometry
#SBATCH --partition=dept_cpu
#SBATCH --array=0-3
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/score-geometry-%A_%a.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
.venv/bin/python upstream/experiments/score_geometry_diagnosis.py \
  --run artifacts/development/objective_geometry_diagnosis --weights artifacts/tokenizer --index "$SLURM_ARRAY_TASK_ID"
