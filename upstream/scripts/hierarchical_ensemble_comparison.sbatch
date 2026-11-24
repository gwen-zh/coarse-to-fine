#!/bin/bash
#SBATCH --partition=dept_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ctf_author_same
#SBATCH --output=logs/ctf_author_same-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
trial_seed=$((2026+${SLURM_ARRAY_TASK_ID:?array required}))
exec "${CTF_PYTHON:-$PWD/.venv/bin/python}" upstream/experiments/run_coarse_coverage_repair.py --stage author --seed "$trial_seed"
