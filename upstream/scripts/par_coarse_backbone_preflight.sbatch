#!/bin/bash
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=ctf_par_preflight
#SBATCH --output=logs/ctf_par_preflight-%j.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="/tmp/ctf-par-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="/tmp/ctf-par-cache-${SLURM_JOB_ID}"
exec .venv-par-preflight/bin/python upstream/experiments/preflight_par_coarse_backbone.py
