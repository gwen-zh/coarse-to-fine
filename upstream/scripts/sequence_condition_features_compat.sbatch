#!/bin/bash
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=A4500|A100|L40|A4000
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --job-name=ctf_AA_features
#SBATCH --output=logs/ctf_AA_features-%j.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec "${CTF_PYTHON:-$PWD/.venv/bin/python}" upstream/experiments/cache_multisequence_sequence_features_compat.py
