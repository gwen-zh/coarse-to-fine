#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --job-name=hcg_ae_author_recon
#SBATCH --output=logs/hcg_ae_author_recon-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
.venv/bin/python upstream/experiments/compare_two_level_ae_author.py --stage author
.venv/bin/python upstream/experiments/compare_two_level_ae_author.py --stage report
