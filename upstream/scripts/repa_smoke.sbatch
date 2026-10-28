#!/bin/bash
#SBATCH --partition=dept_gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=hcg_repa_smoke
#SBATCH --output=logs/hcg_repa_smoke-%A_%a.out
#SBATCH --gres=gpu:1
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
.venv/bin/python upstream/experiments/train_repa_probe.py --run artifacts/development/repa_probe_v1 --arm repa --smoke
