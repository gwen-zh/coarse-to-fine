#!/bin/bash
#SBATCH --partition=dept_gpu,koes_gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --job-name=hcg_half_decode
#SBATCH --output=logs/hcg_half_decode-%j.out
#SBATCH --gres=gpu:1
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
.venv/bin/python upstream/experiments/run_half_fine_decode_diagnostic.py --run artifacts/development/half_fine_decode_diagnostic_v1
.venv/bin/python upstream/experiments/report_half_fine_decode_diagnostic.py --run artifacts/development/half_fine_decode_diagnostic_v1
