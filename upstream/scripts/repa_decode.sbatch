#!/bin/bash
#SBATCH --partition=dept_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=hcg_repa_decode
#SBATCH --output=logs/hcg_repa_decode-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
ARMS=(ce repa)
.venv/bin/python upstream/experiments/decode_repa_probe.py --run artifacts/development/repa_probe_v1 --arm "${ARMS[$SLURM_ARRAY_TASK_ID]}"
