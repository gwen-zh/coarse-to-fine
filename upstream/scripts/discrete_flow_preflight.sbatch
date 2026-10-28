#!/bin/bash
#SBATCH --partition=dept_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=hcg_dfm_preflight
#SBATCH --output=logs/hcg_dfm_preflight-%A_%a.out
#SBATCH --gres=gpu:1
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
.venv/bin/python upstream/experiments/sample_discrete_flow_probe.py --run artifacts/development/discrete_flow_probe_v1 --preflight-only
