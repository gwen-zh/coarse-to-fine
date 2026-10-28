#!/bin/bash
#SBATCH --partition=koes_gpu,dept_gpu
#SBATCH --constraint=M16|M20|M24|M40|M46
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --job-name=hcg_gpu_full_latency
#SBATCH --output=logs/hcg_gpu_full_latency-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
.venv/bin/python upstream/experiments/run_gpu_pipeline_latency.py --run artifacts/development/gpu_pipeline_latency_v1
