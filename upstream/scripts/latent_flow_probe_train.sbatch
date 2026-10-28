#!/bin/bash
#SBATCH --partition=dept_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --job-name=hcg_flow_train
#SBATCH --output=logs/hcg_flow_train-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
latent_probe_modes=(fm regression)
.venv/bin/python upstream/experiments/train_latent_flow_probe.py --run artifacts/development/latent_flow_probe_v1 --mode "${latent_probe_modes[$SLURM_ARRAY_TASK_ID]}"
