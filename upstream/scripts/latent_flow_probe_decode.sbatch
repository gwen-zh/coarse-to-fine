#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=hcg_flow_decode
#SBATCH --output=logs/hcg_flow_decode-%A_%a.out
#SBATCH --array=0-1
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
latent_probe_modes=(fm regression)
.venv/bin/python upstream/experiments/evaluate_latent_flow_probe.py --run artifacts/development/latent_flow_probe_v1 --mode "${latent_probe_modes[$SLURM_ARRAY_TASK_ID]}"
