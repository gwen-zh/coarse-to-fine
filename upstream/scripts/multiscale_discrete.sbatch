#!/bin/bash
#SBATCH --partition=koes_gpu,dept_gpu
#SBATCH --constraint=M16|M20|M24|M40|M46
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=hcg_multiscale_discrete
#SBATCH --output=logs/hcg_multiscale_discrete-%A_%a.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
if [ "${1:-train}" = smoke ]; then
 .venv/bin/python upstream/experiments/train_multiscale_discrete.py --stage smoke
else
 .venv/bin/python upstream/experiments/train_multiscale_discrete.py --stage train --index "$SLURM_ARRAY_TASK_ID"
fi
