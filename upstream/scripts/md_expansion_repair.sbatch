#!/bin/bash
#SBATCH --partition=koes_gpu,dept_gpu
#SBATCH --constraint=M16|M20|M24|M40|M46
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --array=10
#SBATCH --job-name=hcg_md_expansion
#SBATCH --output=logs/hcg_md_repair-%A_%a.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
.venv/bin/python upstream/experiments/resume_md_expansion.py --index "$SLURM_ARRAY_TASK_ID"
