#!/bin/bash
#SBATCH --job-name=hcg_vqdiag
#SBATCH --partition=dept_gpu
#SBATCH --constraint=A4500|A100|L40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/vqdiag-%j.out
set -euo pipefail
TASK_ROOT=/net/galaxy/home/koes/zhangwen/coarsetofine
cd "$TASK_ROOT"
export HF_HOME="$TASK_ROOT/artifacts/hf_cache"
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
.venv/bin/python upstream/experiments/tokenizer_diagnostics.py \
    --manifest artifacts/development/manifest.json \
    --weights artifacts/tokenizer --output artifacts/development/tokenizer
