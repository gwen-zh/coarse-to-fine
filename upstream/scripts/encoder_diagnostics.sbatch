#!/bin/bash
#SBATCH --job-name=hcg_encdiag
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/encdiag-%j.out
set -euo pipefail
TASK_ROOT=/net/galaxy/home/koes/zhangwen/coarsetofine
cd "$TASK_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
.venv/bin/python upstream/experiments/tokenizer_diagnostics.py \
    --manifest artifacts/development/manifest.json --weights artifacts/tokenizer \
    --output artifacts/development/encoder --encoder-only --device cpu
