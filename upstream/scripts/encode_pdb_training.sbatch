#!/bin/bash
#SBATCH --job-name=hcg_pdb_encode
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pdb-encode-%j.out
set -euo pipefail
TASK_ROOT=/net/galaxy/home/koes/zhangwen/coarsetofine
cd "$TASK_ROOT"
export HF_HOME="$TASK_ROOT/artifacts/hf_cache"
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$TASK_ROOT/artifacts/matplotlib"
export XDG_CACHE_HOME="$TASK_ROOT/artifacts/cache"
export OMP_NUM_THREADS=8
.venv/bin/python upstream/experiments/encode_pdb_training.py \
  --corpus artifacts/pdb_training_v2 --weights artifacts/tokenizer \
  --bridges artifacts/development/residual_tokens --output artifacts/pdb_training_v2/tokens
