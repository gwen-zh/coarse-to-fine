#!/bin/bash
#SBATCH --job-name=hcg_prefix_oracle
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/prefix-oracle-%j.out
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
.venv/bin/python upstream/experiments/oracle_prefix_refinement.py \
  --tokens artifacts/development/residual_tokens --cache artifacts/development/encoder_broad \
  --manifest artifacts/development/manifest.json --weights artifacts/tokenizer \
  --source artifacts/development/depth_generator/residual --output artifacts/development/prefix_refinement/oracle
