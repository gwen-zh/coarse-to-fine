#!/bin/bash
#SBATCH --job-name=hcg_native_joint
#SBATCH --partition=dept_gpu,koes_gpu
#SBATCH --constraint=A4500|A100|L40|A4000|RTX4000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/native-joint-%j.out
set -euo pipefail
TASK_ROOT=/net/galaxy/home/koes/zhangwen/coarsetofine
cd "$TASK_ROOT"
export HF_HOME="$TASK_ROOT/artifacts/hf_cache"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$TASK_ROOT/artifacts/matplotlib"
export XDG_CACHE_HOME="$TASK_ROOT/artifacts/cache"
export OMP_NUM_THREADS=1
# Stop immediately if either real coarse/fine backward fails in the smoke.
.venv/bin/python upstream/experiments/train_native_generator.py \
  --tokens artifacts/pdb_training_v2/native_tokens --diagnostic-tokens artifacts/development/native_tokens \
  --checkpoint artifacts/mp_rank_00_model_states.pt --limit 2 --max-steps 2 --epochs 1 \
  --output artifacts/development/native_generator_smoke
.venv/bin/python upstream/experiments/train_native_generator.py \
  --tokens artifacts/pdb_training_v2/native_tokens --diagnostic-tokens artifacts/development/native_tokens \
  --checkpoint artifacts/mp_rank_00_model_states.pt --epochs 5 \
  --output artifacts/development/native_generator
