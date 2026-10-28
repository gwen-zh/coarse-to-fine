#!/bin/bash
#SBATCH --job-name=hcg_coarse_control
#SBATCH --partition=dept_gpu
#SBATCH --constraint=A4000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=logs/coarse-control-%j.out
set -euo pipefail
TASK_ROOT=/net/galaxy/home/koes/zhangwen/coarsetofine
cd "$TASK_ROOT"
export HF_HOME="$TASK_ROOT/artifacts/hf_cache"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$TASK_ROOT/artifacts/matplotlib"
export XDG_CACHE_HOME="$TASK_ROOT/artifacts/cache"
export OMP_NUM_THREADS=1
.venv/bin/python upstream/experiments/sample_pretrained_coarse_control.py \
  --tokens artifacts/development/residual_tokens --checkpoint artifacts/mp_rank_00_model_states.pt \
  --encoder-weights artifacts/tokenizer/encoder_checkpoint_110.pth \
  --adapted artifacts/development/pretrained_coarse_lora \
  --output artifacts/development/pretrained_coarse_control
