#!/bin/bash
#SBATCH --job-name=hcg_coarse_lora
#SBATCH --partition=koes_gpu,dept_gpu
#SBATCH --constraint=A4500|A100|L40|A4000|RTX4000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/coarse-lora-%j.out
set -euo pipefail
TASK_ROOT=/net/galaxy/home/koes/zhangwen/coarsetofine
cd "$TASK_ROOT"
export HF_HOME="$TASK_ROOT/artifacts/hf_cache"
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$TASK_ROOT/artifacts/matplotlib"
export XDG_CACHE_HOME="$TASK_ROOT/artifacts/cache"
export OMP_NUM_THREADS=1
.venv/bin/python upstream/experiments/train_pretrained_coarse.py \
  --tokens artifacts/development/residual_tokens --checkpoint artifacts/mp_rank_00_model_states.pt \
  --encoder-weights artifacts/tokenizer/encoder_checkpoint_110.pth \
  --output artifacts/development/pretrained_coarse_lora --epochs 5
