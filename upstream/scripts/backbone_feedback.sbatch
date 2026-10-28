#!/bin/bash
#SBATCH --job-name=hcg_backbone
#SBATCH --partition=dept_gpu
#SBATCH --constraint=A4500|A100|L40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/backbone-%j.out
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
.venv/bin/python upstream/experiments/feedback_screen.py \
    --manifest artifacts/development/manifest.json --checkpoint artifacts/mp_rank_00_model_states.pt \
    --baseline artifacts/development/baseline --output artifacts/development/feedback_backbone \
    --rounds 2 --steps 5 --radius 2 --score-mode backbone
.venv/bin/python upstream/experiments/report_backbone.py --root artifacts/development
