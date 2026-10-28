#!/bin/bash
#SBATCH --job-name=hcg_geom_grad_audit
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/geometry-gradient-audit-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
.venv/bin/python upstream/experiments/audit_geometry_gradients.py \
  --cache artifacts/pdb_training_v2/residual_condition_features \
  --training-run artifacts/development/geometry_condition --weights artifacts/tokenizer \
  --output artifacts/development/geometry_gradient_audit
