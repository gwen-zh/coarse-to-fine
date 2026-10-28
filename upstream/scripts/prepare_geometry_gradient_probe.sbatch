#!/bin/bash
#SBATCH --job-name=hcg_geom_prepare
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/geometry-prepare-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
.venv/bin/python upstream/experiments/prepare_geometry_gradient_probe.py \
  --features artifacts/pdb_training_v2/fine_retention_features --corpus artifacts/pdb_training_v2 \
  --weights artifacts/tokenizer --output artifacts/development/geometry_gradient_probe
