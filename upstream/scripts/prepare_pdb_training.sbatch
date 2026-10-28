#!/bin/bash
#SBATCH --job-name=hcg_pdb_training
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pdb-training-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
.venv/bin/python upstream/experiments/prepare_pdb_training.py \
  --benchmark-manifest artifacts/development/manifest.json --output artifacts/pdb_training_v2 \
  --target 1000 --candidates 5000
