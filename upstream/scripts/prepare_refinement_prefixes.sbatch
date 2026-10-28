#!/bin/bash
#SBATCH --job-name=hcg_prefix_cache
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/prefix-cache-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
.venv/bin/python upstream/experiments/prepare_refinement_prefixes.py \
  --tokens artifacts/development/residual_tokens --features artifacts/development/generator_features \
  --source artifacts/development/depth_generator/residual --output artifacts/development/refinement_prefixes
