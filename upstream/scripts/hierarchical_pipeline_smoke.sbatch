#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=hcg_hierarchy_smoke
#SBATCH --output=logs/hcg_hierarchy_smoke-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
.venv/bin/python upstream/experiments/train_hierarchical_tokenizer.py --index 0 --device cpu --smoke
.venv/bin/python upstream/experiments/train_hierarchical_generator.py --index 0 --device cpu --smoke
