#!/bin/bash
#SBATCH --job-name=hcg_native_tokens
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/native-tokens-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
.venv/bin/python upstream/experiments/export_native_training.py \
  --tokens artifacts/pdb_training_v2/tokens --encoder artifacts/tokenizer/encoder_checkpoint_110.pth \
  --output artifacts/pdb_training_v2/native_tokens
.venv/bin/python upstream/experiments/export_native_training.py \
  --tokens artifacts/development/residual_tokens --latent-cache artifacts/development/encoder_broad \
  --validation-only --encoder artifacts/tokenizer/encoder_checkpoint_110.pth \
  --output artifacts/development/native_tokens
