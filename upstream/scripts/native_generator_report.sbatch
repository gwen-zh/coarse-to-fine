#!/bin/bash
#SBATCH --job-name=hcg_native_decode
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/native-report-%j.out
set -euo pipefail
TASK_ROOT=/net/galaxy/home/koes/zhangwen/coarsetofine
cd "$TASK_ROOT"
export HF_HOME="$TASK_ROOT/artifacts/hf_cache"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$TASK_ROOT/artifacts/matplotlib"
export XDG_CACHE_HOME="$TASK_ROOT/artifacts/cache"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
for NATIVE_LABEL in baseline adapted; do
  .venv/bin/python upstream/experiments/evaluate_depth_generator.py \
    --tokens artifacts/development/native_tokens --features artifacts/development/generator_features \
    --manifest artifacts/development/manifest.json --weights artifacts/tokenizer \
    --run artifacts/development/native_generator --mode direct --label "$NATIVE_LABEL"
done
.venv/bin/python upstream/experiments/report_native_generator.py --run artifacts/development/native_generator
