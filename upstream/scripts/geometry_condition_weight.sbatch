#!/bin/bash
#SBATCH --job-name=hcg_geom_weight
#SBATCH --partition=dept_cpu
#SBATCH --array=0-1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/geometry-condition-weight-%A_%a.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
WEIGHT_PLAN=artifacts/development/geometry_weight_plan.json
GEOMETRY_WEIGHT=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidates"][int(sys.argv[2])]["weight"])' "$WEIGHT_PLAN" "$SLURM_ARRAY_TASK_ID")
CONDITION_RUN=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidates"][int(sys.argv[2])]["run"])' "$WEIGHT_PLAN" "$SLURM_ARRAY_TASK_ID")
.venv/bin/python upstream/experiments/train_geometry_condition.py \
  --cache artifacts/pdb_training_v2/residual_condition_features \
  --mid-source artifacts/development/residual_condition/residual \
  --run "$CONDITION_RUN" --weights artifacts/tokenizer --mode geometry \
  --geometry-weight "$GEOMETRY_WEIGHT" --weight-plan "$WEIGHT_PLAN"
.venv/bin/python upstream/experiments/select_residual_condition.py \
  --run "$CONDITION_RUN" --weights artifacts/tokenizer --mode geometry
for CONDITION_LABEL in geometry geometry_shuffled; do
  .venv/bin/python upstream/experiments/evaluate_depth_generator.py \
    --tokens artifacts/development/native_tokens --features artifacts/development/generator_features \
    --manifest artifacts/development/manifest.json --weights artifacts/tokenizer \
    --run "$CONDITION_RUN" --mode direct --label "$CONDITION_LABEL"
done
.venv/bin/python upstream/experiments/report_residual_condition.py \
  --run "$CONDITION_RUN" --baseline artifacts/development/native_generator/baseline --variant geometry
