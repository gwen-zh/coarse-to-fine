#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --job-name=hcg_native_md_flow
#SBATCH --output=logs/hcg_native_md_flow-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == train ]]; then
  flow_seed=$((2026+${SLURM_ARRAY_TASK_ID:?array task required}))
  exec "$python_bin" upstream/experiments/run_atlas_ensemble_flow.py --stage train --seed "$flow_seed"
fi
exec "$python_bin" upstream/experiments/run_atlas_ensemble_flow.py --stage "$1"
