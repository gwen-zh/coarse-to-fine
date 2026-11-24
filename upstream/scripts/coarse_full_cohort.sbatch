#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --job-name=ctf_coarse_full
#SBATCH --output=logs/ctf_coarse_full-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == train ]]; then
  trial_seed=$((2026+${SLURM_ARRAY_TASK_ID:?array required}))
  extra_args=()
  if [[ "${CTF_RESUME:-0}" == 1 ]]; then extra_args+=(--resume); fi
  exec "$python_bin" upstream/experiments/run_coarse_full_cohort.py --stage train --seed "$trial_seed" "${extra_args[@]}"
fi
exec "$python_bin" upstream/experiments/run_coarse_full_cohort.py --stage "$1"
