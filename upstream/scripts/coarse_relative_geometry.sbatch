#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --constraint=32C
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --job-name=ctf_coarse_geom
#SBATCH --output=logs/ctf_coarse_geom-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == train ]]; then
  trial_index=${SLURM_ARRAY_TASK_ID:?array required}
  trial_seed=$((2026+trial_index%2))
  if (( trial_index < 2 )); then trial_arm=cartesian; else trial_arm=radial; fi
  extra_args=()
  if [[ "${CTF_RESUME:-0}" == 1 ]]; then extra_args+=(--resume); fi
  exec "$python_bin" upstream/experiments/run_coarse_relative_geometry.py --stage train --seed "$trial_seed" --arm "$trial_arm" "${extra_args[@]}"
fi
exec "$python_bin" upstream/experiments/run_coarse_relative_geometry.py --stage "$1"
