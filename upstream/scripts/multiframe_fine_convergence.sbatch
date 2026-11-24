#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=ctf_fine_convergence
#SBATCH --output=logs/ctf_fine_convergence-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == train ]];then
  trial_index=${SLURM_ARRAY_TASK_ID:?array required}
  trial_seed=$((2026+trial_index%2))
  trial_arm=uniform
  if ((trial_index>=2));then trial_arm=early;fi
  exec "$python_bin" upstream/experiments/run_multiframe_fine_convergence.py --stage train --arm "$trial_arm" --seed "$trial_seed"
fi
exec "$python_bin" upstream/experiments/run_multiframe_fine_convergence.py --stage "$1"
