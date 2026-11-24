#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --job-name=ctf_early_fine
#SBATCH --output=logs/ctf_early_fine-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == train ]];then
  trial_index=${SLURM_ARRAY_TASK_ID:?array required}
  trial_seed=$((2026+trial_index%2))
  trial_protein=1b2s_E
  if ((trial_index%4>=2));then trial_protein=1ef1_D;fi
  trial_arm=uniform
  if ((trial_index>=4));then trial_arm=early;fi
  exec "$python_bin" upstream/experiments/run_early_time_fine_flow.py --stage train --arm "$trial_arm" --protein "$trial_protein" --seed "$trial_seed"
fi
exec "$python_bin" upstream/experiments/run_early_time_fine_flow.py --stage "$1"
