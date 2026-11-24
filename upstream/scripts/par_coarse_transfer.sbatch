#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=ctf_par_transfer
#SBATCH --output=logs/ctf_par_transfer-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="/tmp/ctf-par-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="/tmp/ctf-par-cache-${SLURM_JOB_ID}"
if [[ "$1" == prepare ]]; then exec .venv/bin/python upstream/experiments/run_par_coarse_transfer.py --stage prepare; fi
if [[ "$1" == report ]]; then exec .venv/bin/python upstream/experiments/report_par_coarse_transfer.py; fi
trial_index=${SLURM_ARRAY_TASK_ID:?array required}
trial_seed=$((2026+trial_index%2))
if (( trial_index < 2 )); then trial_arm=position_only; else trial_arm=sequence; fi
if [[ "$1" == train ]]; then
  extra_args=();if [[ "${CTF_RESUME:-0}" == 1 ]]; then extra_args+=(--resume);fi
  exec .venv-par-preflight/bin/python upstream/experiments/run_par_coarse_transfer.py --stage train --seed "$trial_seed" --arm "$trial_arm" "${extra_args[@]}"
fi
exec .venv/bin/python upstream/experiments/run_par_coarse_transfer.py --stage evaluate --seed "$trial_seed" --arm "$trial_arm"
