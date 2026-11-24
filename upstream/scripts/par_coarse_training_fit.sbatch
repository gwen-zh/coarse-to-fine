#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=ctf_par_fit
#SBATCH --output=logs/ctf_par_fit-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="/tmp/ctf-par-fit-${SLURM_JOB_ID}-mpl" XDG_CACHE_HOME="/tmp/ctf-par-fit-${SLURM_JOB_ID}-cache"
if [[ "$1" == report ]]; then exec .venv/bin/python upstream/experiments/report_par_coarse_training_fit.py; fi
if [[ "$1" == audit || "$1" == prepare ]]; then exec .venv-par-preflight/bin/python upstream/experiments/run_par_coarse_training_fit.py --stage "$1"; fi
if [[ "$1" == bundle ]]; then
  .venv-par-preflight/bin/python upstream/experiments/run_par_coarse_training_fit.py --stage prepare
  for seed in 2026 2027; do
    for arm in position_only sequence; do
      .venv-par-preflight/bin/python upstream/experiments/run_par_coarse_training_fit.py --stage run --seed "$seed" --arm "$arm"
    done
  done
  exec .venv/bin/python upstream/experiments/report_par_coarse_training_fit.py
fi
idx=${SLURM_ARRAY_TASK_ID:?array task required}
seed=$((2026+idx))
for arm in position_only sequence; do
  .venv-par-preflight/bin/python upstream/experiments/run_par_coarse_training_fit.py --stage run --seed "$seed" --arm "$arm"
done
