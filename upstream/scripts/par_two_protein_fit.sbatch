#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=ctf_par_two_fit
#SBATCH --output=logs/ctf_par_two_fit-%j.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="/tmp/ctf-par-two-${SLURM_JOB_ID}-mpl" XDG_CACHE_HOME="/tmp/ctf-par-two-${SLURM_JOB_ID}-cache"
if [[ "$1" == preflight || "$1" == prepare ]]; then exec .venv-par-preflight/bin/python upstream/experiments/run_par_two_protein_fit.py --stage "$1"; fi
if [[ "$1" == report ]]; then exec .venv/bin/python upstream/experiments/report_par_two_protein_fit.py; fi
if [[ "$1" != bundle ]]; then echo 'Expected bundle/preflight/prepare/report' >&2; exit 2; fi
.venv-par-preflight/bin/python upstream/experiments/run_par_two_protein_fit.py --stage preflight
.venv-par-preflight/bin/python upstream/experiments/run_par_two_protein_fit.py --stage prepare
pids=()
for arm in position_beta sequence_beta position_uniform sequence_uniform; do
  (
    export MPLCONFIGDIR="/tmp/ctf-par-two-${SLURM_JOB_ID}-${arm}-mpl" XDG_CACHE_HOME="/tmp/ctf-par-two-${SLURM_JOB_ID}-${arm}-cache"
    exec .venv-par-preflight/bin/python upstream/experiments/run_par_two_protein_fit.py --stage train --arm "$arm"
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then echo 'At least one fixed arm failed; preserve all artifacts and stop.' >&2; exit 1; fi
exec .venv/bin/python upstream/experiments/report_par_two_protein_fit.py
