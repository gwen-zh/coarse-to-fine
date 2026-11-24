#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --job-name=ctf_par_signed_vol
#SBATCH --output=logs/ctf_par_signed_volume-%j.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="/tmp/ctf-par-volume-${SLURM_JOB_ID}-mpl" XDG_CACHE_HOME="/tmp/ctf-par-volume-${SLURM_JOB_ID}-cache"
.venv-par-preflight/bin/python upstream/experiments/run_par_signed_volume.py --stage preflight
.venv-par-preflight/bin/python upstream/experiments/run_par_signed_volume.py --stage prepare
pids=()
for arm in velocity_only velocity_signed_volume; do
  (
    export MPLCONFIGDIR="/tmp/ctf-par-volume-${SLURM_JOB_ID}-${arm}-mpl" XDG_CACHE_HOME="/tmp/ctf-par-volume-${SLURM_JOB_ID}-${arm}-cache"
    exec .venv-par-preflight/bin/python upstream/experiments/run_par_signed_volume.py --stage train --arm "$arm"
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then echo 'Objective arm failed;preserve all artifacts and stop.' >&2; exit 1; fi
exec .venv/bin/python upstream/experiments/run_par_signed_volume.py --stage report
