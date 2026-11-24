#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --job-name=ctf_par_volume_rep
#SBATCH --output=logs/ctf_par_volume_rep-%j.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="/tmp/ctf-par-volume-rep-${SLURM_JOB_ID}-mpl" XDG_CACHE_HOME="/tmp/ctf-par-volume-rep-${SLURM_JOB_ID}-cache"
.venv-par-preflight/bin/python upstream/experiments/run_par_signed_volume_replication.py --stage preflight
.venv-par-preflight/bin/python upstream/experiments/run_par_signed_volume_replication.py --stage prepare
.venv-par-preflight/bin/python upstream/experiments/run_par_signed_volume_replication.py --stage anchor
pids=()
for seed in 2027 2028; do
  for arm in velocity_only velocity_signed_volume; do
    (
      export MPLCONFIGDIR="/tmp/ctf-par-volume-rep-${SLURM_JOB_ID}-${seed}-${arm}-mpl" XDG_CACHE_HOME="/tmp/ctf-par-volume-rep-${SLURM_JOB_ID}-${seed}-${arm}-cache"
      exec .venv-par-preflight/bin/python upstream/experiments/run_par_signed_volume_replication.py --stage train --seed "$seed" --arm "$arm"
    ) &
    pids+=("$!")
  done
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then echo 'A replication run failed;preserve all outputs and stop.' >&2; exit 1; fi
exec .venv/bin/python upstream/experiments/run_par_signed_volume_replication.py --stage report
