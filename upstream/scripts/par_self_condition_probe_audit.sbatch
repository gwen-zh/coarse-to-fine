#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=ctf_par_sc_audit
#SBATCH --output=logs/ctf_par_sc_audit-%j.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="/tmp/ctf-par-sc-audit-${SLURM_JOB_ID}-mpl" XDG_CACHE_HOME="/tmp/ctf-par-sc-audit-${SLURM_JOB_ID}-cache"
exec .venv/bin/python upstream/experiments/review_par_self_condition_probe.py --root artifacts/par_self_condition_probe_recovery_v1
