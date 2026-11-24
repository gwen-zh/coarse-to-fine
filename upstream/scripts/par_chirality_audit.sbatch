#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --job-name=ctf_par_chiral_audit
#SBATCH --output=logs/ctf_par_chirality_audit-%j.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="/tmp/ctf-par-chiral-${SLURM_JOB_ID}-mpl" XDG_CACHE_HOME="/tmp/ctf-par-chiral-${SLURM_JOB_ID}-cache"
.venv-par-preflight/bin/python upstream/experiments/audit_par_coarse_chirality.py --stage prepare
exec .venv-par-preflight/bin/python upstream/experiments/audit_par_coarse_chirality.py --stage audit
