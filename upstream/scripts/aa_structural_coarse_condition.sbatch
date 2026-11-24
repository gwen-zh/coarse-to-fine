#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --constraint=32C
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=ctf_coarse_prompt
#SBATCH --output=logs/ctf_coarse_prompt-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="/tmp/ctf-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="/tmp/ctf-cache-${SLURM_JOB_ID}"
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == run ]]; then
  trial_seed=$((2026+${SLURM_ARRAY_TASK_ID:?array required}))
  exec "$python_bin" upstream/experiments/run_aa_structural_coarse_condition.py --stage run --seed "$trial_seed"
fi
if [[ "$1" == report ]]; then
  exec "$python_bin" upstream/experiments/report_aa_structural_coarse_condition.py
fi
exec "$python_bin" upstream/experiments/run_aa_structural_coarse_condition.py --stage "$1"
