#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=ctf_latent_source
#SBATCH --output=logs/ctf_latent_source-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == diagnose ]]; then
  diagnostic_seed=$((2026+${SLURM_ARRAY_TASK_ID:?array required}))
  exec "$python_bin" upstream/experiments/run_latent_source_diagnostic.py --stage diagnose --seed "$diagnostic_seed"
fi
if [[ "$1" == report ]]; then
  exec "$python_bin" upstream/experiments/report_latent_source_diagnostic.py
fi
exec "$python_bin" upstream/experiments/run_latent_source_diagnostic.py --stage "$1"
