#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --job-name=hcg_ensemble_probe
#SBATCH --output=logs/hcg_ensemble_probe-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
seed=$((2026+${SLURM_ARRAY_TASK_ID:-0}))
exec "$python_bin" upstream/experiments/run_hierarchical_ensemble_probe.py --stage "$1" --seed "$seed"
