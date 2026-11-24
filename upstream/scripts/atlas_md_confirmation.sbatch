#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --job-name=hcg_R3_confirm
#SBATCH --output=logs/hcg_R3_confirm-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == evaluate ]]; then
  confirm_seed=$((2026+${SLURM_ARRAY_TASK_ID:?array task required}))
  exec "$python_bin" upstream/experiments/confirm_atlas_md.py --stage evaluate --seed "$confirm_seed"
fi
exec "$python_bin" upstream/experiments/confirm_atlas_md.py --stage "$1"
