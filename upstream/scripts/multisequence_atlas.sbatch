#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=ctf_multi_MD_data
#SBATCH --output=logs/ctf_multi_MD_data-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == download ]]; then
  exec "$python_bin" upstream/experiments/prepare_multisequence_atlas.py --stage download --index "${SLURM_ARRAY_TASK_ID:?array required}"
fi
exec "$python_bin" upstream/experiments/prepare_multisequence_atlas.py --stage "$1"
