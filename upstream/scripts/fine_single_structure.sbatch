#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --job-name=ctf_fine_single
#SBATCH --output=logs/ctf_fine_single-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == train ]];then
  single_index=${SLURM_ARRAY_TASK_ID:?array required}
  single_seed=$((2026+single_index%2))
  single_case=PDB_single
  if ((single_index>=2));then single_case=MD_single_frame;fi
  exec "$python_bin" upstream/experiments/run_fine_single_structure.py --stage train --case "$single_case" --seed "$single_seed"
fi
exec "$python_bin" upstream/experiments/run_fine_single_structure.py --stage "$1"
