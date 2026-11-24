#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --job-name=ctf_fine_two_state
#SBATCH --output=logs/ctf_fine_two_state-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == train ]];then
  pair_index=${SLURM_ARRAY_TASK_ID:?array required}
  pair_seed=$((2026+pair_index%2))
  pair_protein=1b2s_E
  if ((pair_index>=2));then pair_protein=1ef1_D;fi
  exec "$python_bin" upstream/experiments/run_fine_two_state.py --stage train --protein "$pair_protein" --seed "$pair_seed"
fi
exec "$python_bin" upstream/experiments/run_fine_two_state.py --stage "$1"
