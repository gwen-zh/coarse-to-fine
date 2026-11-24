#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=ctf_multiseq_flow
#SBATCH --output=logs/ctf_multiseq_flow-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == train ]]; then
  multi_index=${SLURM_ARRAY_TASK_ID:?array required}
  multi_seed=$((2026+multi_index%2))
  multi_arm=MD_only
  if ((multi_index>=2)); then multi_arm=MD_plus_PDB; fi
  exec "$python_bin" upstream/experiments/run_multisequence_ensemble_flow.py --stage train --arm "$multi_arm" --seed "$multi_seed"
fi
exec "$python_bin" upstream/experiments/run_multisequence_ensemble_flow.py --stage "$1"
