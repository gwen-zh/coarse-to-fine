#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=ctf_AA_condition
#SBATCH --output=logs/ctf_AA_condition-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == train ]]; then
  condition_index=${SLURM_ARRAY_TASK_ID:?array required}
  condition_seed=$((2026+condition_index%2))
  condition_arm=AA_lookup
  if ((condition_index>=2)); then condition_arm=ESM3_context; fi
  exec "$python_bin" upstream/experiments/run_sequence_condition_probe.py --stage train --arm "$condition_arm" --seed "$condition_seed"
fi
exec "$python_bin" upstream/experiments/run_sequence_condition_probe.py --stage "$1"
