#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --job-name=ctf_fine_eight
#SBATCH --output=logs/ctf_fine_eight-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == train ]];then
  few_index=${SLURM_ARRAY_TASK_ID:?array required}
  few_seed=$((2026+few_index%2))
  few_arm=fixed_total
  if ((few_index>=2));then few_arm=matched_presentations;fi
  exec "$python_bin" upstream/experiments/run_fine_few_targets.py --stage train --arm "$few_arm" --seed "$few_seed"
fi
exec "$python_bin" upstream/experiments/run_fine_few_targets.py --stage "$1"
