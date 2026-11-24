#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --job-name=ctf_fine_32
#SBATCH --output=logs/ctf_fine_32-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == train ]];then
  scaling_index=${SLURM_ARRAY_TASK_ID:?array required}
  scaling_seed=$((2026+scaling_index%2))
  scaling_arm=fixed_total
  if ((scaling_index>=2));then scaling_arm=matched_presentations;fi
  exec "$python_bin" upstream/experiments/run_fine_target_scaling.py --stage train --arm "$scaling_arm" --seed "$scaling_seed"
fi
exec "$python_bin" upstream/experiments/run_fine_target_scaling.py --stage "$1"
