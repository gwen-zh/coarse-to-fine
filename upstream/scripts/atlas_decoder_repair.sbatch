#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --job-name=hcg_decoder_repair
#SBATCH --output=logs/hcg_decoder_repair-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == train ]]; then
  task_index=${SLURM_ARRAY_TASK_ID:?array task required}
  repair_seed=$((2026+task_index%2))
  repair_arm=coordinate
  if ((task_index>=2)); then repair_arm=geometry; fi
  exec "$python_bin" upstream/experiments/run_atlas_decoder_repair.py --stage train --arm "$repair_arm" --seed "$repair_seed"
fi
exec "$python_bin" upstream/experiments/run_atlas_decoder_repair.py --stage "$1"
