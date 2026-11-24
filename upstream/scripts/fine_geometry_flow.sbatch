#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --job-name=ctf_fine_geometry
#SBATCH --output=logs/ctf_fine_geometry-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == train ]]; then
  fine_index=${SLURM_ARRAY_TASK_ID:?array required}
  fine_seed=$((2026+fine_index%2))
  fine_arm=fine_flow_only
  if ((fine_index>=2));then fine_arm=fine_flow_geometry;fi
  exec "$python_bin" upstream/experiments/run_fine_geometry_flow.py --stage train --arm "$fine_arm" --seed "$fine_seed"
fi
if [[ "$1" == report ]];then
  exec "$python_bin" upstream/experiments/report_fine_geometry_flow.py
fi
exec "$python_bin" upstream/experiments/run_fine_geometry_flow.py --stage "$1"
