#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --job-name=hcg_full_PDB_frames
#SBATCH --output=logs/hcg_full_PDB_frames-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == train ]]; then
  full_index=${SLURM_ARRAY_TASK_ID:?array task required}
  full_seed=$((2026+full_index%2))
  full_arm=reference
  if ((full_index>=2)); then full_arm=local; fi
  exec "$python_bin" upstream/experiments/run_block_frame_full_pdb.py --stage train --arm "$full_arm" --seed "$full_seed"
fi
exec "$python_bin" upstream/experiments/run_block_frame_full_pdb.py --stage "$1"
