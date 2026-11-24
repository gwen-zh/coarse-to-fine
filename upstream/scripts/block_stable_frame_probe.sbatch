#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:45:00
#SBATCH --job-name=hcg_stable_block_frames
#SBATCH --output=logs/hcg_stable_block_frames-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == train ]]; then
  stable_index=${SLURM_ARRAY_TASK_ID:?array task required}
  stable_seed=$((2026+stable_index%2))
  stable_arm=reference
  if ((stable_index>=2)); then stable_arm=local; fi
  exec "$python_bin" upstream/experiments/run_block_stable_frame_probe.py --stage train --arm "$stable_arm" --seed "$stable_seed"
fi
exec "$python_bin" upstream/experiments/run_block_stable_frame_probe.py --stage "$1"
