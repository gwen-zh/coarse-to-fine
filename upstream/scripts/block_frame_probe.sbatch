#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:45:00
#SBATCH --job-name=hcg_block_frames
#SBATCH --output=logs/hcg_block_frames-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == train ]]; then
  probe_index=${SLURM_ARRAY_TASK_ID:?array task required}
  probe_seed=$((2026+probe_index%2))
  probe_arm=reference
  if ((probe_index>=2)); then probe_arm=local; fi
  exec "$python_bin" upstream/experiments/run_block_frame_probe.py --stage train --arm "$probe_arm" --seed "$probe_seed"
fi
exec "$python_bin" upstream/experiments/run_block_frame_probe.py --stage "$1"
