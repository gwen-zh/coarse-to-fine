#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --constraint=32C
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=ctf_decoder_angles
#SBATCH --output=logs/ctf_decoder_angles-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="/tmp/ctf-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="/tmp/ctf-cache-${SLURM_JOB_ID}"
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == train ]]; then
  trial_index=${SLURM_ARRAY_TASK_ID:?array required}
  trial_seed=$((2026+trial_index%2))
  if (( trial_index < 2 )); then trial_arm=continuation; else trial_arm=angular; fi
  extra_args=()
  if [[ "${CTF_RESUME:-0}" == 1 ]]; then extra_args+=(--resume); fi
  exec "$python_bin" upstream/experiments/run_decoder_angle_adaptation.py --stage train --seed "$trial_seed" --arm "$trial_arm" "${extra_args[@]}"
fi
if [[ "$1" == report ]]; then
  exec "$python_bin" upstream/experiments/report_decoder_angle_adaptation.py
fi
exec "$python_bin" upstream/experiments/run_decoder_angle_adaptation.py --stage "$1"
