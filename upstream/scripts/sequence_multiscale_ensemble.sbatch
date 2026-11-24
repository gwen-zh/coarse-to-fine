#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --job-name=ctf_multiscale_ensemble
#SBATCH --output=logs/ctf_multiscale_ensemble-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == train ]]; then
  ensemble_index=${SLURM_ARRAY_TASK_ID:?array task required}
  ensemble_seed=$((2026+ensemble_index%2))
  ensemble_arms=(latent16_teacher centers3_teacher centers3_scheduled)
  ensemble_arm=${ensemble_arms[$((ensemble_index/2))]}
  exec "$python_bin" upstream/experiments/run_sequence_multiscale_ensemble.py --stage train --arm "$ensemble_arm" --seed "$ensemble_seed"
fi
exec "$python_bin" upstream/experiments/run_sequence_multiscale_ensemble.py --stage "$1"
