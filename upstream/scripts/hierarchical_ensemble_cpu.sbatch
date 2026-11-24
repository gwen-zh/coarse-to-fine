#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --job-name=ctf_author_cpu
#SBATCH --output=logs/ctf_author_cpu-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == preflight ]]; then
  exec "$python_bin" upstream/experiments/run_hierarchical_ensemble_cpu.py --preflight
fi
trial_seed=$((2026+${SLURM_ARRAY_TASK_ID:?array required}))
exec "$python_bin" upstream/experiments/run_hierarchical_ensemble_cpu.py --seed "$trial_seed"
