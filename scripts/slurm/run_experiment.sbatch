#!/bin/bash
#SBATCH --job-name=ctf_experiment
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
project_root="${CTF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$project_root"
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
if [ "$#" -eq 0 ]; then
  echo 'Usage: sbatch [cluster options] scripts/slurm/run_experiment.sbatch scripts/demo_representation.py [arguments]' >&2
  exit 2
fi
exec "$python_bin" "$@"
