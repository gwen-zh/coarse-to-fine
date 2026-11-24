#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --constraint=32C
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=ctf_codec_interface
#SBATCH --output=logs/ctf_codec_interface-%j.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
exec "$python_bin" upstream/experiments/run_author_codec_interface.py --stage "${1:?stage required}"
