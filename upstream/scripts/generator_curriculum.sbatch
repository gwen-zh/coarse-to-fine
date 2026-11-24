#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --job-name=ctf_PDB_curriculum
#SBATCH --output=logs/ctf_PDB_curriculum-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == train ]]; then
  curriculum_index=${SLURM_ARRAY_TASK_ID:?array required}
  curriculum_seed=$((2026+curriculum_index%2))
  curriculum_arm=interleaved
  if ((curriculum_index>=2)); then curriculum_arm=PDB_then_MD; fi
  exec "$python_bin" upstream/experiments/run_generator_curriculum.py --stage train --arm "$curriculum_arm" --seed "$curriculum_seed"
fi
exec "$python_bin" upstream/experiments/run_generator_curriculum.py --stage "$1"
