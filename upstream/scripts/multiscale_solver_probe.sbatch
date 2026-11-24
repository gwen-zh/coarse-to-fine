#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --job-name=ctf_solver_probe
#SBATCH --output=logs/ctf_solver_probe-%A_%a.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin="${CTF_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$1" == evaluate ]]; then
  solver_index=${SLURM_ARRAY_TASK_ID:?array required}
  solver_seed=$((2026+solver_index%2))
  solvers=(euler16 heun16 euler64 heun64)
  exec "$python_bin" upstream/experiments/run_multiscale_solver_probe.py --stage evaluate --solver "${solvers[$((solver_index/2))]}" --seed "$solver_seed"
fi
exec "$python_bin" upstream/experiments/run_multiscale_solver_probe.py --stage "$1"
