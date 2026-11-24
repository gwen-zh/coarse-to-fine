#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --job-name=ctf_par_capacity
#SBATCH --output=logs/ctf_par_capacity-%j.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="/tmp/ctf-par-capacity-${SLURM_JOB_ID}-mpl" XDG_CACHE_HOME="/tmp/ctf-par-capacity-${SLURM_JOB_ID}-cache"
if [[ "$1" == report ]]; then exec .venv/bin/python upstream/experiments/report_par_coarse_capacity.py; fi
if [[ "$1" == preflight ]]; then exec .venv-par-preflight/bin/python upstream/experiments/run_par_coarse_capacity.py --stage preflight; fi
if [[ "$1" != bundle ]]; then echo 'Expected bundle/preflight/report' >&2; exit 2; fi
.venv-par-preflight/bin/python upstream/experiments/run_par_coarse_capacity.py --stage preflight
.venv-par-preflight/bin/python upstream/experiments/run_par_coarse_capacity.py --stage prepare
.venv-par-preflight/bin/python upstream/experiments/run_par_coarse_capacity.py --stage diagnose
.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, 'upstream/experiments')
from run_par_coarse_capacity import state_update
state_update(status='capacity_arms_starting', all_models_running=False)
PY
pids=()
for arm in adapter_only adapter_flow_lora; do
  (
    export MPLCONFIGDIR="/tmp/ctf-par-capacity-${SLURM_JOB_ID}-${arm}-mpl" XDG_CACHE_HOME="/tmp/ctf-par-capacity-${SLURM_JOB_ID}-${arm}-cache"
    exec .venv-par-preflight/bin/python upstream/experiments/run_par_coarse_capacity.py --stage train --arm "$arm"
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then echo 'Capacity arm failed;preserve artifacts and stop.' >&2; exit 1; fi
exec .venv/bin/python upstream/experiments/report_par_coarse_capacity.py
