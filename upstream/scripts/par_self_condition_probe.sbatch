#!/bin/bash
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=ctf_par_sc_probe
#SBATCH --output=logs/ctf_par_sc_probe-%j.out
set -euo pipefail
cd "${CTF_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/third_party/par-protein:$PWD/third_party/par_preflight_openfold_v1:$PWD/upstream/experiments${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="/tmp/ctf-par-sc-${SLURM_JOB_ID}-mpl" XDG_CACHE_HOME="/tmp/ctf-par-sc-${SLURM_JOB_ID}-cache"
.venv-par-preflight/bin/python upstream/experiments/run_par_self_condition_probe.py --stage preflight
.venv-par-preflight/bin/python upstream/experiments/run_par_self_condition_probe.py --stage prepare
pids=()
for seed in 2029 2030; do
  for arm in native polar axial; do
    (
      export MPLCONFIGDIR="/tmp/ctf-par-sc-${SLURM_JOB_ID}-${seed}-${arm}-mpl" XDG_CACHE_HOME="/tmp/ctf-par-sc-${SLURM_JOB_ID}-${seed}-${arm}-cache"
      exec .venv-par-preflight/bin/python upstream/experiments/run_par_self_condition_probe.py --stage run --seed "$seed" --arm "$arm"
    ) &
    pids+=("$!")
  done
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then echo 'A fixed SC probe worker failed; preserve outputs and stop.' >&2; exit 1; fi
exec .venv-par-preflight/bin/python upstream/experiments/run_par_self_condition_probe.py --stage report
