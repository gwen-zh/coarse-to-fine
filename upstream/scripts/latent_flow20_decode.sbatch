#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=hcg_flow20_decode
#SBATCH --output=logs/hcg_flow20_decode-%A_%a.out
#SBATCH --array=0-1
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
latent_probe_modes=(fm regression)
latent_probe_mode="${latent_probe_modes[$SLURM_ARRAY_TASK_ID]}"
.venv/bin/python - "$latent_probe_mode" <<'READY'
import json,sys,time
from pathlib import Path
r=Path('artifacts/development/latent_flow_probe_20epoch_v1');mode=sys.argv[1]
protocol=json.loads((r/'protocol.json').read_text())
expected=[r/mode/'summary.json']+[r/mode/'predictions'/(name+'.npz') for name in protocol['geometry_targets']]
for attempt in range(16):
    if all(path.exists() for path in expected):break
    if attempt==15:raise RuntimeError('Training outputs not visible for '+mode)
    time.sleep(2)
READY
.venv/bin/python upstream/experiments/evaluate_latent_flow_probe.py --run artifacts/development/latent_flow_probe_20epoch_v1 --mode "${latent_probe_modes[$SLURM_ARRAY_TASK_ID]}"
