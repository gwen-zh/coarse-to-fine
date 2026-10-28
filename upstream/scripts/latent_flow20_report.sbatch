#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=hcg_flow20_report
#SBATCH --output=logs/hcg_flow20_report-%A_%a.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
.venv/bin/python - <<'READY'
import json,time
from pathlib import Path
r=Path('artifacts/development/latent_flow_probe_20epoch_v1')
protocol=json.loads((r/'protocol.json').read_text())
expected=[r/mode/'decoded'/(name+'.json') for mode in ['fm','regression'] for name in protocol['geometry_targets']]
for attempt in range(16):
    missing=[str(p) for p in expected if not p.exists()]
    if not missing:break
    if attempt==15:raise RuntimeError('Decode outputs not visible: '+str(missing))
    time.sleep(2)
READY
.venv/bin/python upstream/experiments/report_latent_flow_probe.py --run artifacts/development/latent_flow_probe_20epoch_v1
