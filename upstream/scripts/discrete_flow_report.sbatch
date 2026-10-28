#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --job-name=hcg_dfm_report
#SBATCH --output=logs/hcg_dfm_report-%A_%a.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
.venv/bin/python - <<'READY'
import json,time
from pathlib import Path
r=Path('artifacts/development/discrete_flow_probe_v1');p=json.loads((r/'protocol.json').read_text())
files=[r/'decoded'/(x['id']+'.json') for x in p['targets']]
for attempt in range(31):
    if all(x.exists() for x in files):break
    if attempt==30:raise RuntimeError('Decoded cohort not all complete/visible')
    time.sleep(2)
READY
.venv/bin/python upstream/experiments/report_discrete_flow_probe.py --run artifacts/development/discrete_flow_probe_v1 
