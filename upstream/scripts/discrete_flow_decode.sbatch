#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=hcg_dfm_decode
#SBATCH --output=logs/hcg_dfm_decode-%A_%a.out
#SBATCH --array=0-7
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
.venv/bin/python - <<'READY'
import json,time
from pathlib import Path
r=Path('artifacts/development/discrete_flow_probe_v1');p=json.loads((r/'protocol.json').read_text())
files=[r/('sampling_chunk'+str(i)+'.json') for i in range(p['chunks'])]
for attempt in range(31):
    if all(x.exists() for x in files):break
    if attempt==30:raise RuntimeError('Sampling chunks not all complete/visible')
    time.sleep(2)
READY
.venv/bin/python upstream/experiments/decode_discrete_flow_probe.py --run artifacts/development/discrete_flow_probe_v1 --chunk "$SLURM_ARRAY_TASK_ID"
