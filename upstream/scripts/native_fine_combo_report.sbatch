#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --job-name=hcg_nativefine_report
#SBATCH --output=logs/hcg_nativefine_report-%A_%a.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
.venv/bin/python - <<'READY'
import time
from pathlib import Path
r=Path('artifacts/development/native_fine_combo_v1')
files=[r/('chunk'+str(i)+'.json') for i in range(4)]
for attempt in range(31):
    if all(p.exists() for p in files):break
    if attempt==30:raise RuntimeError('All search chunks must finish')
    time.sleep(2)
READY
.venv/bin/python upstream/experiments/report_native_fine_combo.py --run artifacts/development/native_fine_combo_v1 
