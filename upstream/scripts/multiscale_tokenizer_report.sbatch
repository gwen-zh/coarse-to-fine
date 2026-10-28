#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --job-name=hcg_multiscale_report
#SBATCH --output=logs/hcg_multiscale_report-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python - <<'PY'
import time
from pathlib import Path
for _ in range(12):
 if all(Path('artifacts/development/multiscale_tokenizer_v1',arm,'complete.json').exists() for arm in ('local','global')):break
 time.sleep(5)
PY
.venv/bin/python upstream/experiments/report_multiscale_tokenizer.py --run artifacts/development/multiscale_tokenizer_v1
