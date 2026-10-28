#!/bin/bash
#SBATCH --partition=dept_gpu,koes_gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=hcg_halffine_train
#SBATCH --output=logs/hcg_halffine_train-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
.venv/bin/python - <<'PY'
import json,time
from pathlib import Path
p=Path('artifacts/development/half_fine_tokenizer_v1/smoke.json')
for _ in range(12):
 if p.exists():break
 time.sleep(5)
assert json.loads(p.read_text())['status']=='passed'
PY
arms=(none local global)
.venv/bin/python upstream/experiments/train_half_fine_tokenizer.py --run artifacts/development/half_fine_tokenizer_v1 --stage train --arm "${arms[$SLURM_ARRAY_TASK_ID]}"
