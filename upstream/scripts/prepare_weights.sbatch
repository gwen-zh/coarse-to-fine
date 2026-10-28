#!/bin/bash
#SBATCH --job-name=hcg_weights
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/weights-%j.out
set -euo pipefail
TASK_ROOT=/net/galaxy/home/koes/zhangwen/coarsetofine
cd "$TASK_ROOT"
export HF_HOME="$TASK_ROOT/artifacts/hf_cache"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
mkdir -p artifacts
target=artifacts/mp_rank_00_model_states.pt
expected=f1f5d3e255283e0f89e2fbf47314e633
if [[ ! -f "$target" ]] || [[ $(stat -c %s "$target") != 5634910416 ]]; then
    curl -fL --retry 5 --continue-at - --output "$target" \
        https://zenodo.org/api/records/19412545/files/mp_rank_00_model_states.pt/content
fi
printf '%s  %s\n' "$expected" "$target" | md5sum --check
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="EvolutionaryScale/esm3-sm-open-v1",
                         allow_patterns=["data/*"], max_workers=4)
Path("artifacts/esm_snapshot.json").write_text(json.dumps({"path": path}, indent=2))
print("ESM3 data snapshot:", path)
PY
