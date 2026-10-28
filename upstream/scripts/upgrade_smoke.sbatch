#!/bin/bash
#SBATCH --job-name=hcg_smoke
#SBATCH --partition=dept_gpu
#SBATCH --constraint=A100|L40|A4500
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/smoke-%j.out
set -euo pipefail
TASK_ROOT=/net/galaxy/home/koes/zhangwen/coarsetofine
cd "$TASK_ROOT/upstream"
export HF_HOME="$TASK_ROOT/artifacts/hf_cache"
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$TASK_ROOT/artifacts/matplotlib"
export XDG_CACHE_HOME="$TASK_ROOT/artifacts/cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
TASK_PYTHON="$TASK_ROOT/.venv/bin/python"
"$TASK_PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
"$TASK_PYTHON" -m unittest discover -s tests -v
cd hierarchical_ConfGen
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$TASK_PYTHON" slm/sample_esmdiff_var.py \
    --input data/targets/bpti --output "$TASK_ROOT/artifacts/smoke/baseline" \
    --ckpt "$TASK_ROOT/artifacts/mp_rank_00_model_states.pt" \
    --num_samples 2 --num_steps 5 --seed 0
for policy in geometry random global; do
    "$TASK_PYTHON" slm/sample_esmdiff_var.py \
        --input data/targets/bpti --output "$TASK_ROOT/artifacts/smoke/$policy" \
        --ckpt "$TASK_ROOT/artifacts/mp_rank_00_model_states.pt" \
        --num_samples 2 --num_steps 5 --seed 0 \
        --feedback_rounds 1 --feedback_steps 5 --feedback_radius 2 \
        --feedback_policy "$policy" --feedback_fixed_budget
done
