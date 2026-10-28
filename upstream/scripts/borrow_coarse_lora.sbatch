#!/bin/bash
#SBATCH --job-name=hcg_borrow_coarse
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:20:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/borrow-coarse-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
BORROW_JOB=57549778
QUEUED_JOB=57551352
STEP_PID=""
PAUSED=0
finish() {
  status=$?
  trap - EXIT TERM INT HUP
  if [[ -n "$STEP_PID" ]] && kill -0 "$STEP_PID" 2>/dev/null; then
    kill -TERM "$STEP_PID" 2>/dev/null || true
    wait "$STEP_PID" || true
  fi
  if [[ "$PAUSED" == 1 ]]; then
    scancel --signal=CONT --full "$BORROW_JOB" || true
    echo "RESTORE_SENT job=$BORROW_JOB status=$status time=$(date -Is)"
  fi
  exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM INT HUP
scontrol show job "$BORROW_JOB" -o | python -c '
import sys
s = sys.stdin.read()
assert "JobState=RUNNING" in s and "NodeList=g122 " in s
assert "ArrayJobId=57547597 " in s and "ArrayTaskId=39 " in s
'
PAUSED=1
scancel --signal=STOP --full "$BORROW_JOB"
echo "PAUSE_SENT job=$BORROW_JOB time=$(date -Is)"
srun --jobid="$BORROW_JOB" --overlap --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=1 --gres=gpu:1 --time=01:15:00 \
  bash -c '
    set -euo pipefail
    cd /net/galaxy/home/koes/zhangwen/coarsetofine
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
    echo "BORROWED_GPU_STARTED host=$(hostname) time=$(date -Is)"
    bash upstream/scripts/pretrained_coarse_lora.sbatch
  ' &
STEP_PID=$!
wait "$STEP_PID"
STEP_PID=""
echo "BORROWED_COARSE_EXPERIMENT_COMPLETE time=$(date -Is)"
scancel "$QUEUED_JOB" || true
