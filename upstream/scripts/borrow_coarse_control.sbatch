#!/bin/bash
#SBATCH --job-name=hcg_borrow_control
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:25:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/borrow-control-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
BORROW_JOB=57550217
QUEUED_JOB=57566885
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
s=sys.stdin.read()
assert "JobState=RUNNING" in s and "NodeList=g122 " in s
assert "UserId=zhangwen(" in s and "JobName=tw88_mps4_b15_i20 " in s
assert "ArrayJobId=57547598 " in s and "ArrayTaskId=59 " in s
'
PAUSED=1
scancel --signal=STOP --full "$BORROW_JOB"
echo "PAUSE_SENT job=$BORROW_JOB time=$(date -Is)"
srun --jobid="$BORROW_JOB" --overlap --exact -N1 -n1 -c1 --gres=gpu:1 --time=00:20:00 \
  bash -c '
    set -euo pipefail
    cd /net/galaxy/home/koes/zhangwen/coarsetofine
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
    echo "BORROWED_CONTROL_STARTED host=$(hostname) time=$(date -Is)"
    ps -o pid,stat,comm -p 105227,105230,105232,105234,105240
    nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
    bash upstream/scripts/pretrained_coarse_control.sbatch
  ' &
STEP_PID=$!
wait "$STEP_PID"
STEP_PID=""
echo "BORROWED_CONTROL_COMPLETE time=$(date -Is)"
scancel "$QUEUED_JOB" || true
