#!/bin/bash
#SBATCH --job-name=hcg_restore_tw88
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=00:05:00
#SBATCH --output=logs/restore-tw88-%j.out
set -euo pipefail
# Independent afterany fallback, in addition to the controller's EXIT trap.
scancel --signal=CONT --full 57549778
echo "RESTORE_FALLBACK_SENT job=57549778 time=$(date -Is)"
