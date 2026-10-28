#!/bin/bash
#SBATCH --job-name=hcg_restore_control
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=00:05:00
#SBATCH --output=logs/restore-control-%j.out
set -euo pipefail
scancel --signal=CONT --full 57550217
echo "RESTORE_FALLBACK_SENT job=57550217 time=$(date -Is)"
