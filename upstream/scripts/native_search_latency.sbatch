#!/bin/bash
#SBATCH --job-name=hcg_native_latency
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/native-search-latency-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
export HF_HOME="$PWD/artifacts/hf_cache" HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export MPLCONFIGDIR="$PWD/artifacts/matplotlib" XDG_CACHE_HOME="$PWD/artifacts/cache"
.venv/bin/python upstream/experiments/benchmark_native_search_generator.py --run artifacts/development/native_search_latency_v1
for native_timing_index in 0 1 2 3 4 5 6 7; do
 .venv/bin/python upstream/experiments/benchmark_native_search_decoder.py --run artifacts/development/native_search_latency_v1 --index "$native_timing_index"
done
.venv/bin/python upstream/experiments/report_native_search_latency.py --run artifacts/development/native_search_latency_v1
