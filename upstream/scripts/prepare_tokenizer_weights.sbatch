#!/bin/bash
#SBATCH --job-name=hcg_vqweights
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/vqweights-%j.out
set -euo pipefail
cd /net/galaxy/home/koes/zhangwen/coarsetofine
mkdir -p artifacts/tokenizer
for name in encoder_checkpoint_110.pth decoder_checkpoint_110.pth; do
    if [[ "$name" == encoder* ]]; then
        expected=c100495eb3b13b41318e1c365bd6be58
        size=129087994
    else
        expected=0ac4f398181534b31aeff59e7cb7ea94
        size=2483418455
    fi
    target="artifacts/tokenizer/$name"
    if [[ ! -f "$target" ]] || [[ $(stat -c %s "$target") != "$size" ]]; then
        curl --silent --show-error -fL --retry 5 --continue-at - --output "$target" \
            "https://zenodo.org/api/records/19412545/files/$name/content"
    fi
    printf '%s  %s\n' "$expected" "$target" | md5sum --check
done
