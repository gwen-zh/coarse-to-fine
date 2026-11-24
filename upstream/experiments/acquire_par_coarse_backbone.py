"""Acquire one pinned public PAR checkpoint without touching existing weights."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request


ROOT=Path("artifacts/par_coarse_backbone_v1")
REVISION="7fb609346f5d9a2baa76add1f387aee1bdb8874b"
FILENAME="checkpoints/3scale_by_ratio_60m/200k.ckpt"
EXPECTED_SIZE=1405820768
EXPECTED_SHA256="e6fbed2fc5bcd33f502094f499dde4b5252e074a2c664a7853308cf49419ff65"


def main():
    directory=ROOT/"official_checkpoint";directory.mkdir(parents=True,exist_ok=True)
    target=directory/"PAR_3scale_by_ratio_60m_200k.ckpt"
    if target.exists():raise RuntimeError("Checkpoint already present; verify instead of overwriting")
    temporary=target.with_suffix(".part")
    if temporary.exists():raise RuntimeError("Inspect a partial download before a new attempt")
    url=f"https://huggingface.co/ByteDance-Seed/PAR/resolve/{REVISION}/{FILENAME}"
    digest=hashlib.sha256();size=0;last=0;started=time.perf_counter()
    with urllib.request.urlopen(url,timeout=120) as source,temporary.open("wb") as output:
        while chunk:=source.read(8*1024*1024):
            output.write(chunk);digest.update(chunk);size+=len(chunk)
            if size-last>=128*1024*1024:
                print("PAR_DOWNLOAD_MB",round(size/1024**2),flush=True);last=size
    if size!=EXPECTED_SIZE or digest.hexdigest()!=EXPECTED_SHA256:
        raise RuntimeError("Official checkpoint size/SHA256 verification failed")
    temporary.replace(target)
    metadata=dict(status="verified",repository="ByteDance-Seed/PAR",revision=REVISION,file=FILENAME,
        local_path=str(target),bytes=size,sha256=digest.hexdigest(),download_seconds=time.perf_counter()-started,
        source="Official public HuggingFace repository; fixed LFS object SHA256",training_started=False)
    (ROOT/"checkpoint_acquisition.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("PAR_CHECKPOINT_VERIFIED",size,digest.hexdigest(),flush=True)


if __name__=="__main__":main()
