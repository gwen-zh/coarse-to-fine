"""Additional dependency hash guard; the frozen capacity experiment is unchanged."""
import hashlib
import json
import runpy
import sys
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

if __name__=='__main__':
    guard=json.loads(Path('artifacts/codebook_capacity_v1/dependency_guard_v1.json').read_text())
    if sha(guard['protocol'])!=guard['protocol_sha256']:raise RuntimeError('Frozen parent protocol changed')
    for path,digest in guard['source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Guarded capacity dependency changed: '+path)
    script='upstream/experiments/run_codebook_capacity.py'
    sys.argv=[script]+sys.argv[1:]
    runpy.run_path(script,run_name='__main__')
