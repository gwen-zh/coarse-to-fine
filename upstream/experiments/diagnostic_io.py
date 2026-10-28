"""Small reporting I/O without importing model or CUDA dependencies."""
import json
from pathlib import Path


def atomic_json(path,value):
    path=Path(path)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)
