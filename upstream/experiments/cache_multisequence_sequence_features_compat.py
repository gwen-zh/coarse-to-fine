"""Process-local device compatibility for the frozen failed extraction source.

The vendored ESM3 loader requires torch.device, not a device string. Preserve
the original request/source hashes and record this technical fix separately.
No features or training updates existed when the first extraction failed.
"""
from pathlib import Path
import json

import torch

import cache_multisequence_sequence_features as extraction
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha


def main():
    manifest_path = extraction.RUN / "technical_compatibility.json"
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["source_sha256"].items():
        if sha(name) != digest:
            raise RuntimeError("Changed technical compatibility source")
    original_loader = extraction.ESM3.from_pretrained
    def load_with_device(model_name, device="cpu"):
        return original_loader(model_name, device=torch.device(device))
    extraction.ESM3.from_pretrained = staticmethod(load_with_device)
    extraction.main()
    path = extraction.RUN / "feature_extraction.json"
    result = json.loads(path.read_text())
    result["technical_compatibility"] = dict(path=str(manifest_path), sha256=sha(manifest_path),
        change="Convert the loader device string to torch.device; model, AA requests and inference settings unchanged.")
    atomic_json(path, result)


if __name__ == "__main__":
    main()
