"""Frozen original ESM3 AA-only features, with a compatibility anchor."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hierarchical_VQ_VAE"))
from esm.models.esm3 import ESM3
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.utils.constants.esm3 import data_root
from esm.utils.encoding import tokenize_sequence
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha

RUN = Path("artifacts/sequence_condition_probe_v1")


@torch.no_grad()
def main():
    p = json.loads((RUN / "feature_requests.json").read_text())
    for path, digest in p["source_sha256"].items():
        if sha(path) != digest:
            raise RuntimeError("Changed AA feature extraction input")
    if (RUN / "feature_extraction.json").exists():
        raise RuntimeError("Feature extraction already complete")
    if not torch.cuda.is_available():
        raise RuntimeError("Feature extraction requires a GPU")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    special = dict(cls_token="<cls>", eos_token="<eos>", mask_token="<mask>", pad_token="<pad>")
    EsmSequenceTokenizer._get_token = lambda self, name: special[name]
    tokenizer = EsmSequenceTokenizer()
    if [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.mask_token_id, tokenizer.pad_token_id] != [0, 2, 32, 1]:
        raise RuntimeError("Sequence vocabulary changed")
    started = time.perf_counter()
    model = ESM3.from_pretrained("esm3_sm_open_v1", device="cuda").eval().requires_grad_(False)
    def extract(sequence):
        tokens = tokenize_sequence(sequence, tokenizer).to("cuda")[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            values = model(sequence_tokens=tokens).embeddings[0, 1:-1].float().cpu().numpy()
        if values.shape != (len(sequence), 1536) or not np.isfinite(values).all():
            raise RuntimeError("Nonfinite or misaligned sequence features")
        return values
    anchor = p["anchor"]
    if sha(anchor["path"]) != anchor["sha256"]:
        raise RuntimeError("Anchor cache changed")
    with np.load(anchor["path"], allow_pickle=False) as z:
        old = z["h"].astype(np.float32)
    values = extract(anchor["sequence"])
    relative = float(np.sqrt(np.mean((values-old)**2)) / max(1e-8, np.sqrt(np.mean(old**2))))
    cosine = float(np.mean((values*old).sum(-1) / np.maximum(1e-8, np.linalg.norm(values, axis=-1)*np.linalg.norm(old, axis=-1))))
    compatible = relative <= p["compatibility_gate"]["relative_RMSE_max"] and cosine >= p["compatibility_gate"]["mean_residue_cosine_min"]
    atomic_json(RUN / "anchor_check.json", dict(relative_RMSE=relative, mean_residue_cosine=cosine,
        max_absolute_difference=float(np.max(np.abs(values-old))), passed=compatible))
    if not compatible:
        raise RuntimeError("New extraction is incompatible with existing sequence features")
    directory = RUN / "new_sequence_features"
    directory.mkdir(exist_ok=True)
    rows = []
    for index, row in enumerate(p["requests"]):
        h = extract(row["sequence"])
        if index == 0 and not np.array_equal(h, extract(row["sequence"])):
            raise RuntimeError("Extraction is not deterministic on this GPU")
        path = directory / (row["id"] + ".npz")
        np.savez_compressed(path, h=h)
        rows.append(dict(row, path=str(path), sha256=sha(path)))
        print("PURE_SEQUENCE_FEATURE", row["id"], h.shape, flush=True)
    weights = data_root("esm3") / "data/weights/esm3_sm_open_v1.pth"
    atomic_json(RUN / "feature_extraction.json", dict(status="complete", features=rows,
        anchor_compatibility_passed=True, anchor_relative_RMSE=relative, anchor_cosine=cosine,
        request_sha256=sha(RUN / "feature_requests.json"), weight_path=str(weights), weight_sha256=sha(weights),
        gpu=torch.cuda.get_device_name(), parameter_dtype=str(next(model.parameters()).dtype),
        seconds=time.perf_counter()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        coordinates_used=False, structure_tokens_used=False,
        scope="Original ESM3 sequence_tokens-only forward; all other tracks omitted/default-masked. No structural encoder/decoder forward is requested."))
    print("PURE_SEQUENCE_CACHE_COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    main()
