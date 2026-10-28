"""Export tested dictionary/embedding pairs as compact tokenizer component patches."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--distillation", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.distillation / "summary.json"
    summary = json.loads(summary_path.read_text())
    if len(summary["targets"]) != summary["validation_proteins"]:
        raise ValueError("Reconstruction validation is incomplete")
    digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    with np.load(args.control / "dictionaries.npz") as codes, np.load(args.distillation / "medium_embeddings.npz") as tables:
        for name in tables.files:
            code_key = "released512" if name == "released512_distilled" else name
            patch = dict(format_version=1, level="medium", method=name,
                         role="negative_control" if name == "released512_distilled" else "candidate",
                         changes_token_ids=name != "released512_distilled",
                         requires_generator_retraining=name != "released512_distilled",
                         train_proteins=summary["train_proteins"], validation_proteins=summary["validation_proteins"],
                         validation_summary_sha256=digest,
                         encoder_weight=torch.from_numpy(codes[code_key].copy()),
                         decoder_weight=torch.from_numpy(tables[name].copy()))
            prefix = "control" if name == "released512_distilled" else "medium"
            destination = args.distillation / f"{prefix}_{name}.pt"
            torch.save(patch, destination)
            print(destination, flush=True)


if __name__ == "__main__":
    main()
