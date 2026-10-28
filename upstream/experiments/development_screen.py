"""Baseline failure-rate screen with cached tokens, coordinates and per-sample seeds.

Uses only the diffusion network and structure decoder on GPU. The unrelated
ESM3 language model and function decoder are not instantiated for inference.
"""
import argparse
import json
from pathlib import Path
import random
import sys
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hierarchical_ConfGen"))
from esm.pretrained import ESM3_structure_decoder_v0
from esm.sdk.api import ESMProtein
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.tokenization.structure_tokenizer import StructureTokenizer
from esm.utils.decoding import decode_structure
from esm.utils.encoding import tokenize_sequence
from slm.geometry_feedback import assess_ca
from slm.utils.checkpoint_utils import load_state_dict_from_lightning_ckpt
from structural_metrics import ensemble_summary, summarize_backbone


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.samples < 1 or args.steps < 1:
        parser.error("samples and steps must be positive")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("This screen requires a Slurm GPU allocation")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    targets = manifest["selected"][:args.limit]
    model = load_state_dict_from_lightning_ckpt(args.checkpoint, device=device)
    decoder = ESM3_structure_decoder_v0(device="cpu").to(device).eval()
    sequence_tokenizer, structure_tokenizer = EsmSequenceTokenizer(), StructureTokenizer()
    summaries = []
    for target in targets:
        directory = args.output / target["id"]
        directory.mkdir(exist_ok=True)
        sequence = target["sequence"]
        length = len(sequence)
        seq = tokenize_sequence(sequence, sequence_tokenizer).to(device)[None]
        lengths = torch.tensor([length], device=device)
        references = []
        for path in target["paths"]:
            protein = ESMProtein.from_pdb(ROOT / path)
            if protein.sequence == sequence:
                references.append(protein.coordinates[:, 1].cpu().numpy())
        rows, ensemble = [], []
        for seed in args.seeds:
            for sample in range(args.samples):
                stem = f"seed{seed}_sample{sample}"
                npz = directory / f"{stem}.npz"
                metadata = directory / f"{stem}.json"
                if npz.exists() and metadata.exists():
                    coords = np.load(npz)["atom37"]
                    row = json.loads(metadata.read_text())
                    if row["steps"] != args.steps or coords.shape[0] != length:
                        raise ValueError(f"Cached sample protocol mismatch: {metadata}")
                else:
                    actual_seed = seed * 100000 + sample
                    seed_all(actual_seed)
                    started = perf_counter()
                    fine, coarse = model.ddpm_sample(seq, num_steps=args.steps,
                                                     lengths=lengths, return_coarse=True)
                    coords, _, _ = decode_structure(fine[0], decoder, structure_tokenizer, sequence)
                    coords = coords.cpu().numpy()
                    report = assess_ca(coords[:, 1])
                    row = dict(target=target["id"], tier=target["tier"], length=length,
                               seed=seed, sample=sample, actual_seed=actual_seed,
                               steps=args.steps, elapsed_seconds=perf_counter() - started,
                               network_calls=args.steps + 2,
                               peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                               repair_proxy=report.summary(),
                               metrics=summarize_backbone(coords, references))
                    np.savez_compressed(npz, fine=fine[0].cpu().numpy(),
                                        coarse=coarse[0].cpu().numpy(), atom37=coords)
                    atomic_json(metadata, row)
                    # Save one viewable structure per target, all coordinates in NPZ.
                    if seed == args.seeds[0] and sample == 0:
                        ESMProtein(sequence=sequence, coordinates=torch.from_numpy(coords)).to_pdb(directory / "example.pdb")
                rows.append(row)
                ensemble.append(coords[:, 1])
                print(json.dumps(row), flush=True)
        result = dict(target=target["id"], length=length, tier=target["tier"], samples=len(rows),
                      proxy_invalid_samples=sum(bool(r["repair_proxy"]["bad_residues"]) for r in rows),
                      upstream_clash_invalid_samples=sum(not r["metrics"].get("upstream_ca_clash_valid", False) for r in rows),
                      cn_outlier_samples=sum(bool(r["metrics"].get("backbone_cn_outliers", 1)) for r in rows),
                      network_calls=sum(r.get("network_calls", r["steps"] + 2) for r in rows),
                      elapsed_seconds=sum(r["elapsed_seconds"] for r in rows),
                      **ensemble_summary(ensemble, references))
        atomic_json(directory / "summary.json", result)
        summaries.append(result)
        atomic_json(args.output / "summary.json", dict(protocol=manifest["protocol"],
                    note=manifest["caveat"], targets=summaries))
    print("DEVELOPMENT_SCREEN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
