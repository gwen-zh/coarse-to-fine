"""Paired, fixed-budget feedback comparisons starting from cached baseline tokens."""
import argparse
import json
from pathlib import Path
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
from esm.utils.constants import esm3 as C
from esm.utils.decoding import decode_structure
from esm.utils.encoding import tokenize_sequence
from slm.geometry_feedback import geometry_feedback, assess_ca, assess_backbone, backbone_improves
from slm.utils.checkpoint_utils import load_state_dict_from_lightning_ckpt
from development_screen import atomic_json, seed_all
from structural_metrics import ensemble_summary, summarize_backbone


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--policies", nargs="+", choices=["geometry", "random", "global"],
                        default=["geometry", "random", "global"])
    parser.add_argument("--adaptive-budget", action="store_true")
    parser.add_argument("--score-mode", choices=["ca", "backbone"], default="ca")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    device = torch.device("cuda:0")
    model = load_state_dict_from_lightning_ckpt(args.checkpoint, device=device)
    decoder = ESM3_structure_decoder_v0(device="cpu").to(device).eval()
    seq_tokenizer, structure_tokenizer = EsmSequenceTokenizer(), StructureTokenizer()
    all_summaries = []
    for target in manifest["selected"][:args.limit]:
        sequence = target["sequence"]
        seq = tokenize_sequence(sequence, seq_tokenizer).to(device)[None]
        lengths = torch.tensor([len(sequence)], device=device)
        references = [ESMProtein.from_pdb(ROOT / path).coordinates[:, 1].numpy()
                      for path in target["paths"]]
        cached = sorted((args.baseline / target["id"]).glob("seed*.npz"))
        if not cached:
            raise ValueError(f"Missing baseline target: {target['id']}")
        for policy in args.policies:
            directory = args.output / policy / target["id"]
            directory.mkdir(parents=True, exist_ok=True)
            rows, ensemble = [], []
            for path in cached:
                metadata = directory / (path.stem + ".json")
                destination = directory / path.name
                if metadata.exists() and destination.exists():
                    row = json.loads(metadata.read_text())
                    if (row["rounds"], row["steps"], row["radius"]) != (args.rounds, args.steps, args.radius):
                        raise ValueError(f"Cached feedback protocol mismatch: {metadata}")
                    if row.get("adaptive_budget", False) != args.adaptive_budget:
                        raise ValueError(f"Cached budget mode mismatch: {metadata}")
                    if row.get("score_mode", "ca") != args.score_mode:
                        raise ValueError(f"Cached score mode mismatch: {metadata}")
                    with np.load(destination) as arrays:
                        coords = arrays["atom37"]
                else:
                    base_row = json.loads(path.with_suffix(".json").read_text())
                    with np.load(path) as arrays:
                        fine = torch.from_numpy(arrays["fine"].copy()).to(device)[None]
                        coarse = torch.from_numpy(arrays["coarse"].copy()).to(device)[None]
                    seed_all(base_row["actual_seed"] + 10000000)

                    def decode(tokens):
                        padded = torch.cat([tokens.new_tensor([C.FINE_STRUCTURE_BOS_TOKEN]),
                                            tokens, tokens.new_tensor([C.FINE_STRUCTURE_EOS_TOKEN])])
                        return decode_structure(padded, decoder, structure_tokenizer, sequence)[0]

                    started = perf_counter()
                    new_fine, new_coarse, trace = geometry_feedback(
                        model, seq, fine, coarse, lengths,
                        lambda tokens, index: decode(tokens) if args.score_mode == "backbone" else decode(tokens)[:, 1],
                        rounds=args.rounds,
                        num_steps=args.steps, radius=args.radius, policy=policy,
                        fixed_budget=not args.adaptive_budget,
                        assessment_fn=assess_backbone if args.score_mode == "backbone" else None,
                        acceptance_fn=backbone_improves if args.score_mode == "backbone" else None)
                    coords = decode(new_fine[0, 1:-1]).numpy()
                    row = dict(target=target["id"], policy=policy,
                               baseline=str(path), seed=base_row["seed"], sample=base_row["sample"],
                               rounds=args.rounds, steps=args.steps, radius=args.radius,
                               adaptive_budget=args.adaptive_budget,
                               score_mode=args.score_mode,
                               elapsed_seconds=perf_counter() - started,
                               network_calls=sum(event.get("network_calls", 0) for event in trace),
                               trace=trace, repair_proxy=assess_ca(coords[:, 1]).summary(),
                               metrics=summarize_backbone(coords, references))
                    np.savez_compressed(destination, fine=new_fine[0].cpu().numpy(),
                                        coarse=new_coarse[0].cpu().numpy(), atom37=coords)
                    atomic_json(metadata, row)
                rows.append(row)
                ensemble.append(coords[:, 1])
            summary = dict(target=target["id"], policy=policy, length=target["length"], samples=len(rows),
                           score_mode=args.score_mode, adaptive_budget=args.adaptive_budget,
                           proxy_invalid_samples=sum(bool(r["repair_proxy"]["bad_residues"]) for r in rows),
                           upstream_clash_invalid_samples=sum(not r["metrics"].get("upstream_ca_clash_valid", False) for r in rows),
                           cn_outlier_samples=sum(bool(r["metrics"].get("backbone_cn_outliers", 1)) for r in rows),
                           accepted_proposals=sum(bool(t.get("accepted", False)) for r in rows for t in r["trace"]),
                           network_calls=sum(r["network_calls"] for r in rows),
                           elapsed_seconds=sum(r["elapsed_seconds"] for r in rows),
                           **ensemble_summary(ensemble, references))
            all_summaries.append(summary)
            atomic_json(directory / "summary.json", summary)
            atomic_json(args.output / "summary.json", dict(note=manifest["caveat"], targets=all_summaries))
            print(json.dumps(summary), flush=True)
    print("FEEDBACK_SCREEN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
