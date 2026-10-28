"""Freeze exploratory targets before inspecting generation outcomes."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hierarchical_ConfGen"))
from esm.sdk.api import ESMProtein


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = {}
    excluded = []
    for path in sorted((ROOT / "hierarchical_ConfGen/data/targets").glob("*/*.pdb")):
        if path.parent.name == "bpti":
            continue  # Already used in the smoke test.
        try:
            protein = ESMProtein.from_pdb(path)
            seq = protein.sequence
            coords = protein.coordinates
            if not seq or "|" in seq or "X" in seq or not 40 <= len(seq) <= 512:
                raise ValueError("sequence outside eligibility criteria")
            if coords.shape[0] != len(seq) or not torch.isfinite(coords[:, :3]).all():
                raise ValueError("missing backbone coordinates")
            key = hashlib.sha256(seq.encode()).hexdigest()
            tier = "short" if len(seq) <= 128 else "medium" if len(seq) <= 256 else "long"
            if key not in groups:
                groups[key] = dict(id=path.stem, sequence=seq, length=len(seq), tier=tier,
                                   sequence_hash=key, sources=[], paths=[],
                                   split="holdout" if int(key[:8], 16) % 5 == 0 else "development")
            groups[key]["paths"].append(str(path.relative_to(ROOT)))
            if path.parent.name not in groups[key]["sources"]:
                groups[key]["sources"].append(path.parent.name)
        except Exception as error:
            excluded.append(dict(path=str(path.relative_to(ROOT)), reason=str(error)))
    selected = []
    for tier in ["short", "medium", "long"]:
        candidates = sorted((g for g in groups.values()
                             if g["split"] == "development" and g["tier"] == tier),
                            key=lambda g: g["sequence_hash"])
        # Round-robin by source for a mix of folded and disordered targets.
        buckets = defaultdict(list)
        for candidate in candidates:
            buckets[candidate["sources"][0]].append(candidate)
        while len([x for x in selected if x["tier"] == tier]) < 4 and any(buckets.values()):
            for source in sorted(buckets):
                if buckets[source] and len([x for x in selected if x["tier"] == tier]) < 4:
                    selected.append(buckets[source].pop(0))
    result = dict(protocol="exploratory-development-v1", selection="sequence hash, length tier, source round-robin",
                  caveat="Published benchmark targets repurposed for development; exact-sequence grouping only, not homology-separated test data.",
                  selected=selected, all_groups=list(groups.values()), excluded=excluded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Eligible unique sequences:", len(groups), "splits:", dict(Counter(g["split"] for g in groups.values())))
    print("Selected:", [(g["id"], g["length"], g["sources"]) for g in selected])


if __name__ == "__main__":
    main()
