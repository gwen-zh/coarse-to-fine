"""Read-only update/exposure audit after the frozen multi-frame comparison."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from run_multiframe_early_fine import ROOT, verify


def main():
    p = json.loads((ROOT / "protocol.json").read_text()); verify(p)
    labels = torch.load(ROOT / "train_labels.pt", map_location="cpu", weights_only=True)
    eligible = {(c["protein"], c["id"]) for c in labels["PDB_train"]}
    md_frames = sum(len(cases) for cases in labels["train"].values())
    buffers = {"coarse_mean", "coarse_std", "fine_mean", "fine_std", "fixed_aa_lookup"}
    records = []; sources = [Path(__file__), ROOT / "protocol.json", ROOT / "train_labels.pt"]
    for seed in p["seeds"]:
        before = torch.load(p["checkpoints"][str(seed)], map_location="cpu", weights_only=False)["model"]
        names = [k for k in before if k not in buffers and not k.startswith("coarse_head.")]
        if sum(before[k].numel() for k in names) != p["trainable_parameters"]:
            raise RuntimeError("Parameter/buffer audit does not match the actual trainable count")
        for arm in p["arms"]:
            out = ROOT / arm / f"seed{seed}"
            saved = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
            if saved["updates"] != p["updates"] or saved["protocol_sha256"] != sha(ROOT / "protocol.json"):
                raise RuntimeError("Wrong completed checkpoint")
            after = saved["model"]
            for key, value in before.items():
                torch.testing.assert_close(after["coarse_model." + key], value, atol=0, rtol=0)
                if key in buffers or key.startswith("coarse_head."):
                    torch.testing.assert_close(after["fine_model." + key], value, atol=0, rtol=0)
            numerator = denominator = 0.; changed = 0
            for key in names:
                a, b = after["fine_model." + key], before[key]
                numerator += float((a.double() - b.double()).square().sum())
                denominator += float(b.double().square().sum())
                changed += int(not torch.equal(a, b))
            if numerator <= 0:
                raise RuntimeError("Fine weights did not change")
            trace = json.loads((out / "trace.json").read_text())
            pdb = Counter((c["protein"], c["case_id"]) for step in trace for c in step["batch"] if c["domain"] == "PDB")
            md = Counter((c["protein"], c["case_id"]) for step in trace for c in step["batch"] if c["domain"] == "MD")
            records.append(dict(arm=arm, seed=seed, updates=len(trace),
                changed_trainable_tensors=changed, trainable_tensors=len(names),
                fine_parameter_relative_L2_change=(numerator / denominator) ** .5,
                coarse_weights_and_fine_buffers_bitwise_unchanged=True,
                new_PDB_presentations=sum(pdb.values()), eligible_PDB_cases=len(eligible),
                unique_PDB_cases=len(pdb), new_PDB_presentations_per_eligible_case=sum(pdb.values()) / len(eligible),
                PDB_min_max_presentations=[min(pdb[k] for k in eligible), max(pdb[k] for k in eligible)],
                new_MD_presentations=sum(md.values()), eligible_MD_frames=md_frames, unique_MD_frames=len(md),
                new_MD_presentations_per_eligible_frame=sum(md.values()) / md_frames,
                first_128_mean_training_loss=float(np.mean([x["loss"] for x in trace[:128]])),
                last_128_mean_training_loss=float(np.mean([x["loss"] for x in trace[-128:]]))))
            sources += [out / "last.pt", out / "trace.json"]
    verify(p)
    result = dict(status="complete", records=records,
        source_sha256={str(f): sha(f) for f in sources},
        scope="New 2048-update continuation only. Exposure counts exclude earlier parent training; a small exposure count or falling stochastic loss does not establish that a longer run will improve generation.",
        training_performed=False, model_weights_unchanged_by_audit=True)
    atomic_json(ROOT / "continuation_audit.json", result)
    for row in records:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
