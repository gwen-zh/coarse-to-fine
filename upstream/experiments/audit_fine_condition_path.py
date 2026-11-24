"""Read-only layer response on the already selected two-state training probes.

Fixed fine noise, time zero: no native fine information enters this intervention.
At later times the matched/swapped teacher comparison is only a sensitivity
probe: the swapped joint input can be outside the training distribution.
"""
import json
from pathlib import Path

import numpy as np
import torch

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from joint_sequence_predictor import encode_aa
from run_fine_single_structure import key_seed, make_model
from run_fine_two_state import verify
from run_sequence_multiscale_ensemble import load_codec
from structural_metrics import aligned_rmsd


ROOT = Path("artifacts/fine_condition_path_v1")
PARENT = Path("artifacts/fine_two_state_v1")


def rms(value):
    return float(value.square().mean().sqrt())


@torch.no_grad()
def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "report.json").exists():
        raise RuntimeError("Use a fresh read-only audit namespace")
    p = json.loads((PARENT / "protocol.json").read_text())
    verify(p)
    pairs = torch.load(PARENT / "pairs.pt", map_location="cpu", weights_only=True)
    codec = load_codec(p, "centers3")
    records = []
    for name in p["proteins"]:
        cases = [pairs[name]["a"], pairs[name]["b"]]
        aa = encode_aa(cases[0]["sequence"])
        m = len(cases[0]["coarse"])
        context = [codec.context(c["coarse"], len(aa)) for c in cases]
        for seed in p["seeds"]:
            for phase in ["initial", "final"]:
                model = make_model(p, cases[0], seed)
                if phase == "final":
                    saved = torch.load(PARENT / name / f"seed{seed}" / "last.pt",
                                       map_location="cpu", weights_only=False)
                    if saved["updates"] != p["updates"] or saved["protocol_sha256"] != sha(PARENT / "protocol.json"):
                        raise RuntimeError("Wrong two-state checkpoint")
                    model.load_state_dict(saved["model"])
                model.eval().requires_grad_(False)
                before = {k: v.clone() for k, v in model.state_dict().items()}
                cs = [model.normalize_coarse(c["coarse"]) for c in cases]
                fs = [model.normalize_fine(c["fine_continuous"]) for c in cases]
                residue, _ = model.aa_conditions(aa)
                traces = []
                teacher = []
                for index in range(4):
                    rng = torch.Generator().manual_seed(key_seed(seed, name, "condition-path", index))
                    noise = torch.randn(fs[0].shape, generator=rng)
                    snapshots = []
                    for c in cs:
                        outputs = {}
                        hooks = [model.transformer.register_forward_pre_hook(
                            lambda module, args: outputs.update(input=args[0][0].clone()))]
                        for depth, layer in enumerate(model.transformer.layers):
                            def capture(module, args, result, depth=depth):
                                outputs[f"layer{depth + 1}"] = result[0].clone()
                            hooks.append(layer.register_forward_hook(capture))
                        try:
                            velocity = model.fine_velocity(aa, noise, 0., c)
                        finally:
                            for hook in hooks:
                                hook.remove()
                        outputs["velocity"] = velocity
                        snapshots.append(outputs)
                    traces.append({label: dict(
                        fine_response_rms=rms(snapshots[1][label][m:] - snapshots[0][label][m:]),
                        fine_activation_rms=rms(snapshots[0][label][m:]))
                        for label in snapshots[0] if label != "velocity"})
                    traces[-1]["velocity"] = dict(
                        fine_response_rms=rms(snapshots[1]["velocity"] - snapshots[0]["velocity"]),
                        fine_activation_rms=rms(snapshots[0]["velocity"]))
                for t in [0., .1, .5, .9]:
                    values = []
                    for side in [0, 1]:
                        for index in range(4):
                            rng = torch.Generator().manual_seed(key_seed(seed, name, "condition-teacher", index))
                            noise = torch.randn(fs[side].shape, generator=rng)
                            state = (1 - t) * noise + t * fs[side]
                            a = model.fine_velocity(aa, state, t, cs[side])
                            b = model.fine_velocity(aa, state, t, cs[1 - side])
                            truth = fs[side] - noise
                            values.append((float((a - truth).square().mean()),
                                           float((b - truth).square().mean()), rms(a - b)))
                    mean = np.mean(values, axis=0)
                    teacher.append(dict(time=t, matched_mse=float(mean[0]), swapped_mse=float(mean[1]),
                                        condition_velocity_response_rms=float(mean[2])))
                records.append(dict(protein=name, seed=seed, phase=phase,
                    coarse_aligned_center_rmsd_A=aligned_rmsd(cases[0]["coarse"].numpy(), cases[1]["coarse"].numpy()),
                    normalized_coarse_delta_rms=rms(cs[1] - cs[0]),
                    decoder_local_geometry_delta_rms=rms(context[1]["geometry"] - context[0]["geometry"]),
                    native_fine_delta_rms=rms(fs[1] - fs[0]),
                    aa_condition_rms=rms(residue),
                    broadcast_coarse_projection_delta_rms=rms(model.coarse_to_fine(cs[1]) - model.coarse_to_fine(cs[0])),
                    layer_trace={label: {metric: float(np.mean([s[label][metric] for s in traces]))
                        for metric in traces[0][label]} for label in traces[0]}, teacher=teacher))
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    verify(p)
    sources = [Path(__file__), PARENT / "protocol.json", PARENT / "pairs.pt", PARENT / "report.json",
               PARENT / "assessment.json"]
    sources += [PARENT / n / f"seed{s}" / "last.pt" for n in p["proteins"] for s in p["seeds"]]
    result = dict(status="complete", records=records, training_performed=False, model_weights_unchanged=True,
        source_sha256={str(x): sha(x) for x in sources},
        scope="Read-only activation and teacher sensitivity audit on two preselected native training pairs; no new coordinates, development, test or R3 used.",
        caveat="Small coarse changes need not determine all fine variation in a real ensemble. Exact two-point fitting is an engineering probe; swapped teacher inputs can be outside the joint training distribution.")
    atomic_json(ROOT / "report.json", result)
    for row in records:
        print(row["protein"], row["seed"], row["phase"],
              {k: round(v["fine_response_rms"], 5) for k, v in row["layer_trace"].items()}, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
