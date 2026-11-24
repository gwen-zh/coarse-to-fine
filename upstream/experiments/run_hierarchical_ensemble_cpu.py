"""CPU runtime of the frozen same-cohort author comparison.

Sampling receives only AA sequences. Native coordinates are loaded after token
generation and used only for descriptive evaluation. No selection or repair.
"""
import gc
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from diagnostic_io import atomic_json
from native_ensemble_metrics import compare_ensembles
from multisequence_flow_data import physical_histogram_edges
from run_multisequence_ensemble_flow import sample_seed
from compare_two_level_ae_author import metrics


@torch.no_grad()
def run(p, root, labels, seed):
    torch.set_num_threads(16)
    # Import in the same order as the already validated original-path runner.
    from train_pretrained_coarse import load_state_dict_from_lightning_ckpt, EsmSequenceTokenizer, tokenize_sequence
    from evaluate_residual_vq import ProjectedDecoder, decode_embedding
    spec = p["author"]; out = root / f"seed{seed}/author"; out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists(): raise RuntimeError("Author comparison already complete")
    if (out / "sampling_started.json").exists(): raise RuntimeError("Inspect partial original-model run before retrying")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    atomic_json(out / "sampling_started.json", dict(seed=seed, host=platform.node(), budgets=spec["coarse_steps"],
        samples=spec["samples"], input="AA sequence only", policy=spec["policy"]))
    device = torch.device("cpu"); clock = time.perf_counter()
    model = load_state_dict_from_lightning_ckpt(Path(spec["generator"]["path"]), device=device).eval().requires_grad_(False)
    if not model.noise_removal or model.coarse_mask_index != 32: raise RuntimeError("Wrong released interface")
    tokenizer = EsmSequenceTokenizer(); model_load_seconds = time.perf_counter() - clock
    calls = {"coarse": 0, "fine": 0}
    def hook(module, args, kwargs, result):
        calls["coarse" if kwargs.get("scale", 0) == 0 else "fine"] += 1
    handle = model.net.register_forward_hook(hook, with_kwargs=True)
    arrays = {}; timing = {}; per_sample = {}
    # A fixed untimed warmup uses no scored sample seed.
    first = spec["targets"][0]
    seq = tokenize_sequence(first["sequence"], tokenizer).to(device)[None]
    torch.manual_seed(981); pass
    model.ddpm_sample(seq, lengths=torch.tensor([first["length"]], device=device), num_steps=5)
    for item in spec["targets"]:
        name = item["protein"]; n = item["length"]
        seq = tokenize_sequence(item["sequence"], tokenizer).to(device)[None]
        for steps in spec["coarse_steps"]:
            key = (name, steps); fine_values = []; coarse_values = []; seconds = []; metadata = []
            for index in range(spec["samples"]):
                sample_key = sample_seed(seed, name, index)
                torch.manual_seed(sample_key); pass; np.random.seed(sample_key)
                calls.update(coarse=0, fine=0); pass; started = time.perf_counter()
                fine, coarse = model.ddpm_sample(seq, lengths=torch.tensor([n], device=device), num_steps=steps, return_coarse=True)
                pass; elapsed = time.perf_counter() - started
                if calls != {"coarse": steps + 1, "fine": 1}: raise RuntimeError("Unexpected original backbone budget")
                f = fine[0, 1:n+1].cpu().numpy(); c = coarse[0, 1:n+1].cpu().numpy()
                if f.shape != (n,) or c.shape != (n,) or np.any((f < 0) | (f >= 4096)) or np.any((c < 0) | (c >= 32)):
                    raise RuntimeError("Invalid original token output")
                fine_values.append(f); coarse_values.append(c); seconds.append(elapsed)
                metadata.append(dict(sample=index, seed=sample_key, seconds=elapsed, backbone_calls=dict(calls)))
                if (index + 1) % 8 == 0:
                    atomic_json(out / "progress.json", dict(phase="author_sampling", protein=name, coarse_steps=steps,
                        samples=index + 1, total_per_group=spec["samples"], completed_groups=len(arrays)))
                    print("AUTHOR_ENSEMBLE_SAMPLE", seed, name, steps, index + 1, flush=True)
            arrays[key] = np.stack(fine_values); timing[key] = seconds; per_sample[key] = metadata
            np.savez_compressed(out / f"{name}_steps{steps}_tokens.npz", fine=arrays[key], coarse=np.stack(coarse_values))
            atomic_json(out / f"{name}_steps{steps}_sampling.json", dict(samples=metadata))
    handle.remove()
    if any(v.requires_grad or v.grad is not None for v in model.parameters()): raise RuntimeError("Author weights were not frozen")
    del fine, coarse, model; gc.collect(); pass
    started = time.perf_counter()
    decoder = ProjectedDecoder(d_model=1280, n_heads=20, n_layers=30)
    saved = torch.load(spec["decoder"]["path"], map_location="cpu", weights_only=False, mmap=True)
    decoder.load_state_dict(saved.get("decode_proj", saved), strict=True); del saved
    decoder.to(device).eval().requires_grad_(False); decoder_load_seconds = time.perf_counter() - started
    # Coordinates enter only here, after every scored token array is saved.
    dev = torch.load(labels / "development_labels.pt", map_location="cpu", weights_only=True)
    rows = []
    for item in spec["targets"]:
        name = item["protein"]; n = item["length"]; cases = dev["validation"][name]
        if cases[0]["sequence"] != item["sequence"]: raise RuntimeError("Reference AA mapping changed")
        refs = [cases[int(i)] for i in np.linspace(0, len(cases) - 1, min(128, len(cases)), dtype=int)]
        reference = np.stack([c["target"].numpy() for c in refs]); edges, rg = physical_histogram_edges(n)
        fine = torch.from_numpy(arrays[(name, spec["coarse_steps"][0])][0]).long().to(device)
        decode_embedding(decoder, decoder.embed(fine)); pass  # untimed decoder warmup
        for steps in spec["coarse_steps"]:
            key = (name, steps); backbones = []; decode_seconds = []; failures = []
            for index, ids in enumerate(arrays[key]):
                tokens = torch.from_numpy(ids).long().to(device); pass; started = time.perf_counter()
                bb = decode_embedding(decoder, decoder.embed(tokens))
                pass; decode_seconds.append(time.perf_counter() - started)
                x = bb.cpu().numpy(); backbones.append(x)
                if x.shape != (n, 3, 3): raise RuntimeError("Wrong original decoder output shape")
                if not np.isfinite(x).all(): failures.append(dict(sample=index, code="author_decoder_nonfinite"))
            bb = np.stack(backbones)
            row = dict(protein=name, split="development", outlet="full", coarse_steps=steps,
                generated_samples=spec["samples"], reference_frames=len(reference), metrics_available=not failures,
                failures=failures, failure_count=len(failures), coarse_metrics=None, geometry=None,
                mean_generation_seconds=float(np.mean(timing[key])), mean_decode_seconds=float(np.mean(decode_seconds)),
                mean_full_generation_seconds=float(np.mean(timing[key]) + np.mean(decode_seconds)),
                timing_device="CPU FP32,16threads", timing_scope="Warm generator+decoder; excludes weight loads, token transfer, features and metric evaluation. CPU thread counts differ from our small-model runs; no matched-resource speed claim.")
            if not failures:
                row.update(compare_ensembles(bb[:, :, 1], reference[:, :, 1], edges, rg, reference[0, :, 1]))
                from structural_metrics import aligned_rmsd
                geometry = []
                for x in bb:
                    nearest = int(np.argmin([aligned_rmsd(x[:, 1], ref[:, 1]) for ref in reference]))
                    geometry.append(metrics(x, reference[nearest]))
                row["geometry"] = {k: float(np.mean([x[k] for x in geometry])) for k in geometry[0]}
            rows.append(row)
            np.savez_compressed(out / f"{name}_steps{steps}_backbone.npz", backbone=bb, fine=arrays[key], valid=np.isfinite(bb).all(axis=(1, 2, 3)))
            atomic_json(out / "evaluation_progress.json", dict(completed_groups=len(rows), total=4))
            print("AUTHOR_ENSEMBLE_EVALUATED", seed, name, steps, "failures", len(failures), flush=True)
    atomic_json(out / "complete.json", dict(status="complete", seed=seed, rows=rows, host=platform.node(),
        gpu=None, cpu_threads=torch.get_num_threads(), model_load_seconds=model_load_seconds, decoder_load_seconds=decoder_load_seconds,
        all_samples_retained=True, input="AA sequence only", original_weights_frozen=True, scope=spec["scope"]))


if __name__ == "__main__":
    import argparse
    from hierarchical_pipeline_io import sha
    from run_coarse_target_scaling import verify
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=[2026, 2027])
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    root = Path("artifacts/coarse_coverage_repair_v1")
    addendum = json.loads((root / "author_cpu_runtime.json").read_text())
    if sha(root / "protocol.json") != addendum["protocol_sha256"] or sha(__file__) != addendum["cpu_runner_sha256"]:
        raise RuntimeError("CPU runtime addendum changed")
    protocol = json.loads((root / "protocol.json").read_text()); verify(protocol)
    if args.preflight:
        from train_pretrained_coarse import load_state_dict_from_lightning_ckpt, EsmSequenceTokenizer, tokenize_sequence
        from evaluate_residual_vq import ProjectedDecoder, decode_embedding
        torch.set_num_threads(16); torch.manual_seed(981)
        item = protocol["author"]["targets"][0]
        model = load_state_dict_from_lightning_ckpt(Path(protocol["author"]["generator"]["path"]), device="cpu").eval().requires_grad_(False)
        seq = tokenize_sequence(item["sequence"], EsmSequenceTokenizer())[None]
        started = time.perf_counter()
        with torch.no_grad():
            fine, coarse = model.ddpm_sample(seq, lengths=torch.tensor([item["length"]]), num_steps=25, return_coarse=True)
        sample_seconds = time.perf_counter() - started
        ids = fine[0, 1:item["length"]+1].clone()
        if ids.shape != (item["length"],) or (ids < 0).any() or (ids >= 4096).any():
            raise RuntimeError("Invalid CPU preflight fine tokens")
        del model; gc.collect()
        decoder = ProjectedDecoder(d_model=1280, n_heads=20, n_layers=30)
        saved = torch.load(protocol["author"]["decoder"]["path"], map_location="cpu", mmap=True, weights_only=False)
        decoder.load_state_dict(saved.get("decode_proj", saved), strict=True); del saved
        decoder.eval().requires_grad_(False)
        started = time.perf_counter()
        with torch.no_grad(): bb = decode_embedding(decoder, decoder.embed(ids))
        decode_seconds = time.perf_counter() - started
        if bb.shape != (item["length"], 3, 3) or not torch.isfinite(bb).all():
            raise RuntimeError("Invalid CPU preflight backbone")
        atomic_json(root / "author_cpu_preflight.json", dict(status="passed", coarse_steps=25,
            fine_tokens=len(ids), sample_seconds=sample_seconds, decode_seconds=decode_seconds,
            torch_version=torch.__version__, host=platform.node(), threads=16,
            input="AA only; no native coordinates loaded", scored_samples=0))
        print("AUTHOR_CPU_PREFLIGHT_PASSED", sample_seconds, decode_seconds, flush=True)
    else:
        if json.loads((root / "author_cpu_preflight.json").read_text())["status"] != "passed":
            raise RuntimeError("CPU preflight must pass first")
        run(protocol, root, Path("artifacts/multiframe_early_fine_v1"), args.seed)

