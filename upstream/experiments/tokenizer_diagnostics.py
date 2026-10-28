"""Measure released tokenizer utilization and per-level reconstruction, without generation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
# The two upstream packages have different esm implementations. Use the actual
# tokenizer package here, in a separate process from the generation package.
sys.path.insert(0, str(ROOT / "Hierarchical_VQ_VAE"))
from esm.models.new_vqvae import ProjectedEncoder, ProjectedDecoder, apply_special_tokens
from esm.sdk.api import ESMProtein
from structural_metrics import aligned_rmsd


def entropy(counts):
    counts = np.asarray(counts, float)
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def token_statistics(tokens, size):
    values = np.concatenate(tokens)
    counts = np.bincount(values, minlength=size)
    h = entropy(counts)
    return dict(tokens=len(values), vocabulary=size, used=int((counts > 0).sum()),
                utilization=float((counts > 0).mean()), entropy_bits=h,
                perplexity=float(2 ** h), counts=counts.tolist())


def conditional_entropy(x, y, nx, ny):
    """Plugin estimate H(Y|X); biased with sparse counts, descriptive only."""
    x, y = np.concatenate(x), np.concatenate(y)
    counts = np.bincount(x * ny + y, minlength=nx * ny)
    return entropy(counts) - entropy(np.bincount(x, minlength=nx))


def load_weights(module, path, nested_key):
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if nested_key in state:
        state = state[nested_key]
    if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"Unsupported checkpoint schema: {path}")
    module.load_state_dict(state, strict=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--all-development", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    encoder = ProjectedEncoder(d_model=1024, n_heads=1, v_heads=128, n_layers=2,
                               d_out=128, n_codes=4096)
    load_weights(encoder, args.weights / "encoder_checkpoint_110.pth", "encoder_weight")
    encoder.to(device).eval()
    decoder = None
    if not args.encoder_only:
        decoder = ProjectedDecoder(d_model=1280, n_heads=20, n_layers=30)
        load_weights(decoder, args.weights / "decoder_checkpoint_110.pth", "decode_proj")
        decoder.to(device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    tokens = {name: [] for name in ["coarse", "medium", "fine"]}
    rows = []
    latent = []
    hook = encoder.pre_vq_proj.register_forward_hook(lambda m, i, o: latent.append(o.detach()))
    targets = (sorted((t for t in manifest["all_groups"] if t["split"] == "development"),
                      key=lambda t: t["sequence_hash"])
               if args.all_development else manifest["selected"])
    for target in targets:
        # One structure per sequence prevents ensemble size from dominating usage.
        path = ROOT / target["paths"][0]
        protein = ESMProtein.from_pdb(path)
        coords = protein.coordinates[None].to(device)
        n = len(protein.sequence)
        lengths = torch.tensor([n], device=device)
        latent.clear()
        output = encoder.encode(coords)
        z = latent[0]
        large, medium, small = output[3:6]
        decoded = None
        if decoder is not None:
            padded = [apply_special_tokens(t, lengths, k + 2, k + 1, k + 3)
                      for t, k in [(large, 4096), (medium, 512), (small, 32)]]
            decoded = decoder.decode(*padded, lengths)
        record = dict(target=target["id"], length=n, levels={})
        arrays = {"latent": z[0].cpu().numpy()}
        for name, index, q, key in [("coarse", small, output[2], "small"),
                                    ("medium", medium, output[1], "medium"),
                                    ("fine", large, output[0], "large")]:
            ids = index[0].cpu().numpy()
            tokens[name].append(ids)
            arrays[name] = ids
            record["levels"][name] = dict(latent_mse=float(torch.mean((q - z) ** 2)))
            if decoded is not None:
                prediction = decoded[f"bb_pred_{key}"][0, 1:n + 1].cpu().numpy()
                reference = coords[0, :, :3].cpu().numpy()
                arrays[f"backbone_{name}"] = prediction
                record["levels"][name].update(
                    ca_rmsd=aligned_rmsd(prediction[:, 1], reference[:, 1]),
                    backbone_rmsd=aligned_rmsd(prediction.reshape(-1, 3), reference.reshape(-1, 3)))
        np.savez_compressed(args.output / f"{target['id']}.npz", **arrays)
        rows.append(record)
        print(json.dumps(record), flush=True)
    hook.remove()
    stats = {name: token_statistics(tokens[name], k)
             for name, k in [("coarse", 32), ("medium", 512), ("fine", 4096)]}
    for name, codebook in [("coarse", encoder.small_codebook),
                            ("medium", encoder.medium_codebook), ("fine", encoder.codebook)]:
        norms = codebook.embeddings.norm(dim=-1).cpu().numpy()
        used = np.asarray(stats[name]["counts"]) > 0
        stats[name]["mean_used_embedding_norm"] = float(norms[used].mean())
        stats[name]["mean_unused_embedding_norm"] = float(norms[~used].mean()) if (~used).any() else None
    summary = dict(encoder_only=args.encoder_only, all_development=args.all_development,
                   note="Exploratory development structures; sparse-count entropies are not evidence of semantic redundancy.",
                   targets=rows,
                   codebooks=stats,
                   conditional_entropy_bits={
                       "medium_given_coarse": conditional_entropy(tokens["coarse"], tokens["medium"], 32, 512),
                       "fine_given_coarse": conditional_entropy(tokens["coarse"], tokens["fine"], 32, 4096),
                       "fine_given_medium": conditional_entropy(tokens["medium"], tokens["fine"], 512, 4096)})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("TOKENIZER_DIAGNOSTICS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
