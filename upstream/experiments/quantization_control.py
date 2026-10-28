"""Capacity control on frozen latents; not a retrained structure generator.

Fit dictionary controls on a protein-level development training split and
evaluate on separate development proteins. The held-out manifest split is never
read. Quantization error is a proxy and cannot establish structure improvements.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits
import torch


def assignments(z, centers, chunk=512):
    indices = []
    for start in range(0, len(z), chunk):
        x = z[start:start + chunk]
        distance = (x * x).sum(1)[:, None] - 2 * x @ centers.T + (centers * centers).sum(1)
        indices.append(distance.argmin(1))
    return np.concatenate(indices)


def evaluate(items, centers):
    indices, errors = [], []
    squared = 0.0
    scalars = 0
    for name, z in items:
        ids = assignments(z, centers)
        error = float(np.mean((z - centers[ids]) ** 2))
        errors.append(dict(target=name, latent_mse=error))
        indices.append(ids)
        squared += error * z.size
        scalars += z.size
    counts = np.bincount(np.concatenate(indices), minlength=len(centers))
    p = counts[counts > 0] / counts.sum()
    return dict(proteins=len(items), residues=int(counts.sum()), used=int((counts > 0).sum()),
                vocabulary=len(centers), perplexity=float(2 ** -(p * np.log2(p)).sum()),
                residue_weighted_mse=squared / scalars,
                protein_weighted_mse=float(np.mean([e["latent_mse"] for e in errors])),
                targets=errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoder-cache", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    train, validation = [], []
    split = {}
    for target in manifest["all_groups"]:
        if target["split"] != "development":
            continue
        subset = "validation" if int(target["sequence_hash"][8:16], 16) % 5 == 0 else "train"
        split[target["id"]] = subset
        with np.load(args.encoder_cache / f"{target['id']}.npz") as arrays:
            item = target["id"], arrays["latent"].copy()
        (train if subset == "train" else validation).append(item)
    z_train = np.concatenate([z for _, z in train])
    state = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = state.get("encoder_weight", state)
    centers = dict(released32=state["small_codebook.embeddings"].numpy(),
                   released512=state["medium_codebook.embeddings"].numpy())
    recipe = dict(random_state=0, n_init=3, batch_size=2048, max_iter=100, reassignment_ratio=0.01)
    with threadpool_limits(limits=args.threads):
        for size in [32, 512]:
            model = MiniBatchKMeans(n_clusters=size, **recipe).fit(z_train)
            centers[f"kmeans{size}"] = model.cluster_centers_
        rng = np.random.default_rng(0)
        centers["data_init512"] = z_train[rng.choice(len(z_train), 512, replace=False)]
        evaluations = {name: evaluate(validation, codes) for name, codes in centers.items()}
    np.savez_compressed(args.output / "dictionaries.npz", **centers)
    result = dict(note="Latent-space capacity control only; new dictionaries need decoder and generator adaptation before structure generation.",
                  split=split, train_proteins=len(train), validation_proteins=len(validation),
                  training_residues=len(z_train), recipe=recipe, validation=evaluations)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"train_proteins": len(train), "validation_proteins": len(validation),
                      "validation": {name: {k: v for k, v in value.items() if k != "targets"}
                                     for name, value in evaluations.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
