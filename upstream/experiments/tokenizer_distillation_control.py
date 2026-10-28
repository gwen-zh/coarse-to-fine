"""Test whether healthy medium dictionaries can drive a frozen decoder.

Learn medium embedding means from fine-token embeddings on training proteins,
then evaluate structure reconstruction on the 40 validation proteins. This is
an exploratory tokenizer control; it does not update the conformation generator.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Hierarchical_VQ_VAE"))
from esm.models.new_vqvae import ProjectedDecoder
from esm.sdk.api import ESMProtein
from quantization_control import assignments
from structural_metrics import aligned_rmsd, summarize_backbone


def fit_table(training, centers, fine_table, special_table, cached_medium=False):
    sums = np.zeros((len(centers), fine_table.shape[1]), dtype=np.float64)
    counts = np.zeros(len(centers), dtype=np.int64)
    for z, fine_ids, medium_ids in training:
        ids = medium_ids if cached_medium else assignments(z, centers)
        np.add.at(sums, ids, fine_table[fine_ids])
        counts += np.bincount(ids, minlength=len(centers))
    occupied = counts > 0
    means = np.zeros_like(sums, dtype=np.float32)
    means[occupied] = (sums[occupied] / counts[occupied, None]).astype(np.float32)
    if (~occupied).any():
        means[~occupied] = means[np.flatnonzero(occupied)[assignments(centers[~occupied], centers[occupied])]]
    return np.concatenate([means, special_table], axis=0), int((~occupied).sum())


def pad(ids, size, device):
    return torch.tensor(np.concatenate([[size + 2], ids, [size + 1]]), dtype=torch.long, device=device)[None]


def decode_table(decoder, ids, table):
    tokens = pad(ids, len(table) - 5, table.device)
    length = tokens.shape[1]
    sequence_id = torch.zeros_like(tokens)
    x = F.embedding(tokens, table)
    x, _, _ = decoder.decoder_stack.forward(
        x, affine=None, affine_mask=None, sequence_id=sequence_id, chain_id=sequence_id,
        attn_mask=torch.zeros((1, length, length), device=table.device))
    _, bb = decoder.affine_output_projection(
        x, affine=None, affine_mask=torch.zeros((1, length), dtype=torch.bool, device=table.device))
    return bb[0, 1:-1]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    protocol = json.loads((args.control / "summary.json").read_text())
    targets = {t["id"]: t for t in manifest["all_groups"] if t["split"] == "development"}
    decoder = ProjectedDecoder(d_model=1280, n_heads=20, n_layers=30)
    state = torch.load(args.weights / "decoder_checkpoint_110.pth", map_location="cpu", mmap=True, weights_only=False)
    decoder.load_state_dict(state.get("decode_proj", state), strict=True)
    del state
    fine_table = decoder.embed.weight.detach().numpy().copy()
    special_table = decoder.medium_embed.weight[512:].detach().numpy().copy()
    training = []
    for name, split in protocol["split"].items():
        if split == "train":
            with np.load(args.cache / f"{name}.npz") as arrays:
                training.append((arrays["latent"].copy(), arrays["fine"].copy(), arrays["medium"].copy()))
    learned, centers, empty_counts = {}, {}, {}
    with np.load(args.control / "dictionaries.npz") as arrays, threadpool_limits(limits=1):
        for name in ["kmeans512", "data_init512", "released512_distilled"]:
            centers[name] = arrays["released512" if name == "released512_distilled" else name].copy()
            learned[name], empty_counts[name] = fit_table(
                training, centers[name], fine_table, special_table, cached_medium=name == "released512_distilled")
    del training
    np.savez_compressed(args.output / "medium_embeddings.npz", **learned)
    device = torch.device(args.device)
    decoder.to(device).eval()
    tables = {name: torch.from_numpy(value).to(device) for name, value in learned.items()}
    tables.update(fine=decoder.embed.weight, released512=decoder.medium_embed.weight)
    rows = []
    for name, split in protocol["split"].items():
        if split != "validation":
            continue
        target = targets[name]
        with np.load(args.cache / f"{name}.npz") as arrays:
            z = arrays["latent"].copy()
            fine, medium, coarse = [arrays[key].copy() for key in ["fine", "medium", "coarse"]]
        reference = ESMProtein.from_pdb(ROOT / target["paths"][0]).coordinates[:, :3].numpy()
        ids = dict(fine=fine, released512=medium, released512_distilled=medium)
        with threadpool_limits(limits=1):
            ids.update({method: assignments(z, codes) for method, codes in centers.items()
                        if method != "released512_distilled"})
        if not rows:
            length = torch.tensor([len(z)], device=device)
            original = decoder.decode(pad(fine, 4096, device), pad(medium, 512, device),
                                      pad(coarse, 32, device), length)["bb_pred_medium"][0, 1:-1]
            shortcut = decode_table(decoder, medium, decoder.medium_embed.weight)
            torch.testing.assert_close(shortcut, original, atol=1e-5, rtol=1e-5)
        row = dict(target=name, length=len(z), methods={})
        for method, table in tables.items():
            prediction = decode_table(decoder, ids[method], table).cpu().numpy()
            row["methods"][method] = dict(
                ca_rmsd=aligned_rmsd(prediction[:, 1], reference[:, 1]),
                backbone_rmsd=aligned_rmsd(prediction.reshape(-1, 3), reference.reshape(-1, 3)),
                metrics=summarize_backbone(prediction))
        rows.append(row)
        print(json.dumps(row), flush=True)
        (args.output / "summary.json").write_text(json.dumps(dict(
            note="Frozen-decoder reconstruction control on development validation proteins, not an upgraded generator or independent test benchmark.",
            train_proteins=protocol["train_proteins"], validation_proteins=protocol["validation_proteins"],
            empty_training_clusters=empty_counts, targets=rows), indent=2) + "\n")
    print("TOKENIZER_DISTILLATION_CONTROL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
