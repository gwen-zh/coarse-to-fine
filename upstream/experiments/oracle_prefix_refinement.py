"""Greedy reference-assisted continuation after the SAME generated coarse tokens.

This diagnoses the existing residual dictionaries' correction capacity. It is
neither an inference method nor a bound on globally optimal structural quality.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_residual_vq import ROOT, ProjectedDecoder, ESMProtein, decode_embedding
from esm.models.residual_bridge import load_residual_bridge
from refinement_generator import greedy_oracle_refinement
from structural_metrics import summarize_backbone
from train_residual_vq import atomic_json


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    for name in ['tokens', 'manifest', 'cache', 'weights', 'source', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    args.output.mkdir(exist_ok=True, parents=True)
    decoder = ProjectedDecoder(d_model=1280, n_heads=20, n_layers=30)
    state = torch.load(args.weights / 'decoder_checkpoint_110.pth', mmap=True, map_location='cpu', weights_only=False)
    decoder.load_state_dict(state.get('decode_proj', state), strict=True)
    del state
    decoder.eval().requires_grad_(False)
    bridge, _ = load_residual_bridge(args.tokens / 'residual_bridge.pt')
    groups = {t['id']: t for t in json.loads(args.manifest.read_text())['all_groups'] if t['split'] == 'development'}
    targets = [t for t in json.loads((args.tokens / 'manifest.json').read_text())['targets'] if t['subset'] == 'validation']
    cfg = json.loads((args.source / 'summary.json').read_text())['protocol']
    rows = []
    for t in targets:
        ref = ESMProtein.from_pdb(ROOT / groups[t['id']]['paths'][0])
        assert ref.sequence == t['sequence']
        with np.load(args.cache / (t['id'] + '.npz')) as a:
            z = torch.from_numpy(a['latent'].copy())[None]
        for sample in range(cfg['samples']):
            path = args.output / f"{t['id']}_sample{sample}.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
                continue
            with np.load(args.source / 'predictions' / f"{t['id']}_sample{sample}.npz") as a:
                predicted = torch.from_numpy(a['tokens'].copy()).long()[None]
            corrected = greedy_oracle_refinement(bridge, z, predicted[..., 0])
            assert torch.equal(corrected[..., 0], predicted[..., 0])
            fine = bridge.fine_tokens(corrected)[0]
            bb = decode_embedding(decoder, decoder.embed(fine)).numpy()
            row = dict(target=t['id'], sample=sample,
                       predicted_latent_mse=F.mse_loss(bridge.prefix_latent(predicted), z).item(),
                       greedy_oracle_latent_mse=F.mse_loss(bridge.prefix_latent(corrected), z).item(),
                       metrics=summarize_backbone(bb, [ref.coordinates[:, 1].numpy()]))
            atomic_json(path, row)
            np.savez_compressed(path.with_suffix('.npz'), tokens=corrected[0].numpy(), backbone=bb)
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = dict(note='Reference-assisted GREEDY nearest-code continuation on the same generated coarse. '
                        'Uses validation reference latent. Not usable generation or a global optimality bound.',
                   proteins=len(targets), samples=len(rows),
                   predicted_latent_mse=float(np.mean([r['predicted_latent_mse'] for r in rows])),
                   greedy_oracle_latent_mse=float(np.mean([r['greedy_oracle_latent_mse'] for r in rows])),
                   reference_ca_rmsd=float(np.mean([r['metrics']['closest_available_reference_ca_rmsd'] for r in rows])),
                   ca_clash_validity=float(np.mean([r['metrics']['upstream_ca_clash_valid'] for r in rows])),
                   coarse_mismatches=0, targets=rows)
    atomic_json(args.output / 'summary.json', summary)
    print('GREEDY_ORACLE_REFINEMENT_COMPLETE', json.dumps({k:v for k,v in summary.items() if k != 'targets'}), flush=True)


if __name__ == '__main__':
    main()
