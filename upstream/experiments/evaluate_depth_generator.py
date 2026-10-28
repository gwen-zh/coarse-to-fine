"""Decode fully predicted depth tokens; references are used only after sampling."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_residual_vq import ROOT, ProjectedDecoder, ESMProtein, decode_embedding
from esm.models.residual_bridge import load_residual_bridge
from structural_metrics import summarize_backbone, aligned_rmsd
from train_residual_vq import atomic_json


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--mode', choices=['residual', 'parallel', 'direct'], required=True)
    p.add_argument('--label', help='Optional output subdirectory for controlled refinement variants')
    args = p.parse_args()
    torch.set_num_threads(8)
    directory = args.run / (args.label or args.mode)
    training = json.loads((directory / 'summary.json').read_text())
    groups = {t['id']: t for t in json.loads(args.manifest.read_text())['all_groups']
              if t['split'] == 'development'}
    selected = json.loads((args.tokens / 'manifest.json').read_text())['targets']
    selected = [t for t in selected if t['subset'] == 'validation']
    decoder = ProjectedDecoder(d_model=1280, n_heads=20, n_layers=30)
    state = torch.load(args.weights / 'decoder_checkpoint_110.pth', map_location='cpu', mmap=True, weights_only=False)
    decoder.load_state_dict(state.get('decode_proj', state), strict=True)
    del state
    decoder.eval().requires_grad_(False)
    bridge = None
    if args.mode != 'direct':
        bridge, metadata = load_residual_bridge(args.tokens / f'{args.mode}_bridge.pt')
        assert metadata['generator_retraining_required']
    decoded = directory / 'decoded'
    decoded.mkdir(exist_ok=True)
    records = []
    for t in selected:
        path = decoded / (t['id'] + '.json')
        if path.exists():
            records.append(json.loads(path.read_text()))
            continue
        reference = ESMProtein.from_pdb(ROOT / groups[t['id']]['paths'][0])
        assert reference.sequence == t['sequence']
        refs = [reference.coordinates[:, 1].numpy()]
        with np.load(args.features / (t['id'] + '.npz')) as a:
            baseline_fine = torch.from_numpy(a['sampled_fine'][1:-1].copy()).long()
        baseline = decode_embedding(decoder, decoder.embed(baseline_fine)).numpy()
        row = dict(target=t['id'], baseline=summarize_backbone(baseline, refs), samples=[])
        ensemble = []
        for sample in range(training['protocol']['samples']):
            with np.load(directory / 'predictions' / f"{t['id']}_sample{sample}.npz") as a:
                tokens = torch.from_numpy(a['tokens'].copy()).long()
            fine = tokens[:, 0] if bridge is None else bridge.fine_tokens(tokens)
            assert fine.shape == baseline_fine.shape
            assert torch.all((fine >= 0) & (fine < 4096))
            bb = decode_embedding(decoder, decoder.embed(fine)).numpy()
            row['samples'].append(dict(sample=sample, metrics=summarize_backbone(bb, refs)))
            ensemble.append(bb[:, 1])
            np.savez_compressed(decoded / f"{t['id']}_sample{sample}.npz", backbone=bb, fine=fine.numpy())
        pairs = [aligned_rmsd(x, y) for i, x in enumerate(ensemble) for y in ensemble[i+1:]]
        row['mean_pairwise_ca_rmsd'] = float(np.mean(pairs)) if pairs else None
        atomic_json(path, row)
        records.append(row)
        print(json.dumps(row), flush=True)
    methods = {}
    for method in ['baseline', args.mode]:
        metrics = ([r['baseline'] for r in records] if method == 'baseline' else
                   [m['metrics'] for r in records for m in r['samples']])
        references = [r['closest_available_reference_ca_rmsd'] for r in metrics
                      if r.get('closest_available_reference_ca_rmsd') is not None]
        methods[method] = dict(samples=len(metrics),
            ca_clash_validity=float(np.mean([r.get('upstream_ca_clash_valid', False) for r in metrics])),
            mean_reference_ca_rmsd=float(np.mean(references)) if references else None,
            missing_reference_rmsd=len(metrics) - len(references),
            mean_cn_outliers=float(np.mean([r['backbone_cn_outliers'] for r in metrics
                                            if 'backbone_cn_outliers' in r])))
    atomic_json(directory / 'structure_summary.json', dict(protocol=training['protocol'],
                note='Fully predicted new tokens. Sequence-only frozen upstream scaffold context is cached '
                     'and its seven large-model network calls are excluded from adapter runtime. '
                     'Small development pilot, one upstream scaffold per protein, not a formal ensemble benchmark.',
                proteins=len(records), methods=methods,
                mean_pairwise_ca_rmsd=float(np.mean([r['mean_pairwise_ca_rmsd'] for r in records])),
                targets=records))
    print('DEPTH_GENERATOR_STRUCTURE_EVALUATION_COMPLETE', args.mode, json.dumps(methods), flush=True)


if __name__ == '__main__':
    main()
