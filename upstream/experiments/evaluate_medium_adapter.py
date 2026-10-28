"""Decode selected adapters on development validation; sampled mode is sequence-only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from development_screen import (ROOT, ESM3_structure_decoder_v0, StructureTokenizer,
                                decode_structure, ESMProtein, atomic_json)
from structural_metrics import summarize_backbone


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--adapters', type=Path, required=True)
    args = p.parse_args()
    groups = json.loads(args.manifest.read_text())['all_groups']
    split = json.loads((args.control / 'summary.json').read_text())['split']
    groups = [t for t in groups if t['split'] == 'development' and split[t['id']] == 'validation']
    projection = torch.load(args.features / 'fine_projection.pt', weights_only=True)
    weight, bias = (projection[k][:4096].cuda() for k in ['weight', 'bias'])
    decoder = ESM3_structure_decoder_v0(device='cpu').cuda().eval()
    tokenizer = StructureTokenizer()
    runs = json.loads((args.adapters / 'summary.json').read_text())['runs']
    methods = ['baseline'] + [f"{r['variant']}_seed{r['seed']}" for r in runs]
    results = []
    for method in methods:
        directory = args.adapters / 'decoded' / method
        directory.mkdir(exist_ok=True, parents=True)
        records = []
        for t in groups:
            path = directory / (t['id'] + '.json')
            if path.exists():
                records.append(json.loads(path.read_text()))
                continue
            reference = ESMProtein.from_pdb(ROOT / t['paths'][0])
            assert reference.sequence == t['sequence']
            refs = [reference.coordinates[:, 1].numpy()]
            n = t['length']
            if method == 'baseline':
                with np.load(args.features / (t['id'] + '.npz')) as a:
                    predictions = {'sampled': a['sampled_fine'][1:-1].copy()}
                    h = torch.from_numpy(a['teacher_h']).cuda()
                    predictions['teacher'] = F.linear(h, weight, bias).argmax(-1).cpu().numpy()
            else:
                with np.load(args.adapters / method / (t['id'] + '.npz')) as a:
                    predictions = {mode: a[mode + '_fine'].copy() for mode in ['teacher', 'sampled']}
            row = dict(target=t['id'], method=method, modes={})
            coords = {}
            for mode, tokens in predictions.items():
                assert tokens.shape == (n,)
                assert np.all((tokens >= 0) & (tokens < 4096))
                fine = torch.from_numpy(np.concatenate([[4098], tokens, [4097]])).long().cuda()
                x, _, _ = decode_structure(fine, decoder, tokenizer, t['sequence'])
                x = x.cpu().numpy()
                coords[mode + '_atom37'] = x
                row['modes'][mode] = summarize_backbone(x, refs)
            np.savez_compressed(directory / (t['id'] + '.npz'), **coords)
            atomic_json(path, row)
            records.append(row)
        modes = {}
        for mode in ['teacher', 'sampled']:
            rows = [r['modes'][mode] for r in records]
            # Keep names from the metric implementation, rather than silently
            # assigning physical meaning to the CA and C-N diagnostic proxies.
            modes[mode] = {key: float(np.mean([r[key] for r in rows]))
                           for key in rows[0] if isinstance(rows[0][key], (int, float, bool))
                           and all(r.get(key) is not None for r in rows)}
        result = dict(method=method, proteins=len(records), modes=modes, targets=records)
        results.append(result)
        atomic_json(directory / 'summary.json', result)
        atomic_json(args.adapters / 'structure_comparison.json',
                    dict(note='Development validation only; one generated coarse sample per protein. '
                              'Teacher mode uses reference coarse tokens and is a diagnostic, not generation. '
                              'No diversity or ensemble-distribution conclusion.', methods=results))
        print(json.dumps(dict(method=method, modes=modes)), flush=True)
    print('MEDIUM_STRUCTURE_EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
