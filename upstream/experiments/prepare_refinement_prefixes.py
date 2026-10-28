"""Freeze the preceding policy's prefixes on development training proteins."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from depth_generator import DepthGenerator, sequence_ids
from refinement_generator import generation_seed
from train_residual_vq import atomic_json


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.source / 'best.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    cfg = checkpoint['protocol']
    assert cfg['mode'] == 'residual'
    model = DepthGenerator(sizes=cfg['sizes'], width=cfg['width'], layers=cfg['layers'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    manifest = json.loads((args.tokens / 'manifest.json').read_text())
    protocol = dict(version=1, source_checkpoint=str(path.resolve()),
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    samples=cfg['samples'], seed=cfg['seed'],
                    policy='frozen first residual generator, full-sequence sampled_h only',
                    warning='Training prefixes come from a policy fitted on those same training proteins; '
                            'not out-of-fold estimates. Validation prefixes reuse prior cached samples.')
    out = args.output / 'protocol.json'
    if out.exists() and json.loads(out.read_text()) != protocol:
        raise ValueError('Prefix cache provenance mismatch')
    atomic_json(out, protocol)
    rows = []
    for t in manifest['targets']:
        out = args.output / (t['id'] + '.npz')
        if not out.exists():
            predictions = []
            if t['subset'] == 'validation':
                for sample in range(cfg['samples']):
                    with np.load(args.source / 'predictions' / f"{t['id']}_sample{sample}.npz") as a:
                        predictions.append(a['tokens'].copy())
            else:
                with np.load(args.features / (t['id'] + '.npz')) as a:
                    h = torch.from_numpy(a['sampled_h'].copy())[None]
                seq = sequence_ids(t['sequence'])[None]
                for sample in range(cfg['samples']):
                    values, _ = model.generate(h, seq, generation_seed(t['id'], sample, cfg['seed']))
                    predictions.append(values[0].numpy())
            np.savez_compressed(out, predicted=np.stack(predictions))
        rows.append(dict(target=t['id'], subset=t['subset']))
        print(t['id'], t['subset'], flush=True)
    atomic_json(args.output / 'summary.json', dict(protocol=protocol, targets=rows))
    print('REFINEMENT_PREFIX_CACHE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
