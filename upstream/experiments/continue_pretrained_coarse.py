"""Feed sampled pretrained coarse plans to a fixed existing residual refiner.

Only sequence-derived context and predicted coarse tokens enter generation.
This diagnoses compatibility; it is not a matched-compute LoRA ablation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from depth_generator import DepthGenerator, sequence_ids
from train_residual_vq import atomic_json


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    for name in ['tokens', 'features', 'coarse', 'refiner', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    checkpoint = torch.load(args.refiner, map_location='cpu', weights_only=True)
    cfg = checkpoint.get('source_protocol', checkpoint.get('protocol'))
    model = DepthGenerator(sizes=cfg['sizes'], width=cfg['width'], layers=cfg['layers'])
    model.load_state_dict(checkpoint['refiner'] if 'refiner' in checkpoint else checkpoint['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    coarse_protocol = json.loads((args.coarse / 'protocol.json').read_text())
    token_hash = hashlib.sha256((args.tokens / 'manifest.json').read_bytes()).hexdigest()
    if coarse_protocol['token_manifest_sha256'] != token_hash or cfg['token_manifest_sha256'] != token_hash:
        raise ValueError('Coarse and refiner token semantics/split manifest differ')
    protocol = dict(version=1, samples=cfg['samples'], mode='residual',
                    refiner_sha256=hashlib.sha256(args.refiner.read_bytes()).hexdigest(),
                    coarse_adapter_sha256=hashlib.sha256((args.coarse / 'best_adapter.pt').read_bytes()).hexdigest(),
                    token_manifest_sha256=token_hash, features=cfg['feature_protocol'],
                    note='External pretrained-LoRA coarse + fixed existing tiny residual refiner. '
                         'References never enter sampling. Different coarse architecture and cost from old '
                         'tiny-coarse control; not a causal LoRA-only or matched-compute ablation.')
    args.output.mkdir(exist_ok=True, parents=True)
    path = args.output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Continuation output protocol mismatch')
    atomic_json(path, protocol)
    pred_dir = args.output / 'predictions'
    pred_dir.mkdir(exist_ok=True)
    rows = []
    for item in json.loads((args.tokens / 'manifest.json').read_text())['targets']:
        if item['subset'] != 'validation':
            continue
        name = item['id']
        with np.load(args.features / (name + '.npz')) as a:
            features = torch.from_numpy(a['sampled_h'].copy())[None]
        with np.load(args.coarse / 'predictions' / (name + '.npz')) as a:
            coarse = torch.from_numpy(a['coarse'].copy()).long()
        sequence = sequence_ids(item['sequence'])[None]
        if coarse.shape != (cfg['samples'], item['length']):
            raise ValueError('Unexpected coarse sample dimensions')
        for sample in range(cfg['samples']):
            tokens, calls = model.refine_coarse(features, sequence, coarse[sample:sample+1])
            if not torch.equal(tokens[..., 0], coarse[sample:sample+1]):
                raise RuntimeError('External coarse changed during continuation')
            np.savez_compressed(pred_dir / f'{name}_sample{sample}.npz', tokens=tokens[0].numpy())
            rows.append(dict(target=name, sample=sample, refiner_calls=calls))
        print('CONTINUED', name, flush=True)
    atomic_json(args.output / 'summary.json', dict(protocol=protocol, samples=rows,
                proteins=len({r['target'] for r in rows}), coarse_mismatches=0))
    print('PRETRAINED_COARSE_CONTINUATION_COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
