"""Recover published coarse IDs from cached encoder latents; retain native fine IDs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_residual_vq import atomic_json


def nearest(z, centers):
    return (z.square().sum(-1, keepdim=True) - 2*z@centers.T + centers.square().sum(-1)).argmin(-1)


def main():
    p = argparse.ArgumentParser()
    for name in ['tokens', 'encoder', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--latent-cache', type=Path)
    p.add_argument('--validation-only', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    source = json.loads((args.tokens / 'manifest.json').read_text())
    state = torch.load(args.encoder, map_location='cpu', mmap=True, weights_only=False)
    state = state.get('encoder_weight', state)
    coarse_codes = state['small_codebook.embeddings'].clone()
    fine_codes = state['codebook.embeddings'].clone()
    del state
    protocol = dict(version=1, semantics='published small32 and original fine4096; no token remapping',
                    source_manifest_sha256=hashlib.sha256((args.tokens/'manifest.json').read_bytes()).hexdigest(),
                    encoder_sha256=hashlib.sha256(args.encoder.read_bytes()).hexdigest(),
                    latent_cache=str(args.latent_cache.resolve()) if args.latent_cache else None,
                    validation_only=args.validation_only)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Native token export provenance changed')
    atomic_json(path, protocol)
    rows = []
    for item in source['targets']:
        if args.validation_only and item['subset'] != 'validation':
            continue
        source_path = args.latent_cache / (item['id']+'.npz') if args.latent_cache else args.tokens / item['tokens']
        with np.load(source_path) as a:
            z = torch.from_numpy(a['latent'].copy())
            fine = torch.from_numpy(a['fine'].copy()).long()
            cached_coarse = torch.from_numpy(a['coarse'].copy()).long() if 'coarse' in a.files else None
        if z.shape != (item['length'], 128) or not torch.isfinite(z).all():
            raise ValueError('Invalid source latent')
        # Confirm original-codebook semantics before using cached fine labels.
        if not torch.equal(nearest(z, fine_codes), fine):
            raise ValueError('Cached fine IDs do not match published encoder: ' + item['id'])
        coarse = nearest(z, coarse_codes)
        if cached_coarse is not None and not torch.equal(coarse, cached_coarse):
            raise ValueError('Original coarse semantics do not match source')
        path = args.output / (item['id'] + '.npz')
        np.savez_compressed(path, coarse=coarse.numpy(), fine=fine.numpy())
        rows.append(dict(item, tokens=path.name))
    atomic_json(args.output/'manifest.json', dict(protocol=protocol, targets=rows))
    print('NATIVE_TOKEN_EXPORT_COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
