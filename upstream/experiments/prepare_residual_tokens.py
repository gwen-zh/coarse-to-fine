"""Export usable prefix bridge and development-only next-stage token data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Hierarchical_VQ_VAE'))
from esm.models.residual_bridge import ResidualTokenBridge, load_residual_bridge
from train_residual_vq import atomic_json


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.weights / 'encoder_checkpoint_110.pth', mmap=True, map_location='cpu', weights_only=False)
    fine_codes = state.get('encoder_weight', state)['codebook.embeddings'].detach().clone()
    del state
    split = json.loads((args.control / 'summary.json').read_text())['split']
    groups = sorted((t for t in json.loads(args.manifest.read_text())['all_groups'] if t['split'] == 'development'),
                    key=lambda t: t['sequence_hash'])
    protocol = dict(version=1, source_run=str(args.run.resolve()),
                    source_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    warning='Development-only pilot data; exact-sequence split is not homology isolation. '
                            'One structure per sequence, not ensemble training. Held-out manifest groups excluded.')
    output_protocol = args.output / 'protocol.json'
    if output_protocol.exists() and json.loads(output_protocol.read_text()) != protocol:
        raise ValueError('Token export protocol mismatch')
    atomic_json(output_protocol, protocol)
    bridges = {}
    for mode in ['residual', 'parallel']:
        checkpoint = torch.load(args.run / mode / 'best.pt', map_location='cpu', weights_only=True)
        sizes = checkpoint['protocol']['sizes']
        centers = [checkpoint['quantizer'][f'centers_{i}'] for i in range(len(sizes))]
        bundle = dict(version=1, sizes=sizes, centers=centers, fine_codes=fine_codes,
                      mode=mode, metadata=dict(source_run=str(args.run.resolve()),
                      selected_epoch=checkpoint['epoch'], seed=checkpoint['protocol']['seed'],
                      encoder='frozen published ESM3 hierarchical encoder',
                      decoder='original fine lookup and frozen decoder trunk',
                      generator_trained=False, generator_retraining_required=True,
                      prefix_bits=[sum(int(np.ceil(np.log2(k))) for k in sizes[:i+1]) for i in range(len(sizes))]))
        path = args.output / f'{mode}_bridge.pt'
        torch.save(bundle, path)
        bridges[mode], _ = load_residual_bridge(path)
    rows = []
    for t in groups:
        with np.load(args.cache / (t['id'] + '.npz')) as a:
            latent = torch.from_numpy(a['latent'].copy())
        arrays = {}
        for mode, bridge in bridges.items():
            tokens = bridge.encode(latent)
            arrays[mode + '_tokens'] = tokens.numpy()
            arrays[mode + '_fine_bridge'] = bridge.fine_tokens(tokens).numpy()
            # The packaged bridge must reproduce the already evaluated token
            # assignments on every available development-validation target.
            checked = args.run / 'decoded_fine_bridge' / (t['id'] + '.npz')
            if checked.exists():
                with np.load(checked) as a:
                    for i in range(len(bridge.sizes)):
                        np.testing.assert_array_equal(tokens[:, i], a[f'{mode}_tokens_{i}'])
        np.savez_compressed(args.output / (t['id'] + '.npz'), **arrays)
        rows.append(dict(id=t['id'], sequence=t['sequence'], length=t['length'],
                         subset=split[t['id']], sequence_hash=t['sequence_hash'],
                         tokens=t['id'] + '.npz'))
    atomic_json(args.output / 'manifest.json', dict(protocol=protocol, targets=rows,
                training_proteins=sum(r['subset'] == 'train' for r in rows),
                validation_proteins=sum(r['subset'] == 'validation' for r in rows)))
    print('RESIDUAL_TOKEN_DATA_READY', len(rows), flush=True)


if __name__ == '__main__':
    main()
