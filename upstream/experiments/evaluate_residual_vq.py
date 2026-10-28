"""Decode learned cumulative representations through the actual frozen trunk."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from residual_quantizer import PrefixQuantizer, DecoderAdapter
from train_residual_vq import atomic_json
from structural_metrics import aligned_rmsd, summarize_backbone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Hierarchical_VQ_VAE'))
from esm.models.new_vqvae import ProjectedDecoder
from esm.sdk.api import ESMProtein


def decode_embedding(decoder, embedding):
    x = torch.cat([decoder.embed.weight[4098:4099], embedding,
                   decoder.embed.weight[4097:4098]])[None]
    n = x.shape[1]
    ids = torch.zeros((1, n), dtype=torch.long, device=x.device)
    x, _, _ = decoder.decoder_stack.forward(x, affine=None, affine_mask=None,
                    sequence_id=ids, chain_id=ids,
                    attn_mask=torch.zeros((1, n, n), device=x.device))
    _, bb = decoder.affine_output_projection(x, affine=None,
                    affine_mask=torch.zeros((1, n), dtype=torch.bool, device=x.device))
    return bb[0, 1:-1]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--bridge', choices=['adapter', 'fine_codebook'], default='adapter')
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    split = json.loads((args.control / 'summary.json').read_text())['split']
    groups = [t for t in json.loads(args.manifest.read_text())['all_groups']
              if t['split'] == 'development' and split[t['id']] == 'validation']
    decoder = ProjectedDecoder(d_model=1280, n_heads=20, n_layers=30)
    state = torch.load(args.weights / 'decoder_checkpoint_110.pth', map_location='cpu', mmap=True, weights_only=False)
    decoder.load_state_dict(state.get('decode_proj', state), strict=True)
    del state
    decoder.to(args.device).eval().requires_grad_(False)
    fine_codes = None
    if args.bridge == 'fine_codebook':
        state = torch.load(args.weights / 'encoder_checkpoint_110.pth', map_location='cpu', mmap=True, weights_only=False)
        fine_codes = state.get('encoder_weight', state)['codebook.embeddings'].detach().clone().to(args.device)
        del state
    models = {}
    for mode in ['residual', 'parallel']:
        checkpoint = torch.load(args.run / mode / 'best.pt', map_location=args.device, weights_only=True)
        q = PrefixQuantizer(mode=mode).to(args.device)
        adapter = DecoderAdapter().to(args.device)
        q.load_state_dict(checkpoint['quantizer'])
        adapter.load_state_dict(checkpoint['adapter'])
        models[mode] = q.eval(), adapter.eval()
    directory = args.run / ('decoded' if args.bridge == 'adapter' else 'decoded_fine_bridge')
    directory.mkdir(exist_ok=True)
    records = []
    for t in groups:
        path = directory / (t['id'] + '.json')
        if path.exists():
            records.append(json.loads(path.read_text()))
            continue
        with np.load(args.cache / (t['id'] + '.npz')) as a:
            z = torch.from_numpy(a['latent'].copy()).to(args.device)
            fine = torch.from_numpy(a['fine'].copy()).long().to(args.device)
        reference = ESMProtein.from_pdb(ROOT / t['paths'][0])
        assert reference.sequence == t['sequence']
        ref = reference.coordinates[:, :3].numpy()
        row = dict(target=t['id'], length=t['length'], methods={})
        arrays = {}
        embeddings = dict(fine_reference=[decoder.embed(fine)])
        if fine_codes is not None:
            recovered = PrefixQuantizer.nearest(z, fine_codes)
            if not torch.equal(recovered, fine):
                raise ValueError('Original fine codebook does not reproduce cached fine IDs')
        for mode, (q, adapter) in models.items():
            prefixes, tokens = q(z)
            embeddings[mode] = ([adapter(prefix) for prefix in prefixes] if fine_codes is None else
                                [decoder.embed(PrefixQuantizer.nearest(prefix, fine_codes)) for prefix in prefixes])
            for i, ids in enumerate(tokens):
                arrays[f'{mode}_tokens_{i}'] = ids.cpu().numpy()
        for method, values in embeddings.items():
            rows = []
            for i, embedding in enumerate(values):
                prediction = decode_embedding(decoder, embedding).cpu().numpy()
                arrays[f'{method}_backbone_{i}'] = prediction
                rows.append(dict(ca_rmsd=aligned_rmsd(prediction[:, 1], ref[:, 1]),
                                 backbone_rmsd=aligned_rmsd(prediction.reshape(-1, 3), ref.reshape(-1, 3)),
                                 metrics=summarize_backbone(prediction, [ref[:, 1]])))
            row['methods'][method] = rows
        np.savez_compressed(directory / (t['id'] + '.npz'), **arrays)
        atomic_json(path, row)
        records.append(row)
        print(json.dumps(row), flush=True)
    methods = {}
    for method in ['residual', 'parallel', 'fine_reference']:
        means = []
        for level in range(len(records[0]['methods'][method])):
            rows = [r['methods'][method][level] for r in records]
            means.append(dict(ca_rmsd=float(np.mean([r['ca_rmsd'] for r in rows])),
                              backbone_rmsd=float(np.mean([r['backbone_rmsd'] for r in rows])),
                              ca_clash_validity=float(np.mean([r['metrics']['upstream_ca_clash_valid'] for r in rows]))))
        methods[method] = dict(prefix_means=means,
                              nonincreasing_ca_rmsd_proteins=sum(all(v[i]['ca_rmsd'] <= v[i-1]['ca_rmsd']
                                                                   for i in range(1, len(v)))
                                      for v in [r['methods'][method] for r in records]))
    name = 'structure_comparison.json' if args.bridge == 'adapter' else 'structure_comparison_fine_bridge.json'
    atomic_json(args.run / name,
                dict(note='Frozen-encoder reconstruction on development validation. No generator or independent-test gain claimed.',
                     bridge=args.bridge, proteins=len(records), methods=methods, targets=records))
    print(json.dumps(methods), flush=True)
    print('RESIDUAL_VQ_STRUCTURE_EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
