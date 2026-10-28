"""Paired medium-supervised versus direct head adaptation on frozen features.

Selection uses protein-mean fine CE on sampled coarse inputs of the 40 development
validation sequences. The independent manifest holdout is never read or encoded.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from medium_adapter import MediumAdapter, DirectAdapter, matched_direct_rank
from quantization_control import assignments


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


@torch.no_grad()
def evaluate(adapter, items, weight, bias, permutation=None, save=None):
    records = []
    for item in items:
        row = dict(target=item['target'], modes={})
        predictions = {}
        for mode in ['teacher', 'sampled']:
            h = torch.from_numpy(item[mode + '_h']).to(weight.device)
            y = torch.from_numpy(item['fine']).long().to(weight.device)
            amended, medium = adapter(h, permutation=permutation)
            logits = F.linear(amended, weight, bias)
            ids = logits.argmax(-1)
            m = dict(fine_ce=F.cross_entropy(logits, y).item(),
                     fine_accuracy=(ids == y).float().mean().item())
            if medium is not None:
                labels = torch.from_numpy(item['medium']).long().to(weight.device)
                m['medium_ce'] = F.cross_entropy(medium, labels).item()
                m['medium_accuracy'] = (medium.argmax(-1) == labels).float().mean().item()
            row['modes'][mode] = m
            predictions[mode + '_fine'] = ids.cpu().numpy()
        if save is not None:
            np.savez_compressed(save / (item['target'] + '.npz'), **predictions)
        records.append(row)
    means = {mode: {key: float(np.mean([r['modes'][mode][key] for r in records]))
                    for key in records[0]['modes'][mode]} for mode in ['teacher', 'sampled']}
    return dict(proteins=len(records), means=means, targets=records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    groups = [t for t in json.loads(args.manifest.read_text())['all_groups'] if t['split'] == 'development']
    groups.sort(key=lambda t: t['sequence_hash'])
    control = json.loads((args.control / 'summary.json').read_text())
    centers = np.load(args.control / 'dictionaries.npz')['kmeans512']
    train, validation = [], []
    for t in groups:
        subset = control['split'][t['id']]
        with np.load(args.features / (t['id'] + '.npz')) as a:
            item = {key: a[key].copy() for key in ['teacher_h', 'sampled_h', 'fine', 'medium', 'latent']}
        item['target'] = t['id']
        item['kmeans_medium'] = assignments(item.pop('latent'), centers)
        (train if subset == 'train' else validation).append(item)
    assert len(train) == control['train_proteins'] and len(validation) == control['validation_proteins']
    projection = torch.load(args.features / 'fine_projection.pt', weights_only=True)
    weight, bias = (projection[key][:4096].to(args.device) for key in ['weight', 'bias'])
    width = weight.shape[1]
    protocol = dict(version=1, seeds=args.seeds, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                    medium_loss_weight=0.25, training_modes=['teacher', 'sampled'],
                    selection='minimum protein-mean sampled fine CE; initial identity eligible',
                    train_proteins=len(train), validation_proteins=len(validation),
                    note='Small development adaptation, not independent benchmark or ensemble training.')
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Training output protocol mismatch')
    atomic_json(protocol_path, protocol)
    h = torch.from_numpy(np.concatenate([t[m + '_h'] for t in train for m in ['teacher', 'sampled']])).to(args.device)
    fine = torch.from_numpy(np.concatenate([t['fine'] for t in train for _ in range(2)])).long().to(args.device)
    summaries = []
    for seed in args.seeds:
        for variant in ['direct', 'released512', 'kmeans512']:
            directory = args.output / f'{variant}_seed{seed}'
            directory.mkdir(exist_ok=True)
            if (directory / 'summary.json').exists():
                summaries.append(json.loads((directory / 'summary.json').read_text()))
                continue
            torch.manual_seed(seed)
            template = MediumAdapter(width=width)
            medium_count = sum(p.numel() for p in template.parameters())
            adapter = (DirectAdapter(width=width, rank=matched_direct_rank(width, medium_count))
                       if variant == 'direct' else template).to(args.device)
            key = 'kmeans_medium' if variant == 'kmeans512' else 'medium'
            train_medium = torch.from_numpy(np.concatenate([t[key] for t in train for _ in range(2)])).long().to(args.device)
            val = [dict(t, medium=t[key]) for t in validation]
            initial = evaluate(adapter, val, weight, bias)
            best_score = initial['means']['sampled']['fine_ce']
            best_epoch, best = 0, copy.deepcopy(adapter.state_dict())
            history = [dict(epoch=0, validation=initial['means'])]
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)
            generator = torch.Generator(device=args.device).manual_seed(seed + 1000)
            for epoch in range(1, args.epochs + 1):
                adapter.train()
                order = torch.randperm(len(h), generator=generator, device=args.device)
                total_loss = 0.0
                for start in range(0, len(h), args.batch_size):
                    indices = order[start:start + args.batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    amended, medium = adapter(h[indices])
                    loss = F.cross_entropy(F.linear(amended, weight, bias), fine[indices])
                    if medium is not None:
                        loss = loss + protocol['medium_loss_weight'] * F.cross_entropy(medium, train_medium[indices])
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite adaptation loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item() * len(indices)
                adapter.eval()
                result = evaluate(adapter, val, weight, bias)
                score = result['means']['sampled']['fine_ce']
                if score < best_score:
                    best_score, best_epoch, best = score, epoch, copy.deepcopy(adapter.state_dict())
                row = dict(variant=variant, seed=seed, epoch=epoch, train_loss=total_loss / len(h), validation=result['means'])
                history.append(row)
                print(json.dumps(row), flush=True)
                atomic_json(directory / 'history.json', history)
            adapter.load_state_dict(best)
            torch.save(dict(state_dict={k: v.cpu() for k, v in best.items()}, variant=variant,
                            seed=seed, epoch=best_epoch, width=width,
                            rank=adapter.down.out_features if variant == 'direct' else adapter.embedding.shape[1],
                            feature_protocol=json.loads((args.features / 'protocol.json').read_text())),
                       directory / 'adapter.pt')
            selected = evaluate(adapter, val, weight, bias, save=directory)
            shuffled = None
            if variant != 'direct':
                permutation = torch.randperm(512, generator=torch.Generator().manual_seed(9001)).to(args.device)
                shuffled = evaluate(adapter, val, weight, bias, permutation=permutation)['means']
            summary = dict(variant=variant, seed=seed, selected_epoch=best_epoch,
                           parameters=sum(p.numel() for p in adapter.parameters()),
                           baseline=initial['means'], selected=selected, shuffled_medium=shuffled)
            atomic_json(directory / 'summary.json', summary)
            summaries.append(summary)
            atomic_json(args.output / 'summary.json', dict(protocol=protocol, runs=summaries))
    print('MEDIUM_ADAPTATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
