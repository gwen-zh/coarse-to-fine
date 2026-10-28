"""Development-only next-depth prediction and fully predicted-prefix pilot."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from depth_generator import DepthGenerator, sequence_ids
from train_residual_vq import atomic_json


@torch.no_grad()
def validate(model, items):
    model.eval()
    rows = []
    for item in items:
        stage_rows = []
        for stage, size in enumerate(model.sizes):
            current = torch.full_like(item['sequence'], size)
            lower = [item['labels'][..., i] for i in range(stage)]
            logits = model(item['features'], item['sequence'], lower, current, stage)
            labels = item['labels'][..., stage]
            stage_rows.append(dict(ce=F.cross_entropy(logits.reshape(-1, size), labels.flatten()).item(),
                                   accuracy=(logits.argmax(-1) == labels).float().mean().item()))
        rows.append(dict(target=item['target'], stages=stage_rows))
    means = [dict(ce=float(np.mean([r['stages'][i]['ce'] for r in rows])),
                  accuracy=float(np.mean([r['stages'][i]['accuracy'] for r in rows])))
             for i in range(len(model.sizes))]
    return dict(means=means, targets=rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['residual', 'parallel', 'direct'], required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--samples', type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    directory = args.output / args.mode
    directory.mkdir(exist_ok=True, parents=True)
    manifest = json.loads((args.tokens / 'manifest.json').read_text())
    train, validation = [], []
    for t in manifest['targets']:
        with np.load(args.features / (t['id'] + '.npz')) as a:
            # Explicitly forbid using teacher_h in the generation pilot.
            features = torch.from_numpy(a['sampled_h'].copy())[None]
            direct_labels = a['fine'].copy()
        if args.mode == 'direct':
            labels = torch.from_numpy(direct_labels).long()[:, None]
        else:
            with np.load(args.tokens / t['tokens']) as a:
                labels = torch.from_numpy(a[args.mode + '_tokens'].copy()).long()
        seq = sequence_ids(t['sequence'])[None]
        assert labels.shape[0] == seq.shape[1] == features.shape[1] == t['length']
        item = dict(target=t['id'], features=features, sequence=seq, labels=labels[None])
        (train if t['subset'] == 'train' else validation).append(item)
    sizes = [4096] if args.mode == 'direct' else [32, 512, 4096]
    protocol = dict(version=1, mode=args.mode, sizes=sizes, epochs=args.epochs, seed=args.seed,
                    width=128, layers=2, lr=0.0003, training_crop=256, samples=args.samples,
                    context='sampled_h only: frozen published generator, sequence-only coarse sampling, 5 steps',
                    legacy_context_network_calls_per_protein=7,
                    train_proteins=len(train), validation_proteins=len(validation),
                    token_manifest_sha256=hashlib.sha256((args.tokens / 'manifest.json').read_bytes()).hexdigest(),
                    feature_protocol=json.loads((args.features / 'protocol.json').read_text()),
                    selection='minimum development protein-mean teacher-prefix CE/log(vocab), averaged across stages',
                    warning='Small development-only interface pilot. Exact-sequence split, one training structure '
                            'per sequence, cached upstream context. Not a standalone generator, fair final '
                            'rate/compute comparison, ensemble generalization claim or independent test.')
    path = directory / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Generator pilot protocol mismatch')
    atomic_json(path, protocol)
    model = DepthGenerator(sizes=sizes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['lr'], weight_decay=0.01)
    rng = torch.Generator().manual_seed(args.seed + 900)
    best_score, best_epoch, best = float('inf'), 0, None
    history = []
    started = perf_counter()
    print(json.dumps(dict(event='DEPTH_GENERATOR_TRAINING_STARTED', **protocol)), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for index in torch.randperm(len(train), generator=rng).tolist():
            item = train[index]
            n = item['sequence'].shape[1]
            start = int(torch.randint(max(1, n - 256 + 1), (), generator=rng))
            end = min(n, start + 256)
            seq = item['sequence'][:, start:end]
            features = item['features'][:, start:end]
            labels = item['labels'][:, start:end]
            stage = int(torch.randint(len(sizes), (), generator=rng))
            size = sizes[stage]
            lower = []
            for level in range(stage):
                ids = labels[..., level].clone()
                ids[torch.rand(ids.shape, generator=rng) < 0.1] = sizes[level]
                lower.append(ids)
            current = torch.full_like(seq, size)
            mask = torch.ones_like(seq, dtype=torch.bool)
            if stage == 0 and len(sizes) > 1:
                probability = 0.1 + 0.9 * float(torch.rand((), generator=rng))
                mask = torch.rand(seq.shape, generator=rng) < probability
                if not mask.any():
                    mask[0, 0] = True
                current = torch.where(mask, current, labels[..., stage])
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, seq, lower, current, stage)
            loss = F.cross_entropy(logits[mask], labels[..., stage][mask])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite generator loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        if epoch == 1 or epoch % 2 == 0 or epoch == args.epochs:
            result = validate(model, validation)
            score = float(np.mean([r['ce'] / np.log(k) for r, k in zip(result['means'], sizes)]))
            if score < best_score:
                best_score, best_epoch = score, epoch
                best = copy.deepcopy(model.state_dict())
                torch.save(dict(state_dict=best, protocol=protocol, epoch=epoch,
                                parameters=sum(p.numel() for p in model.parameters())), directory / 'best.pt')
            row = dict(epoch=epoch, train_ce=total / len(train), validation=result['means'],
                       normalized_selection_score=score, elapsed_seconds=perf_counter() - started)
            history.append(row)
            atomic_json(directory / 'history.json', history)
            print(json.dumps(row), flush=True)
        model.train()
    model.load_state_dict(best)
    predictions = directory / 'predictions'
    predictions.mkdir(exist_ok=True)
    rows = []
    for item in validation:
        for sample in range(args.samples):
            sample_seed = (args.seed * 1000000 + sample * 10000 +
                           int(hashlib.sha256(item['target'].encode()).hexdigest()[:8], 16)) % (2 ** 32)
            t0 = perf_counter()
            # Generation has no access to item['labels'].
            tokens, calls = model.generate(item['features'], item['sequence'], seed=sample_seed)
            np.savez_compressed(predictions / f"{item['target']}_sample{sample}.npz", tokens=tokens[0].numpy())
            rows.append(dict(target=item['target'], sample=sample, seed=sample_seed,
                             adapter_network_calls=calls, elapsed_seconds=perf_counter() - t0))
    atomic_json(directory / 'summary.json', dict(protocol=protocol, selected_epoch=best_epoch,
                parameters=sum(p.numel() for p in model.parameters()),
                elapsed_seconds=perf_counter() - started, teacher_prefix_validation=validate(model, validation),
                samples=rows))
    print('DEPTH_GENERATOR_PILOT_COMPLETE', args.mode, flush=True)


if __name__ == '__main__':
    main()
