"""Matched continuation / sampled-prefix / compensation-target experiment."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from depth_generator import DepthGenerator, sequence_ids
from refinement_generator import FrozenCoarseRefiner, compensation_labels, generation_seed
from train_residual_vq import atomic_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Hierarchical_VQ_VAE'))
from esm.models.residual_bridge import load_residual_bridge


@torch.no_grad()
def validate_generated(model, validation, bridge, seed, samples):
    rows = []
    for item in validation:
        for sample in range(samples):
            predicted, calls = model.generate(item['features'], item['sequence'],
                                             generation_seed(item['target'], sample, seed))
            expected = item['predicted'][sample:sample+1, :, 0]
            if not torch.equal(predicted[..., 0], expected):
                raise RuntimeError('Frozen coarse predictions changed; paired experiment invalid')
            error = F.mse_loss(bridge.prefix_latent(predicted), item['latent']).item()
            rows.append(dict(target=item['target'], sample=sample, latent_mse=error))
    return dict(mean_latent_mse=float(np.mean([r['latent_mse'] for r in rows])), targets=rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--prefixes', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', choices=['teacher', 'fixed', 'adaptive'], required=True)
    p.add_argument('--epochs', type=int, default=20)
    args = p.parse_args()
    torch.set_num_threads(8)
    directory = args.output / args.variant
    directory.mkdir(exist_ok=True, parents=True)
    source = torch.load(args.source / 'best.pt', map_location='cpu', weights_only=True)
    cfg = source['protocol']
    torch.manual_seed(cfg['seed'])
    base = DepthGenerator(sizes=cfg['sizes'], width=cfg['width'], layers=cfg['layers'])
    base.load_state_dict(source['state_dict'])
    model = FrozenCoarseRefiner(base)
    bridge, _ = load_residual_bridge(args.tokens / 'residual_bridge.pt')
    prefix_protocol = json.loads((args.prefixes / 'protocol.json').read_text())
    source_hash = hashlib.sha256((args.source / 'best.pt').read_bytes()).hexdigest()
    if prefix_protocol.get('source_sha256') != source_hash:
        raise ValueError('Frozen prefix cache and warm-start generator differ')
    groups = json.loads((args.tokens / 'manifest.json').read_text())['targets']
    train, validation = [], []
    for t in groups:
        with np.load(args.features / (t['id'] + '.npz')) as a:
            features = torch.from_numpy(a['sampled_h'].copy())[None]
        with np.load(args.tokens / t['tokens']) as a:
            labels = torch.from_numpy(a['residual_tokens'].copy()).long()[None]
        with np.load(args.cache / (t['id'] + '.npz')) as a:
            latent = torch.from_numpy(a['latent'].copy())[None]
        with np.load(args.prefixes / (t['id'] + '.npz')) as a:
            predicted = torch.from_numpy(a['predicted'].copy()).long()
        item = dict(target=t['id'], features=features, labels=labels, latent=latent,
                    sequence=sequence_ids(t['sequence'])[None], predicted=predicted)
        (train if t['subset'] == 'train' else validation).append(item)
    protocol = dict(version=1, variant=args.variant, mode='residual', samples=cfg['samples'],
                    seed=cfg['seed'], epochs=args.epochs, lr=0.0001,
                    source_checkpoint_sha256=source_hash,
                    prefix_protocol=prefix_protocol,
                    train_proteins=len(train), validation_proteins=len(validation),
                    coarse_frozen=True, train_stages=[1, 2], generated_prefix_probability=0.5,
                    selection='minimum validation final latent MSE with fully predicted prefixes; epoch 0 eligible',
                    warning='Small development mechanism pilot. Teacher has no predicted prefixes. Fixed/adaptive '
                            'share cached previous-policy prefixes, not online updated-policy sampling. '
                            'Only adaptive recomputes nearest residual targets. Same warm start, steps, lr, '
                            'crop, random draws, architecture and frozen coarse policy. Not a pretrained-main-model upgrade.')
    protocol_path = directory / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Refinement run protocol mismatch')
    atomic_json(protocol_path, protocol)
    optimizer = torch.optim.AdamW(model.refiner.parameters(), lr=protocol['lr'], weight_decay=0.01)
    generator = torch.Generator().manual_seed(cfg['seed'] + 1700)
    initial = validate_generated(model, validation, bridge, cfg['seed'], cfg['samples'])
    best_score, best_epoch = initial['mean_latent_mse'], 0
    best = copy.deepcopy(model.refiner.state_dict())
    started = perf_counter()
    history = [dict(epoch=0, validation_latent_mse=best_score)]
    print(json.dumps(dict(event='PREFIX_REFINEMENT_TRAINING_STARTED', **protocol)), flush=True)

    def save_best():
        torch.save(dict(refiner=best, base=model.base.state_dict(), source_protocol=cfg,
                        protocol=protocol, epoch=best_epoch), directory / 'best.pt')

    save_best()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, generated_count, changed, target_count = 0., 0, 0, 0
        for index in torch.randperm(len(train), generator=generator).tolist():
            item = train[index]
            n = item['sequence'].shape[1]
            start = int(torch.randint(max(1, n - 256 + 1), (), generator=generator))
            end = min(n, start + 256)
            stage = int(torch.randint(1, len(model.sizes), (), generator=generator))
            sample = int(torch.randint(cfg['samples'], (), generator=generator))
            coin = float(torch.rand((), generator=generator))
            use_generated = args.variant != 'teacher' and coin < 0.5
            teacher = item['labels'][:, start:end]
            prefix = item['predicted'][sample:sample+1, start:end] if use_generated else teacher
            lower = [prefix[..., i] for i in range(stage)]
            labels = teacher[..., stage]
            if use_generated:
                generated_count += 1
                compensating = compensation_labels(bridge, item['latent'][:, start:end], lower, stage)
                changed += int((compensating != labels).sum())
                target_count += labels.numel()
                if args.variant == 'adaptive':
                    labels = compensating
            seq = item['sequence'][:, start:end]
            current = torch.full_like(seq, model.sizes[stage])
            optimizer.zero_grad(set_to_none=True)
            logits = model(item['features'][:, start:end], seq, lower, current, stage)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite prefix refinement loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.refiner.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        if epoch == 1 or epoch % 4 == 0 or epoch == args.epochs:
            validation_result = validate_generated(model, validation, bridge, cfg['seed'], cfg['samples'])
            score = validation_result['mean_latent_mse']
            if score < best_score:
                best_score, best_epoch = score, epoch
                best = copy.deepcopy(model.refiner.state_dict())
                save_best()
            row = dict(epoch=epoch, train_ce=total / len(train), validation_latent_mse=score,
                       generated_prefix_steps=generated_count,
                       changed_target_fraction=changed / target_count if target_count else None,
                       elapsed_seconds=perf_counter() - started)
            history.append(row)
            atomic_json(directory / 'history.json', history)
            print(json.dumps(row), flush=True)
    model.refiner.load_state_dict(best)
    predictions = directory / 'predictions'
    predictions.mkdir(exist_ok=True)
    rows = []
    for item in validation:
        for sample in range(cfg['samples']):
            seed = generation_seed(item['target'], sample, cfg['seed'])
            tokens, calls = model.generate(item['features'], item['sequence'], seed)
            if not torch.equal(tokens[..., 0], item['predicted'][sample:sample+1, :, 0]):
                raise RuntimeError('Coarse policy changed at selected checkpoint')
            np.savez_compressed(predictions / f"{item['target']}_sample{sample}.npz", tokens=tokens[0].numpy())
            rows.append(dict(target=item['target'], sample=sample, seed=seed, adapter_network_calls=calls))
    atomic_json(directory / 'summary.json', dict(protocol=protocol, selected_epoch=best_epoch,
                initial_validation_latent_mse=initial['mean_latent_mse'],
                selected_validation_latent_mse=best_score, coarse_mismatches=0,
                trainable_parameters=sum(p.numel() for p in model.refiner.parameters()),
                elapsed_seconds=perf_counter() - started, samples=rows))
    print('PREFIX_REFINEMENT_TRAINING_COMPLETE', args.variant, flush=True)


if __name__ == '__main__':
    main()
