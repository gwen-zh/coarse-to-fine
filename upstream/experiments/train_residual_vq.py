"""Train EMA dictionaries and prefix-supervised decoder-input adapters.

Frozen encoder and frozen decoder. Adapter targets are the pretrained fine
embedding, followed by real coordinate reconstruction evaluation in a separate
process. This pilot is not coordinate-loss training or an adapted generator.
"""
import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from residual_quantizer import PrefixQuantizer, DecoderAdapter


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


@torch.no_grad()
def evaluate(quantizer, adapter, items, table):
    records = []
    counts = [torch.zeros(k, dtype=torch.long, device=table.device) for k in quantizer.sizes]
    for target, z, fine in items:
        prefixes, tokens = quantizer(z)
        rows = []
        for level, (prefix, ids) in enumerate(zip(prefixes, tokens)):
            counts[level] += torch.bincount(ids, minlength=quantizer.sizes[level])
            rows.append(dict(latent_mse=F.mse_loss(prefix, z).item(),
                             embedding_mse=F.mse_loss(adapter(prefix), table[fine]).item()))
        records.append(dict(target=target, prefixes=rows))
    mean = [dict(latent_mse=float(np.mean([r['prefixes'][i]['latent_mse'] for r in records])),
                 embedding_mse=float(np.mean([r['prefixes'][i]['embedding_mse'] for r in records])))
            for i in range(len(counts))]
    utilization = []
    for size, c in zip(quantizer.sizes, counts):
        p = c[c > 0].double() / c.sum()
        utilization.append(dict(size=size, used=int((c > 0).sum()), perplexity=float(torch.exp(-(p * p.log()).sum()))))
    return dict(proteins=len(items), means=mean, utilization=utilization, targets=records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    split = json.loads((args.control / 'summary.json').read_text())['split']
    groups = sorted((t for t in json.loads(args.manifest.read_text())['all_groups']
                     if t['split'] == 'development'), key=lambda t: t['sequence_hash'])
    train, validation = [], []
    for t in groups:
        with np.load(args.cache / (t['id'] + '.npz')) as a:
            item = (t['id'], torch.from_numpy(a['latent'].copy()).to(args.device),
                    torch.from_numpy(a['fine'].copy()).long().to(args.device))
        (train if split[t['id']] == 'train' else validation).append(item)
    state = torch.load(args.weights / 'decoder_checkpoint_110.pth', map_location='cpu', mmap=True, weights_only=False)
    table = state.get('decode_proj', state)['embed.weight'].detach().clone().to(args.device)
    del state
    torch.save(dict(fine_embedding=table.cpu()), args.output / 'fine_embedding.pt')
    z = torch.cat([t[1] for t in train])
    fine = torch.cat([t[2] for t in train])
    protocol = dict(version=1, sizes=[32, 512, 4096], epochs=args.epochs, seed=args.seed,
                    batch_size=args.batch_size, lr=0.001, ema_decay=0.99,
                    initialization='training-data samples; sequential residuals for RVQ',
                    restart='EMA count < 0.1 every 5 epochs from training residuals/data',
                    supervision='equal embedding MSE on each of three prefixes',
                    selection='minimum validation protein-mean prefix embedding MSE',
                    train_proteins=len(train), validation_proteins=len(validation),
                    train_residues=len(z), frozen_encoder=True, frozen_decoder=True,
                    parallel_control='each dictionary quantizes the same z; prefix is average of quantized vectors',
                    caveat='Same codebook capacities and prefix code budget, unlike released independent decoder paths. '
                           'Embedding-distillation training; not coordinate-loss training or generator training.')
    path = args.output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('RVQ run protocol mismatch')
    atomic_json(path, protocol)
    print(json.dumps(dict(event='RVQ_TRAINING_STARTED', **protocol)), flush=True)
    summaries = []
    for mode in ['residual', 'parallel']:
        directory = args.output / mode
        directory.mkdir(exist_ok=True)
        if (directory / 'summary.json').exists():
            summaries.append(json.loads((directory / 'summary.json').read_text()))
            continue
        started = perf_counter()
        torch.manual_seed(args.seed)
        quantizer = PrefixQuantizer(mode=mode).to(args.device)
        quantizer.initialize(z, seed=args.seed)
        adapter = DecoderAdapter().to(args.device)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=protocol['lr'], weight_decay=0.01)
        generator = torch.Generator(device=args.device).manual_seed(args.seed + 500)
        best_score = float('inf')
        best_epoch, best = 0, None
        history = []
        for epoch in range(1, args.epochs + 1):
            order = torch.randperm(len(z), generator=generator, device=z.device)
            loss_sum = 0.0
            for start in range(0, len(z), args.batch_size):
                indices = order[start:start + args.batch_size]
                prefixes, _ = quantizer(z[indices], update=True)
                target = table[fine[indices]]
                optimizer.zero_grad(set_to_none=True)
                loss = sum(F.mse_loss(adapter(prefix), target) for prefix in prefixes) / len(prefixes)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite adapter loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                loss_sum += loss.item() * len(indices)
            restarts = quantizer.restart_unused(z, args.seed + epoch) if epoch % 5 == 0 else None
            result = evaluate(quantizer, adapter, validation, table)
            score = float(np.mean([m['embedding_mse'] for m in result['means']]))
            if score < best_score:
                best_score, best_epoch = score, epoch
                best = dict(quantizer=copy.deepcopy(quantizer.state_dict()), adapter=copy.deepcopy(adapter.state_dict()))
                # Save a reviewable checkpoint immediately, not only after training.
                torch.save(dict(**best, mode=mode, epoch=epoch, protocol=protocol), directory / 'best.pt')
            row = dict(mode=mode, epoch=epoch, train_embedding_mse=loss_sum / len(z),
                       validation=result['means'], utilization=result['utilization'],
                       restarted=restarts, elapsed_seconds=perf_counter() - started)
            history.append(row)
            atomic_json(directory / 'history.json', history)
            print(json.dumps(row), flush=True)
        quantizer.load_state_dict(best['quantizer'])
        adapter.load_state_dict(best['adapter'])
        result = evaluate(quantizer, adapter, validation, table)
        summary = dict(mode=mode, selected_epoch=best_epoch, elapsed_seconds=perf_counter() - started,
                       validation=result)
        atomic_json(directory / 'summary.json', summary)
        summaries.append(summary)
        atomic_json(args.output / 'summary.json', dict(protocol=protocol, methods=summaries))
    print('RESIDUAL_VQ_TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
