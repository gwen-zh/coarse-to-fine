"""Fixed-budget development repair with separate coarse/fine composition controls."""
import argparse
import copy
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha, save_checkpoint, restore_checkpoint
from mixed_hierarchical_io import data, tokenizer, model as original_model, sample_seed
from regularized_mixed_generator import RegularizedMixedGenerator, fine_training_loss
from compare_two_level_ae_author import metrics
from structural_metrics import aligned_rmsd, ensemble_summary

RUN = Path('artifacts/mixed_generator_repair_v1')


def protocol():
    p = json.loads((RUN / 'protocol.json').read_text())
    for path, digest in p['source_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('Frozen repair input changed: ' + path)
    return p


def build(p, stats, decoder, head='ce'):
    return RegularizedMixedGenerator(stats['mean'], stats['std'], **p['model'],
                                     fine_head=head, codes=decoder.model.fine_book.codes)


def configure(generator, stage):
    generator.requires_grad_(False).eval()
    part = generator.coarse_flow if stage == 'coarse' else generator.fine
    part.requires_grad_(True).train()
    return part


def update(generator, decoder, item, optimizer, p, stage):
    configure(generator, stage)
    optimizer.zero_grad(set_to_none=True)
    if stage == 'coarse':
        loss, terms = generator.coarse_loss(item['features'], item['coarse'], decoder,
                                             p['coarse']['geometry_weight'])
    else:
        loss, terms = fine_training_loss(generator, item['features'], item['coarse'], item['fine'],
                                         p['fine']['jitter_std'], p['fine']['label_smoothing'])
    loss.backward()
    part = generator.coarse_flow if stage == 'coarse' else generator.fine
    norm = torch.nn.utils.clip_grad_norm_(part.parameters(), 1.)
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise RuntimeError('Nonfinite repair update')
    optimizer.step()
    return dict(loss=float(loss.detach()), **{k: float(v) for k, v in terms.items()})


@torch.no_grad()
def likelihood(generator, items, seed, ids):
    generator.eval()
    before = torch.get_rng_state().clone()
    rows = []
    for item in items:
        if item['row']['id'] not in ids:
            continue
        torch.manual_seed(sample_seed(seed, item['row']['id'], 900))
        loss = generator.fine.denoise_loss(item['features'], generator.normalize(item['coarse']),
                                           item['fine'], rate=.5)
        rows.append(dict(target=item['row']['id'], nll=float(loss)))
    torch.set_rng_state(before)
    if len(rows) != len(ids):
        raise RuntimeError('Likelihood cohort incomplete')
    return dict(nll=float(np.mean([r['nll'] for r in rows])), targets=rows,
                scope='Native coarse, fixed50% mask, no training jitter or smoothed validation labels; descriptive development likelihood.')


@torch.no_grad()
def coarse_evaluation(generator, decoder, items, seed, ids, steps):
    generator.eval()
    selected = [x for x in items if x['row']['id'] in ids]
    rows = []
    for i, item in enumerate(selected):
        h, target = item['features'], item['centers'].numpy()
        other = selected[(i + 1) % len(selected)]['features']
        wrong_h = other[torch.linspace(0, len(other)-1, len(h)).round().long()]
        samples = []
        for sample in range(2):
            key = sample_seed(seed, item['row']['id'], sample)
            centers = decoder.coarse_centers(generator.sample_coarse(h, key, steps), len(h)).numpy()
            wrong = decoder.coarse_centers(generator.sample_coarse(wrong_h, key, steps), len(h)).numpy()
            samples.append(dict(aligned_center_rmsd=aligned_rmsd(centers, target),
                                wrong_sequence_aligned_center_rmsd=aligned_rmsd(wrong, target)))
        rows.append(dict(target=item['row']['id'], samples=samples))
    return dict(metrics={k: float(np.mean([np.mean([s[k] for s in r['samples']]) for r in rows]))
                         for k in rows[0]['samples'][0]}, targets=rows)


@torch.no_grad()
def generation(generator, fine_model, decoder, items, p, seed, directory,
               coarse_source, samples):
    generator.eval()
    fine_model.eval()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    original = Path(p['parent_run']) / 'fine' / f'seed{seed}' / 'final_samples'
    repaired = RUN / 'coarse' / f'seed{seed}' / 'final_samples'
    for item in items:
        name = item['row']['id']
        if name not in p['generation_probe_ids']:
            continue
        records, coords = [], []
        for sample in range(samples):
            key = sample_seed(seed, name, sample)
            if coarse_source == 'fresh_repaired':
                coarse = generator.sample_coarse(item['features'], key, p['coarse']['sampling_steps'])
            elif coarse_source == 'native':
                coarse = item['coarse']
            else:
                cache = original if coarse_source == 'original' else repaired
                with np.load(cache / f'{name}_sample{sample}.npz') as z:
                    coarse = torch.from_numpy(z['coarse'].copy()).float()
            fine = fine_model.fine.sample(item['features'], fine_model.normalize(coarse),
                                          (key + 104729) % (2**32), p['fine']['sampling_steps'], 1.)
            backbone = decoder.decode(coarse, fine).numpy()
            records.append(dict(sample=sample, seed=key, metrics=metrics(backbone, item['bb'].numpy())))
            coords.append(backbone[:, 1])
            np.savez_compressed(directory / f'{name}_sample{sample}.npz',
                                backbone=backbone, coarse=coarse.numpy(), fine=fine.numpy())
        rows.append(dict(target=name, samples=records,
                         spread=ensemble_summary(coords)['mean_pairwise_ca_rmsd']))
    if len(rows) != len(p['generation_probe_ids']):
        raise RuntimeError('Incomplete fixed generation cohort')
    aggregate = {k: float(np.mean([np.mean([s['metrics'][k] for s in r['samples']]) for r in rows]))
                 for k in rows[0]['samples'][0]['metrics']}
    result = dict(status='complete', proteins=len(rows), samples_per_protein=samples,
                  coarse_source=coarse_source, metrics=aggregate,
                  spread=float(np.mean([r['spread'] for r in rows])), targets=rows,
                  scope='Fixed unfiltered draws, no reference acceptance/selection. Native coarse arm is an oracle diagnostic only.')
    atomic_json(directory / 'evaluation.json', result)
    return result


def real_smoke(p, parent, items, stats, decoder, seed):
    results = []
    for stage, head in [('coarse', 'ce'), ('fine', 'ce'), ('fine', 'metric')]:
        torch.manual_seed(seed)
        m = build(p, stats, decoder, head)
        part = configure(m, stage)
        opt = torch.optim.AdamW(part.parameters(), lr=p[stage]['lr'], weight_decay=p['weight_decay'])
        for item in items[:4]:
            update(m, decoder, item, opt, p, stage)
        path = RUN / f'smoke_{stage}_{head}.pt'
        rng = np.random.default_rng(seed)
        save_checkpoint(path, m, opt, rng)
        expected = update(m, decoder, items[0], opt, p, stage)
        expected_model = {k: v.clone() for k, v in m.state_dict().items()}
        expected_adam = copy.deepcopy(opt.state_dict())
        expected_rng = torch.get_rng_state().clone()
        restored = build(p, stats, decoder, head)
        part = configure(restored, stage)
        opt2 = torch.optim.AdamW(part.parameters(), lr=p[stage]['lr'], weight_decay=p['weight_decay'])
        restore_checkpoint(path, restored, opt2, np.random.default_rng(99), torch.device('cpu'))
        actual = update(restored, decoder, items[0], opt2, p, stage)
        if expected != actual or not torch.equal(expected_rng, torch.get_rng_state()):
            raise RuntimeError('Real-data replay loss/RNG differs')
        for k, v in expected_model.items():
            if not torch.equal(v, restored.state_dict()[k]):
                raise RuntimeError('Real-data replay model differs: ' + k)
        for key, values in expected_adam['state'].items():
            for name, value in values.items():
                if not torch.equal(value, opt2.state_dict()['state'][key][name]):
                    raise RuntimeError('Real-data Adam replay differs')
        results.append(dict(stage=stage, head=head, exact_next_update_replay=True,
                            parameters=sum(x.numel() for x in m.parameters())))
    m.eval()
    with torch.no_grad():
        c, f = m.sample(items[0]['features'], seed, 3, 3)
        bb = decoder.decode(c, f)
    if not torch.isfinite(bb).all() or any(v.requires_grad or v.grad is not None for v in decoder.parameters()):
        raise RuntimeError('Frozen tokenizer / finite generation smoke failed')
    atomic_json(RUN / 'smoke.json', dict(status='passed', tests=results,
                                        sequence_only_sample_finite=True, tokenizer_frozen=True))
    print('REPAIR_SMOKE_PASSED', json.dumps(results), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['smoke', 'coarse', 'fine'], required=True)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    p = protocol()
    seed = p['seeds'][args.index if args.stage != 'fine' else args.index // 2]
    head = p['fine_heads'][args.index % 2] if args.stage == 'fine' else 'ce'
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    parent = json.loads(Path(p['parent_protocol']).read_text())
    items, stats = data(parent, seed)
    train = [x for x in items if x['row']['subset'] == 'train']
    decoder = tokenizer(parent, seed)
    if args.stage == 'smoke':
        real_smoke(p, parent, train, stats, decoder, seed)
        return
    if json.loads((RUN / 'smoke.json').read_text())['status'] != 'passed':
        raise RuntimeError('Real smoke required')
    generator = build(p, stats, decoder, head)
    stage = args.stage
    out = RUN / ('coarse' if stage == 'coarse' else f'fine_{head}') / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        raise RuntimeError('Completed run is immutable')
    coarse_sha = None
    if stage == 'fine':
        coarse_root = RUN / 'coarse' / f'seed{seed}'
        if json.loads((coarse_root / 'complete.json').read_text())['status'] != 'complete':
            raise RuntimeError('Complete coarse stage required')
        coarse_sha = sha(coarse_root / 'final.pt')
        saved = torch.load(coarse_root / 'final.pt', map_location='cpu', weights_only=False)
        generator.coarse_flow.load_state_dict({k[len('coarse_flow.'):]: v for k, v in saved['model'].items()
                                               if k.startswith('coarse_flow.')})
    part = configure(generator, stage)
    optimizer = torch.optim.AdamW(part.parameters(), lr=p[stage]['lr'], weight_decay=p['weight_decay'])
    torch.manual_seed(seed + (40000 if stage == 'fine' else 20000))
    rng = np.random.default_rng(seed + (40000 if stage == 'fine' else 20000))
    first, updates, trace = 0, 0, []
    ph = sha(RUN / 'protocol.json')
    if (out / 'last.pt').exists():
        saved = restore_checkpoint(out / 'last.pt', generator, optimizer, rng, torch.device('cpu'))
        if saved['protocol_sha256'] != ph or saved['coarse_checkpoint_sha256'] != coarse_sha:
            raise RuntimeError('Resume protocol mismatch')
        first, updates = saved['epoch'], saved['updates']
        trace = json.loads((out / 'trace.json').read_text())
    started = time.perf_counter()
    for epoch in range(first, p[stage]['epochs']):
        lr = p[stage]['min_lr'] + .5 * (p[stage]['lr'] - p[stage]['min_lr']) * (1 + math.cos(math.pi * epoch / (p[stage]['epochs']-1)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        records = []
        order = rng.permutation(len(train))
        epoch_start = time.perf_counter()
        for step, index in enumerate(order):
            records.append(update(generator, decoder, train[index], optimizer, p, stage))
            updates += 1
            if step % 150 == 0:
                atomic_json(out / 'progress.json', dict(epoch=epoch+1, epochs=p[stage]['epochs'],
                                                       step=step+1, updates=updates,
                                                       loss=records[-1]['loss'], seconds=time.perf_counter()-started))
        record = dict(epoch=epoch+1, updates=updates, lr=lr, seconds=time.perf_counter()-epoch_start,
                      terms={k: float(np.mean([r[k] for r in records])) for k in records[0]},
                      training_order_sha256=__import__('hashlib').sha256(order.tobytes()).hexdigest())
        trace.append(record)
        atomic_json(out / 'trace.json', trace)
        save_checkpoint(out / 'last.pt', generator, optimizer, rng, model_config=generator.config,
                        epoch=epoch+1, updates=updates, seed=seed, head=head,
                        protocol_sha256=ph, coarse_checkpoint_sha256=coarse_sha)
        if epoch+1 in p[stage]['evaluation_epochs']:
            save_checkpoint(out / f'epoch{epoch+1}.pt', generator, optimizer, rng,
                            model_config=generator.config, epoch=epoch+1, updates=updates,
                            seed=seed, head=head, protocol_sha256=ph, coarse_checkpoint_sha256=coarse_sha)
            if stage == 'coarse':
                evaluation = coarse_evaluation(generator, decoder, items, seed,
                                                p['generation_probe_ids'], p['coarse']['sampling_steps'])
            else:
                evaluation = {subset: likelihood(generator, items, seed, ids) for subset, ids in
                              [('train', p['train_probe_ids']), ('validation', p['validation_ids'])]}
                generation(generator, generator, decoder, items, p, seed,
                           out / f'epoch{epoch+1}_samples', 'repaired', 2)
            atomic_json(out / f'epoch{epoch+1}_evaluation.json', evaluation)
        print('GENERATOR_REPAIR_EPOCH', stage, head, seed, json.dumps(record), flush=True)
    save_checkpoint(out / 'final.pt', generator, optimizer, rng, model_config=generator.config,
                    epoch=p[stage]['epochs'], updates=updates, seed=seed, head=head,
                    protocol_sha256=ph, coarse_checkpoint_sha256=coarse_sha)
    if stage == 'coarse':
        old = original_model(parent, stats, 'cpu')
        old.load_state_dict(torch.load(Path(p['parent_run']) / 'fine' / f'seed{seed}' / 'final.pt',
                                       map_location='cpu', weights_only=False)['model'])
        evaluations = dict(repaired_coarse_original_fine=generation(generator, old, decoder, items, p, seed,
                            out / 'final_samples', 'fresh_repaired', p['final_samples']))
    else:
        evaluations = {}
        for source in ['original', 'repaired']:
            evaluations[source] = generation(generator, generator, decoder, items, p, seed,
                                              out / f'final_{source}_samples', source, p['final_samples'])
        evaluations['native_coarse_diagnostic'] = generation(generator, generator, decoder, items, p, seed,
                                                             out / 'native_coarse_samples', 'native', 2)
    atomic_json(out / 'complete.json', dict(status='complete', stage=stage, head=head, seed=seed,
                  epochs=p[stage]['epochs'], updates=updates, seconds=time.perf_counter()-started,
                  parameters=sum(x.numel() for x in generator.parameters()), evaluations=evaluations,
                  protocol_sha256=ph, coarse_checkpoint_sha256=coarse_sha))
    print('GENERATOR_REPAIR_COMPLETE', stage, head, seed, flush=True)


if __name__ == '__main__':
    main()
