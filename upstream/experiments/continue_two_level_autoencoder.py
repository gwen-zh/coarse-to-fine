"""Frozen convergence diagnostic; recover missing Adam/RNG state by exact replay."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from diagnostic_io import atomic_json
from hierarchical_latent_contract import compose_levels, pool_ca, remove_block_translation
from train_two_level_autoencoder import quality
from two_level_autoencoder import TwoLevelAutoencoder, training_loss

RUN = Path('artifacts/development/two_level_autoencoder_convergence_v1')


def initialize(seed, protocol):
    torch.set_num_threads(protocol['threads'])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TwoLevelAutoencoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['lr'], weight_decay=1e-4)
    return dict(model=model, optimizer=optimizer, rng=rng, order=[], step=0)


def update(state, train, weights):
    """Identical data/augmentation/loss/optimizer order to the parent run."""
    step = state['step']
    if step % len(train) == 0:
        state['order'] = state['rng'].permutation(len(train)).tolist()
    x = train[state['order'][step % len(train)]]['bb']
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    q[:, 0] *= torch.linalg.det(q)
    x = x @ q
    model, optimizer = state['model'], state['optimizer']
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(x)
    loss, terms = training_loss(output, x, weights)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite loss; do not exclude')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(norm) or norm <= 0:
        raise RuntimeError('Invalid gradients')
    optimizer.step()
    state['step'] += 1
    return dict(step=state['step'], loss=float(loss), terms={k: float(v) for k, v in terms.items()})


def snapshot(state):
    return dict(model=copy.deepcopy(state['model'].state_dict()),
                optimizer=copy.deepcopy(state['optimizer'].state_dict()),
                torch_rng=torch.get_rng_state().clone(),
                numpy_rng=copy.deepcopy(state['rng'].bit_generator.state),
                order=list(state['order']), step=state['step'])


def restore(state, saved):
    state['model'].load_state_dict(saved['model'])
    state['optimizer'].load_state_dict(saved['optimizer'])
    state['rng'].bit_generator.state = copy.deepcopy(saved['numpy_rng'])
    state['order'] = list(saved['order'])
    state['step'] = saved['step']
    torch.set_rng_state(saved['torch_rng'])


def verify_parent_model(model, saved):
    current = model.state_dict()
    if set(current) != set(saved):
        raise RuntimeError('Parent model keys differ')
    differences = {k: float((current[k] - saved[k]).abs().max()) for k in current}
    mismatch = [k for k in current if not torch.equal(current[k], saved[k])]
    if mismatch:
        raise RuntimeError(f'Exact 2000-step replay failed: {len(mismatch)} tensors; max error {max(differences.values())}')
    return dict(all_tensors_bitwise_equal=True, tensors=len(current), max_abs_error=max(differences.values()))


def boundary_quality(backbone, block_size=4):
    x = backbone.detach().cpu().numpy()
    distances = np.linalg.norm(x[:-1, 2] - x[1:, 0], axis=-1)
    bad = ~np.isfinite(distances) | (distances < 1.) | (distances > 1.8)
    ids = np.arange(len(x) - 1)
    boundary = ids // block_size != (ids + 1) // block_size
    result = {}
    for name, mask in [('within', ~boundary), ('boundary', boundary)]:
        count = int(mask.sum())
        result[name + '_bonds'] = count
        result[name + '_bad'] = int(bad[mask].sum())
        result[name + '_rate'] = float(bad[mask].mean()) if count else None
    return result


@torch.no_grad()
def evaluate(model, items, weights, directory, step):
    """Fixed orientations; no reference-based selection and no RNG consumption."""
    before = torch.get_rng_state().clone()
    model.eval()
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in items:
        row, x = item['row'], item['bb']
        n = len(x)
        target = x - x[:, 1].mean(0)
        centers, residual = pool_ca(target), remove_block_translation(target)
        c, f = model.encode(x)
        full = model.decode(c, f, n)
        zero_fine = model.decode(c, torch.zeros_like(f), n)
        zero_coarse = model.decode(torch.zeros_like(c), f, n)
        drift = float((pool_ca(full['backbone']) - pool_ca(zero_fine['backbone'])).abs().max())
        if drift > 2e-5:
            raise RuntimeError('Fine changed coarse centers')
        variants = dict(full=full['backbone'], zero_fine=zero_fine['backbone'],
                        zero_coarse=zero_coarse['backbone'],
                        oracle_coarse=model.decode_with_anchors(centers, f, n)['backbone'],
                        oracle_fine=compose_levels(full['coarse'], residual))
        metrics = {key: dict(**quality(value, x), **boundary_quality(value)) for key, value in variants.items()}
        rot = x.new_tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        rotated = model(x @ rot)['backbone']
        metrics['rotated_input'] = dict(**quality(rotated, x @ rot), **boundary_quality(rotated))
        loss, terms = training_loss(dict(**full, z_c=c, z_f=f), x, weights)
        rows.append(dict(target=row['id'], subset=row['subset'], length=n,
                         fixed_loss=float(loss), loss_terms={k: float(v) for k, v in terms.items()},
                         coarse_rmse=float((full['coarse'] - centers).square().sum(-1).mean().sqrt()),
                         zero_coarse_rmse=float((zero_coarse['coarse'] - centers).square().sum(-1).mean().sqrt()),
                         fine_mse=float((full['fine_residual'] - residual).square().mean()),
                         zero_fine_mse=float((zero_fine['fine_residual'] - residual).square().mean()),
                         centroid_invariance_max_error=drift, metrics=metrics))
        np.savez_compressed(directory / (row['id'] + '.npz'), z_c=c.numpy(), z_f=f.numpy(),
                            coarse=full['coarse'].numpy(), **{k: v.numpy() for k, v in variants.items()})
    if not torch.equal(before, torch.get_rng_state()):
        raise RuntimeError('Evaluation changed training random state')
    atomic_json(directory / 'evaluation.json', dict(status='complete', step=step, targets=rows))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    pp = RUN / 'protocol.json'
    protocol = json.loads(pp.read_text())
    protocol_hash = hashlib.sha256(pp.read_bytes()).hexdigest()
    for path, digest in protocol['sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Frozen input changed: ' + path)
    parent = Path(protocol['parent_run'])
    p = json.loads((parent / 'protocol.json').read_text())
    seed = p['seeds'][args.index]
    out = RUN / f'seed{seed}'
    out.mkdir(exist_ok=True)
    if (out / 'progress.json').exists() or (out / 'complete.json').exists():
        raise RuntimeError('Existing run; explicit resume required')
    state = initialize(seed, p)
    items = []
    for row in p['targets']:
        with np.load(row['reference']) as data:
            bb = torch.tensor(data['backbone'], dtype=torch.float32)
        if bb.shape != (row['length'], 3, 3) or not torch.isfinite(bb).all():
            raise RuntimeError('Reference mismatch')
        items.append(dict(row=row, bb=bb))
    train = [item for item in items if item['row']['subset'] == 'train']
    started = time.perf_counter()
    replay_trace = []
    for _ in range(p['steps']):
        record = update(state, train, p['weights'])
        if record['step'] == 1 or record['step'] % 100 == 0:
            replay_trace.append(record)
    saved = torch.load(parent / f'seed{seed}/final.pt', map_location='cpu', weights_only=False)
    if saved['seed'] != seed or saved['steps'] != p['steps']:
        raise RuntimeError('Wrong parent checkpoint')
    replay = verify_parent_model(state['model'], saved['model'])
    old_trace = json.loads((parent / f'seed{seed}/training.json').read_text())['trace']
    trace_exact = len(old_trace) == len(replay_trace) and all(
        a['step'] == b['step'] and a['loss'] == b['loss'] and a['terms'] == b['terms']
        for a, b in zip(old_trace, replay_trace))
    if not trace_exact:
        raise RuntimeError('Original training trace did not replay exactly')
    replay.update(seed=seed, step=state['step'], original_loss_trace_exact=True,
                  seconds=time.perf_counter() - started, optimizer_restored_by_replay=True)
    atomic_json(out / 'replay.json', replay)
    print('AE_CONVERGENCE_REPLAY_PASSED', seed, json.dumps(replay), flush=True)

    def save_and_evaluate():
        step = state['step']
        torch.save(dict(**snapshot(state), seed=seed, protocol_sha256=protocol_hash), out / f'step{step}.pt')
        rows = evaluate(state['model'], items, p['weights'], out / f'step{step}', step)
        if step == p['steps']:
            original = json.loads((parent / f'seed{seed}/step{step}/evaluation.json').read_text())
            original_by_id = {r['target']: r for r in original['targets']}
            for row in rows:
                for mode in row['metrics']:
                    for key in ['rmsd', 'cn_rate', 'clashes_per_residue']:
                        if row['metrics'][mode][key] != original_by_id[row['target']]['metrics'][mode][key]:
                            raise RuntimeError('Parent evaluation changed')
        summary = {}
        for subset in ['train', 'validation']:
            selected = [r for r in rows if r['subset'] == subset]
            summary[subset] = {k: float(np.mean([r['metrics']['full'][k] for r in selected]))
                               for k in ['rmsd', 'cn_rate', 'within_rate', 'boundary_rate']}
        atomic_json(out / 'progress.json', dict(step=step, phase='evaluated', seconds=time.perf_counter()-started, subsets=summary))
        print('AE_CONVERGENCE_EVALUATED', seed, step, json.dumps(summary), flush=True)

    save_and_evaluate()
    trace = []
    continuation_started = time.perf_counter()
    while state['step'] < max(protocol['checkpoints']):
        record = update(state, train, p['weights'])
        if state['step'] % 100 == 0:
            record['seconds'] = time.perf_counter() - continuation_started
            trace.append(record)
            atomic_json(out / 'progress.json', dict(**record, phase='training'))
        if state['step'] in protocol['checkpoints']:
            save_and_evaluate()
    atomic_json(out / 'training.json', dict(status='complete', seed=seed, replay=replay,
                total_step=state['step'], additional_updates=state['step']-p['steps'],
                continuation_seconds=time.perf_counter()-continuation_started,
                seconds=time.perf_counter()-started, trace=trace))
    atomic_json(out / 'complete.json', dict(status='complete', seed=seed, protocol_sha256=protocol_hash,
                fixed_checkpoints=protocol['checkpoints'], no_checkpoint_selection=True))
    print('AE_CONVERGENCE_COMPLETE', seed, flush=True)


if __name__ == '__main__':
    main()
