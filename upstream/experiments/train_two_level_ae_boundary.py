"""Fixed16k 2x2 diagnostic; unchanged16k baseline is reused, three new arms train."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from continue_two_level_autoencoder import evaluate, initialize, snapshot
from diagnostic_io import atomic_json
from two_level_ae_boundary import NeighborFineAutoencoder, boundary_terms, boundary_training_loss

RUN = Path('artifacts/development/two_level_ae_boundary_v1')


def make_state(seed, parent, context):
    state = initialize(seed, parent)
    if context:
        before = torch.get_rng_state().clone()
        state['model'] = NeighborFineAutoencoder(state['model'])
        if not torch.equal(before, torch.get_rng_state()):
            raise RuntimeError('Context adapter changed augmentation RNG')
        state['optimizer'] = torch.optim.AdamW(state['model'].parameters(), lr=parent['lr'], weight_decay=1e-4)
    return state


def update(state, train, weights, use_boundary, augmentation_digest):
    step = state['step']
    if step % len(train) == 0:
        state['order'] = state['rng'].permutation(len(train)).tolist()
    item = train[state['order'][step % len(train)]]
    q, _ = torch.linalg.qr(torch.randn(3, 3)); q[:, 0] *= torch.linalg.det(q)
    augmentation_digest.update(item['row']['id'].encode()); augmentation_digest.update(q.numpy().tobytes())
    x = item['bb'] @ q
    model, optimizer = state['model'], state['optimizer']
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss, terms = boundary_training_loss(model(x), x, weights, use_boundary)
    if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(norm) or norm <= 0: raise RuntimeError('Invalid gradient')
    gradients = None
    if step == 0:
        gradients = {name: float(sum(v.grad.square().sum() for v in getattr(model, name).parameters() if v.grad is not None).sqrt())
                     for name in ['coarse_encoder', 'coarse_decoder', 'fine_encoder', 'fine_decoder']}
        if isinstance(model, NeighborFineAutoencoder):
            gradients['new_context_columns'] = float(model.fine_decoder[0].weight.grad[:, model.original_fine_inputs:].norm())
        if not all(np.isfinite(v) and v > 0 for v in gradients.values()): raise RuntimeError('Disconnected module')
    optimizer.step(); state['step'] += 1
    return dict(step=state['step'], loss=float(loss), terms={k: float(v) for k, v in terms.items()}), gradients


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=['smoke', 'train'], required=True)
    parser.add_argument('--index', type=int, default=0); args = parser.parse_args()
    pp = RUN/'protocol.json'; p = json.loads(pp.read_text()); ph = hashlib.sha256(pp.read_bytes()).hexdigest()
    for source, digest in p['sha256'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest: raise RuntimeError('Frozen source changed: '+source)
    parent = json.loads(Path(p['parent_protocol']).read_text())
    items = []
    for row in parent['targets']:
        with np.load(row['reference']) as data: bb = torch.tensor(data['backbone'], dtype=torch.float32)
        if bb.shape != (row['length'], 3, 3) or not torch.isfinite(bb).all(): raise RuntimeError('Reference mismatch')
        items.append(dict(row=row, bb=bb))
    train = [r for r in items if r['row']['subset'] == 'train']
    if args.stage == 'train' and json.loads((RUN/'smoke.json').read_text())['status'] != 'passed': raise RuntimeError('Smoke incomplete')
    cases = p['jobs'] if args.stage == 'train' else [dict(arm=arm, seed=parent['seeds'][0]) for arm in p['arms']]
    if args.stage == 'train': cases = [cases[args.index]]
    smoke_results = []
    for case in cases:
        arm, seed = case['arm'], case['seed']; config = p['arms'][arm]
        state = make_state(seed, parent, config['context'])
        model = state['model']; out = RUN/('smoke' if args.stage=='smoke' else 'runs')/arm/f'seed{seed}'
        out.mkdir(parents=True, exist_ok=True)
        if (out/'progress.json').exists() or (out/'complete.json').exists(): raise RuntimeError('Existing run; explicit resume required')
        parameters = sum(v.numel() for v in model.parameters())
        if parameters != config['parameters']: raise RuntimeError('Parameter mismatch')
        started = time.perf_counter(); trace = []; gradients = None; digest = hashlib.sha256()
        steps = p['smoke_steps'] if args.stage=='smoke' else p['steps']
        def save_and_evaluate():
            step = state['step']; directory = out/f'step{step}'
            torch.save(dict(**snapshot(state), seed=seed, arm=arm, protocol_sha256=ph), out/f'step{step}.pt')
            rows = evaluate(model, items, parent['weights'], directory, step)
            # fixed_loss is the COMMON parent objective. Added boundary objective is logged separately.
            for row, item in zip(rows, items):
                with np.load(directory/(row['target']+'.npz')) as data: pred = torch.from_numpy(data['full'].copy())
                bond, angle = boundary_terms(pred, item['bb'])
                row['boundary_loss_terms'] = dict(bond=float(bond), angle=float(angle))
                row['active_fixed_loss'] = row['fixed_loss'] + (parent['weights']['bond']*float(bond)+parent['weights']['angle']*float(angle) if config['boundary'] else 0.)
            atomic_json(directory/'evaluation.json', dict(status='complete', step=step, arm=arm, seed=seed, targets=rows))
            subsets = {subset: {key: float(np.mean([r['metrics']['full'][key] for r in rows if r['subset']==subset]))
                       for key in ['rmsd', 'cn_rate', 'boundary_rate', 'within_rate']} for subset in ['train','validation']}
            print('AE_BOUNDARY_EVALUATED', arm, seed, step, json.dumps(subsets), flush=True)
        if args.stage == 'train': save_and_evaluate()
        for _ in range(steps):
            record, first_gradients = update(state, train, parent['weights'], config['boundary'], digest)
            if first_gradients is not None: gradients = first_gradients
            if state['step']==1 or state['step']%100==0:
                record['seconds'] = time.perf_counter()-started; trace.append(record)
                atomic_json(out/'progress.json', record)
            if args.stage=='train' and state['step'] in p['checkpoints']: save_and_evaluate()
        if args.stage == 'smoke':
            model.eval()
            with torch.no_grad():
                x = train[0]['bb']; c, f = model.encode(x)
                torch.testing.assert_close(model(x)['backbone'], model.decode(c, f, len(x))['backbone'], atol=0, rtol=0)
            # Test serialized full-state continuation: same next rotation/order, Adam state and model update.
            path = out/'checkpoint.pt'; torch.save(snapshot(state), path)
            record_a, _ = update(state, train, parent['weights'], config['boundary'], hashlib.sha256())
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            from continue_two_level_autoencoder import restore, verify_parent_model
            restored = make_state(seed, parent, config['context'])
            restore(restored, torch.load(path, weights_only=False))
            record_b, _ = update(restored, train, parent['weights'], config['boundary'], hashlib.sha256())
            if record_a != record_b: raise RuntimeError('Real-data checkpoint continuation changed')
            verify_parent_model(restored['model'], expected)
            smoke_results.append(dict(arm=arm, parameters=parameters, steps=steps, gradients=gradients,
                                      full_state_resume_exact=True, latent_only_decode_exact=True))
        else:
            training = dict(status='complete', arm=arm, seed=seed, steps=steps, parameters=parameters,
                            seconds=time.perf_counter()-started, module_gradients=gradients, trace=trace,
                            augmentation_sha256=digest.hexdigest(), device='cpu', threads=parent['threads'])
            atomic_json(out/'training.json', training)
            atomic_json(out/'complete.json', dict(status='complete', arm=arm, seed=seed, protocol_sha256=ph))
            print('AE_BOUNDARY_COMPLETE', arm, seed, flush=True)
    if args.stage=='smoke':
        atomic_json(RUN/'smoke.json', dict(status='passed', arms=smoke_results, scope='Real reference training, gradients, latent-only decoding and serialized next-step replay.'))
        print('AE_BOUNDARY_SMOKE_PASSED', flush=True)


if __name__=='__main__': main()
