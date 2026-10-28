"""Post-hoc frozen-model diagnosis; oracle arms are not deployable methods."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from diagnostic_io import atomic_json
from mixed_hierarchical_io import data, model, tokenizer, sample_seed, sha
from continue_mixed_hierarchical_generator import protocol as parent_protocol
from compare_two_level_ae_author import metrics

RUN = Path('artifacts/mixed_generation_failure_audit_v1')
PARENT = Path('artifacts/mixed_hierarchical_followup_v1')
ARMS = ['native_coarse_native_fine', 'native_coarse_predicted_fine',
        'predicted_coarse_native_fine', 'predicted_coarse_predicted_fine']


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    audit = json.loads((RUN / 'protocol.json').read_text())
    for path, digest in audit['sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('Frozen audit input changed: ' + path)
    p = parent_protocol()
    seed = p['seeds'][args.index]
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    items, stats = data(p, seed)
    selected_ids = set(audit['train_ids'] + audit['validation_ids'])
    selected = [x for x in items if x['row']['id'] in selected_ids]
    if len(selected) != len(selected_ids):
        raise RuntimeError('Incomplete fixed diagnostic cohort')
    generator = model(p, stats, device).eval()
    checkpoint = torch.load(PARENT / 'fine' / f'seed{seed}' / 'final.pt',
                            map_location='cpu', weights_only=False)
    generator.load_state_dict(checkpoint['model'])
    decoder = tokenizer(p, seed, device)
    out = RUN / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        raise RuntimeError('Audit already complete; do not overwrite')
    rows = []
    replay_max = 0.
    started = time.perf_counter()
    for item in selected:
        name = item['row']['id']
        h, native_c, native_f = item['features'], item['coarse'], item['fine']
        reference = item['bb'].numpy()
        # The same 50% masking policy and target-specific seed are used on both splits.
        torch.manual_seed(sample_seed(seed, name, 900))
        ce = float(generator.fine.denoise_loss(h, generator.normalize(native_c),
                                               native_f, rate=.5))
        native_bb = decoder.decode(native_c, native_f).numpy()
        records = []
        for sample in range(audit['samples_per_protein']):
            key = sample_seed(seed, name, sample)
            cached = PARENT / 'fine' / f'seed{seed}' / 'final_samples' / f'{name}_sample{sample}.npz'
            if item['row']['subset'] == 'validation':
                with np.load(cached) as z:
                    c = torch.from_numpy(z['coarse'].copy()).float()
                    f = torch.from_numpy(z['fine'].copy()).long()
                    saved_bb = z['backbone'].copy()
                deployed = decoder.decode(c, f).numpy()
                error = float(np.max(np.abs(deployed - saved_bb)))
                replay_max = max(replay_max, error)
                if error > 2e-5:
                    raise RuntimeError('Cached deployed coordinates fail exact-model replay')
                if name == audit['validation_ids'][0] and sample == 0:
                    replay_c, replay_f = generator.sample(h, key, p['coarse']['sampling_steps'],
                                                        p['fine']['sampling_steps'], p['fine']['temperature'])
                    torch.testing.assert_close(c, replay_c, atol=2e-5, rtol=2e-5)
                    if not torch.equal(f, replay_f):
                        raise RuntimeError('Deployed fine sampling does not reproduce saved output')
            else:
                c, f = generator.sample(h, key, p['coarse']['sampling_steps'],
                                        p['fine']['sampling_steps'], p['fine']['temperature'])
                deployed = decoder.decode(c, f).numpy()
            teacher_f = generator.fine.sample(h, generator.normalize(native_c),
                                              (key + 104729) % (2**32),
                                              p['fine']['sampling_steps'], p['fine']['temperature'])
            variants = {
                ARMS[0]: native_bb,
                ARMS[1]: decoder.decode(native_c, teacher_f).numpy(),
                ARMS[2]: decoder.decode(c, native_f).numpy(),
                ARMS[3]: deployed,
            }
            records.append(dict(sample=sample, seed=key,
                                fine_accuracy_with_native_coarse=float((teacher_f == native_f).float().mean()),
                                fine_accuracy_with_predicted_coarse=float((f == native_f).float().mean()),
                                arms={arm: metrics(bb, reference) for arm, bb in variants.items()}))
        rows.append(dict(target=name, subset=item['row']['subset'],
                         native_coarse_half_masked_ce=ce, samples=records))
        atomic_json(out / 'progress.json', dict(completed=len(rows), total=len(selected),
                                               last_target=name, seconds=time.perf_counter() - started))
        print('MIXED_FAILURE_AUDIT', seed, name, len(rows), len(selected), flush=True)
    groups = {}
    for subset in ['train', 'validation']:
        group = [r for r in rows if r['subset'] == subset]
        arm_metrics = {}
        for arm in ARMS:
            keys = group[0]['samples'][0]['arms'][arm]
            arm_metrics[arm] = {k: float(np.mean([np.mean([s['arms'][arm][k] for s in r['samples']])
                                                for r in group])) for k in keys}
        groups[subset] = dict(proteins=len(group),
                             native_coarse_half_masked_ce=float(np.mean([r['native_coarse_half_masked_ce'] for r in group])),
                             arms=arm_metrics)
    result = dict(status='complete', seed=seed, groups=groups, targets=rows,
                  replay_max_abs_error=replay_max, seconds=time.perf_counter() - started,
                  protocol_sha256=sha(RUN / 'protocol.json'), scope=audit['scope'])
    atomic_json(out / 'complete.json', result)
    lines = ['混合生成失败定位：冻结模型、事后开发诊断', '=' * 52, '', audit['scope'], '']
    for subset, group in groups.items():
        lines.append(f"{subset}: proteins={group['proteins']}, matched50%maskCE={group['native_coarse_half_masked_ce']:.5f}")
        for arm, m in group['arms'].items():
            lines.append(f"  {arm}: CA RMSD={m['ca_rmsd']:.4f} Å, CN={100*m['cn_rate']:.2f}%")
    (out / 'results_zh.rst').write_text('\n'.join(lines) + '\n')
    print('MIXED_FAILURE_AUDIT_COMPLETE', seed, json.dumps(groups), flush=True)


if __name__ == '__main__':
    main()
