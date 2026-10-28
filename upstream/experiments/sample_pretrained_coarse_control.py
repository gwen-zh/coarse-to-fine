"""Sample the exact initialized, untrained coarse model used by the LoRA run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_pretrained_coarse import (load_state_dict_from_lightning_ckpt,
    initialize_new_coarse_semantics, install_coarse_lora,
    EsmSequenceTokenizer, tokenize_sequence, evaluate, sample_coarse, generation_seed, atomic_json)


def main():
    p = argparse.ArgumentParser()
    for name in ['tokens', 'checkpoint', 'encoder_weights', 'adapted', 'output']:
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Allocate a Slurm GPU')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    adapted = json.loads((args.adapted / 'summary.json').read_text())
    cfg = adapted['protocol']
    if hashlib.sha256((args.tokens / 'manifest.json').read_bytes()).hexdigest() != cfg['token_manifest_sha256']:
        raise ValueError('Control token manifest differs from adapted run')
    model = load_state_dict_from_lightning_ckpt(args.checkpoint, device=torch.device('cuda:0'))
    bundle = torch.load(args.tokens / 'residual_bridge.pt', weights_only=True)
    encoder = torch.load(args.encoder_weights, map_location='cpu', weights_only=False, mmap=True)
    old = encoder.get('encoder_weight', encoder)['small_codebook.embeddings'].clone().cuda()
    del encoder
    saved_initialization = json.loads((args.adapted / 'semantic_initialization.json').read_text())
    initialization = initialize_new_coarse_semantics(model, bundle['centers'][0].cuda(), old,
                                                     initialization=saved_initialization)
    del old
    install_coarse_lora(model, cfg['last_blocks'], cfg['rank'])
    model.eval()  # Match original trainable flags and construction context exactly.
    tokenizer = EsmSequenceTokenizer()
    items = []
    for item in json.loads((args.tokens / 'manifest.json').read_text())['targets']:
        if item['subset'] != 'validation':
            continue
        with np.load(args.tokens / item['tokens']) as a:
            labels = torch.from_numpy(a['residual_tokens'][:, 0].copy()).long().cuda()[None]
        items.append(dict(target=item['id'], labels=labels,
                          sequence=tokenize_sequence(item['sequence'], tokenizer).cuda()[None]))
    validation = evaluate(model, items)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / 'initialization_diagnostic.json', dict(
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        observed=validation, expected=adapted['initial_validation'],
        initialization='Replayed saved interpolation; original trainable flags retained'))
    # Guard against unnoticed initialization drift before calling this a paired control.
    for metric in ['mean_ce', 'mean_accuracy']:
        if abs(validation[metric] - adapted['initial_validation'][metric]) > 1e-5:
            raise ValueError(f'Untrained initialization mismatch: {metric}: '
                             f"observed={validation[metric]}, expected={adapted['initial_validation'][metric]}")
    print('UNTRAINED_PRETRAINED_INITIALIZATION_VERIFIED', json.dumps({k:v for k,v in validation.items() if k != 'targets'}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(cfg, epochs=0, control='Exact semantic-remapped pretrained initialization; no optimization')
    path = args.output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Control protocol mismatch')
    atomic_json(path, protocol)
    # No adapters have been trained; keep an explicit checkpoint provenance artifact.
    torch.save(dict(protocol=protocol, epoch=0, initialization=initialization), args.output / 'best_adapter.pt')
    predictions = args.output / 'predictions'
    predictions.mkdir(exist_ok=True)
    for item in items:
        values = [sample_coarse(model, item['sequence'], generation_seed(item['target'], sample))[0].cpu().numpy()
                  for sample in range(2)]
        np.savez_compressed(predictions / (item['target'] + '.npz'), coarse=np.stack(values))
        print('CONTROL_SAMPLED', item['target'], flush=True)
    atomic_json(args.output / 'summary.json', dict(protocol=protocol, validation=validation, samples=2*len(items)))
    print('PRETRAINED_COARSE_UNTRAINED_CONTROL_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
