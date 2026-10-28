"""First real pretrained-main-model coarse adaptation, on development data only.

The released network is reused, with attention LoRA in the last eight blocks
and its pretrained coarse head trained. This is not full-parameter fine-tuning.
"""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'hierarchical_ConfGen'))
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.utils.encoding import tokenize_sequence
from slm.models.model import add_eos_bos_tokens
from slm.utils.checkpoint_utils import load_state_dict_from_lightning_ckpt
from coarse_lora import (install_coarse_lora, initialize_new_coarse_semantics,
                         trainable_state, restore_trainable_state)
from refinement_generator import generation_seed
from train_residual_vq import atomic_json


def raw_coarse_logits(model, sequence, tokens, sigma):
    n = tokens.shape[1] - 2
    conditions = model.sigma_embedder(model._process_sigma(sigma))[:, None].expand(1, n, -1)
    return model.net(var_structure_tokens=tokens, sequence_tokens=sequence,
                     auxiliary_embeddings=conditions,
                     lengths=torch.tensor([n], device=tokens.device), scale=0).structure_logits[:, 1:-1, :32]


@torch.no_grad()
def evaluate(model, items):
    rows = []
    for item in items:
        sequence, labels = item['sequence'], item['labels']
        n = labels.shape[1]
        lengths = torch.tensor([n], device=labels.device)
        current = add_eos_bos_tokens(lengths, 34, 33, 35, torch.full_like(labels, 32))
        t = torch.ones(1, 1, device=labels.device) * (1 - 1e-3)
        sigma = model.noise(t)[0]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
            logits = raw_coarse_logits(model, sequence, current, sigma)
        rows.append(dict(target=item['target'], ce=F.cross_entropy(logits.float().flatten(0, 1), labels.flatten()).item(),
                         accuracy=(logits.argmax(-1) == labels).float().mean().item()))
    return dict(mean_ce=float(np.mean([r['ce'] for r in rows])),
                mean_accuracy=float(np.mean([r['accuracy'] for r in rows])), targets=rows)


@torch.no_grad()
def sample_coarse(model, sequence, seed, steps=5):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    n = sequence.shape[1] - 2
    lengths = torch.tensor([n], device=sequence.device)
    x = add_eos_bos_tokens(lengths, 34, 33, 35,
                           torch.full((1, n), 32, dtype=torch.long, device=sequence.device))
    times = torch.linspace(1, 1e-5, steps + 1, device=sequence.device)
    for step in range(steps):
        t = times[step] * torch.ones(1, 1, device=sequence.device)
        x = model._ddpm_update(x, t, sequence_tokens=sequence, dt=(1-1e-5)/steps,
                               lengths=lengths, scale=0)
        x = add_eos_bos_tokens(lengths, 34, 33, 35, x[:, 1:-1])
    logits, _ = model._model_wrapper_inference(x, sequence_tokens=sequence,
                  sigma=model.noise(times[-1] * torch.ones(1, 1, device=sequence.device))[0],
                  lengths=lengths, scale=0)
    return logits[:, 1:-1].argmax(-1)


def main():
    p = argparse.ArgumentParser()
    for key in ['tokens', 'checkpoint', 'encoder_weights', 'output']:
        p.add_argument('--' + key.replace('_', '-'), type=Path, required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--last-blocks', type=int, default=8)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--max-train-steps', type=int)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Allocate a Slurm GPU for pretrained-main-model training')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    args.output.mkdir(exist_ok=True, parents=True)
    device = torch.device('cuda:0')
    model = load_state_dict_from_lightning_ckpt(args.checkpoint, device=device)
    bundle = torch.load(args.tokens / 'residual_bridge.pt', weights_only=True)
    encoder = torch.load(args.encoder_weights, map_location='cpu', weights_only=False, mmap=True)
    old = encoder.get('encoder_weight', encoder)['small_codebook.embeddings'].clone().to(device)
    del encoder
    initialization = initialize_new_coarse_semantics(model, bundle['centers'][0].to(device), old)
    del old
    modules = install_coarse_lora(model, args.last_blocks, args.rank)
    model.eval()  # pretrained normalization/inference behavior; gradients remain enabled
    tokenizer = EsmSequenceTokenizer()
    train, validation = [], []
    groups = json.loads((args.tokens / 'manifest.json').read_text())['targets']
    for t in groups:
        with np.load(args.tokens / t['tokens']) as a:
            labels = torch.from_numpy(a['residual_tokens'][:, 0].copy()).long().to(device)[None]
        item = dict(target=t['id'], sequence=tokenize_sequence(t['sequence'], tokenizer).to(device)[None], labels=labels)
        (train if t['subset'] == 'train' else validation).append(item)
    protocol = dict(version=1, epochs=args.epochs, last_blocks=args.last_blocks, rank=args.rank,
                    token_manifest_sha256=hashlib.sha256((args.tokens / 'manifest.json').read_bytes()).hexdigest(),
                    max_train_steps=args.max_train_steps, lr=0.0001, seed=0, crop=256,
                    train_proteins=len(train), validation_proteins=len(validation),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    total_parameters=sum(p.numel() for p in model.parameters()),
                    initialized_from=str(args.checkpoint.resolve()), modules=modules,
                    selection='development mean all-masked coarse CE; initialized epoch 0 eligible',
                    scope='Real pretrained-backbone LoRA pilot; tiny development data, not full-parameter '
                          'fine-tuning, final generator, independent evaluation or successful upgrade claim.')
    path = args.output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Pretrained adaptation protocol mismatch')
    atomic_json(path, protocol)
    atomic_json(args.output / 'semantic_initialization.json', initialization)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=protocol['lr'])
    initial = evaluate(model, validation)
    atomic_json(args.output / 'initial_validation.json', initial)
    best_score, best_epoch, best = initial['mean_ce'], 0, trainable_state(model)
    started = perf_counter()
    print(json.dumps(dict(event='PRETRAINED_COARSE_LORA_STARTED', **protocol)), flush=True)
    history = []
    steps = 0
    for epoch in range(1, args.epochs + 1):
        total = 0.
        count = 0
        for index in torch.randperm(len(train)).tolist():
            item = train[index]
            length = item['labels'].shape[1]
            start = int(torch.randint(max(1, length - 256 + 1), ()))
            labels = item['labels'][:, start:start+256]
            n = labels.shape[1]
            seq = torch.cat([item['sequence'][:, :1], item['sequence'][:, 1+start:1+start+n], item['sequence'][:, -1:]], dim=1)
            t = torch.rand(1, 1, device=device).clamp(0.05, 0.999)
            sigma = model.noise(t)[0]
            mask = torch.rand_like(labels, dtype=torch.float) < (1 - torch.exp(-sigma))
            if not mask.any():
                mask[0, 0] = True
            corrupted = torch.where(mask, 32, labels)
            current = add_eos_bos_tokens(torch.tensor([n], device=device), 34, 33, 35, corrupted)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
                logits = raw_coarse_logits(model, seq, current, sigma)
                loss = F.cross_entropy(logits[mask].float(), labels[mask])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite coarse adaptation loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            steps += 1; count += 1; total += loss.item()
            if steps == 1:
                print('FIRST_PRETRAINED_BACKWARD_COMPLETE', flush=True)
            if args.max_train_steps and steps >= args.max_train_steps:
                break
        result = evaluate(model, validation)
        if result['mean_ce'] < best_score:
            best_score, best_epoch, best = result['mean_ce'], epoch, trainable_state(model)
        torch.save(dict(state=best, protocol=protocol, epoch=best_epoch, initialization=initialization),
                   args.output / 'best_adapter.pt')
        row = dict(epoch=epoch, steps=steps, train_ce=total/count, validation=result,
                   peak_gpu_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=perf_counter()-started)
        history.append(row); atomic_json(args.output / 'history.json', history)
        print(json.dumps({k:v for k,v in row.items() if k!='validation'} | {'validation_ce':result['mean_ce'], 'validation_accuracy':result['mean_accuracy']}), flush=True)
        if args.max_train_steps and steps >= args.max_train_steps:
            break
    restore_trainable_state(model, best)
    predictions = args.output / 'predictions'; predictions.mkdir(exist_ok=True)
    for item in validation:
        values = []
        for sample in range(2):
            values.append(sample_coarse(model, item['sequence'], generation_seed(item['target'], sample))[0].cpu().numpy())
        np.savez_compressed(predictions / (item['target']+'.npz'), coarse=np.stack(values))
    atomic_json(args.output / 'summary.json', dict(protocol=protocol, selected_epoch=best_epoch,
                initial_validation=initial, final_validation=evaluate(model, validation),
                elapsed_seconds=perf_counter()-started, steps=steps))
    print('PRETRAINED_COARSE_LORA_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
