"""Adapt original coarse/fine heads and shared pretrained attention LoRA.

Preserves all published token semantics and the frozen structure decoder.
Sampled training prefixes are cached from the initial frozen policy, not online
or out-of-fold predictions. This is an exploratory single-structure pilot.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from train_pretrained_coarse import (load_state_dict_from_lightning_ckpt, EsmSequenceTokenizer,
    tokenize_sequence, raw_coarse_logits, sample_coarse, add_eos_bos_tokens, generation_seed,
    trainable_state, restore_trainable_state, atomic_json)
from native_generator import install_native_lora, choose_fine_prefix, raw_fine_logits


def pad_coarse(values):
    return add_eos_bos_tokens(torch.tensor([values.shape[1]], device=values.device), 34, 33, 35, values)


@torch.no_grad()
def validate(model, items):
    rows = []
    for item in items:
        sigma = model.noise(torch.full((1, 1), .999, device='cuda'))[0]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
            coarse = raw_coarse_logits(model, item['sequence'], pad_coarse(torch.full_like(item['coarse'], 32)), sigma)
            fine = raw_fine_logits(model, item['sequence'], pad_coarse(item['sampled']))
        c = F.cross_entropy(coarse.float().flatten(0, 1), item['coarse'].flatten()).item()
        f = F.cross_entropy(fine.float().flatten(0, 1), item['fine'].flatten()).item()
        rows.append(dict(target=item['target'], coarse_ce=c, fine_ce=f,
                         normalized_ce=.5*(c/math.log(32)+f/math.log(4096))))
    return dict(score=float(np.mean([r['normalized_ce'] for r in rows])), targets=rows)


@torch.no_grad()
def save_samples(model, items, directory, protocol):
    predictions = directory / 'predictions'
    predictions.mkdir(parents=True, exist_ok=True)
    for item in items:
        for sample in range(protocol['samples']):
            coarse = sample_coarse(model, item['sequence'], generation_seed(item['target'], sample))
            fine = raw_fine_logits(model, item['sequence'], pad_coarse(coarse)).argmax(-1)
            np.savez_compressed(predictions / f"{item['target']}_sample{sample}.npz",
                                tokens=fine[0, :, None].cpu().numpy(), coarse=coarse[0].cpu().numpy())
        print('NATIVE_SAMPLED', directory.name, item['target'], flush=True)
    atomic_json(directory / 'summary.json', dict(protocol=protocol, proteins=len(items)))


def main():
    p = argparse.ArgumentParser()
    for name in ['tokens', 'diagnostic_tokens', 'checkpoint', 'output']:
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--limit', type=int, help='Smoke only: cap each data subset')
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Allocate Slurm GPU')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    args.output.mkdir(parents=True, exist_ok=True)
    model = load_state_dict_from_lightning_ckpt(args.checkpoint, device=torch.device('cuda:0'))
    modules = install_native_lora(model)
    model.eval()
    tokenizer = EsmSequenceTokenizer()

    def load_items(directory, subset):
        manifest = json.loads((directory / 'manifest.json').read_text())
        if not manifest['protocol']['semantics'].startswith('published small32'):
            raise ValueError('Native token semantics required')
        selected = [t for t in manifest['targets'] if t['subset'] == subset]
        if args.limit:
            selected = selected[:args.limit]
        result = []
        for t in selected:
            with np.load(directory / t['tokens']) as a:
                c, f = (torch.from_numpy(a[k].copy()).long().cuda()[None] for k in ['coarse', 'fine'])
            if c.shape != f.shape or c.shape[1] != len(t['sequence']) or torch.any((c<0)|(c>=32)) or torch.any((f<0)|(f>=4096)):
                raise ValueError('Invalid native training labels')
            result.append(dict(target=t['id'], coarse=c, fine=f,
                               sequence=tokenize_sequence(t['sequence'], tokenizer).cuda()[None]))
        if not result:
            raise ValueError('Empty subset ' + subset)
        return result

    train = load_items(args.tokens, 'train')
    validation = load_items(args.tokens, 'validation')
    diagnostic = load_items(args.diagnostic_tokens, 'validation')
    if {i['target'] for i in train} & {i['target'] for i in validation+diagnostic}:
        raise ValueError('Training/validation/diagnostic IDs overlap')
    protocol = dict(version=1, epochs=args.epochs, max_steps=args.max_steps, limit=args.limit,
        samples=2, token_semantics='unchanged published coarse32/fine4096', modules=modules,
        initialized_from=str(args.checkpoint.resolve()), checkpoint_bytes=args.checkpoint.stat().st_size,
        last_blocks=8, rank=8, coarse_sampling_steps=5,
        training_manifest_sha256=hashlib.sha256((args.tokens/'manifest.json').read_bytes()).hexdigest(),
        diagnostic_manifest_sha256=hashlib.sha256((args.diagnostic_tokens/'manifest.json').read_bytes()).hexdigest(),
        train_proteins=len(train), validation_proteins=len(validation), diagnostic_proteins=len(diagnostic),
        lr=1e-4, crop=256, generated_prefix_probability=.5, seed=0,
        selection='Mean normalized coarse and fine CE on NEW corpus validation; epoch zero eligible',
        conditioning='50% native / 50% cached original-policy sampled coarse for fine training',
        scope='Pilot: single structures, heuristic homology screening, no thermodynamic claim. '
              '40 previous development proteins are diagnostics, never checkpoint-selection targets. '
              '58 holdout untouched. Cached prefixes are not online or out-of-fold.')
    path = args.output/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Native generator protocol mismatch')
    atomic_json(path, protocol)
    cache = args.output/'initial_prefixes'
    cache.mkdir(exist_ok=True)
    for index, item in enumerate(train+validation):
        path = cache/(item['target']+'.npz')
        if not path.exists():
            predicted = sample_coarse(model, item['sequence'], generation_seed(item['target'], 0))
            np.savez_compressed(path, coarse=predicted[0].cpu().numpy())
        with np.load(path) as a:
            item['sampled'] = torch.from_numpy(a['coarse'].copy()).long().cuda()[None]
        if item['sampled'].shape != item['coarse'].shape:
            raise ValueError('Cached prefix length differs')
        if (index+1)%25 == 0:
            print('NATIVE_PREFIX_CACHE', index+1, flush=True)
    save_samples(model, diagnostic, args.output/'baseline', protocol)
    initial = validate(model, validation)
    best_score, best_epoch, best = initial['score'], 0, trainable_state(model)
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=protocol['lr'])
    rng = torch.Generator().manual_seed(2026)
    torch.manual_seed(0)
    steps, history = 0, []
    started = perf_counter()
    for epoch in range(1, args.epochs+1):
        for index in torch.randperm(len(train), generator=rng).tolist():
            item = train[index]
            n = item['coarse'].shape[1]
            start = int(torch.randint(max(1,n-256+1), (), generator=rng))
            end = min(n, start+256)
            seq = torch.cat([item['sequence'][:,:1],item['sequence'][:,1+start:1+end],item['sequence'][:,-1:]],1)
            native = item['coarse'][:,start:end]
            optimizer.zero_grad(set_to_none=True)
            stage = steps % 2  # two-step smoke necessarily exercises both heads
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
                if stage == 0:
                    sigma = model.noise(torch.rand(1,1,device='cuda').clamp(.05,.999))[0]
                    mask = torch.rand_like(native,dtype=torch.float) < 1-torch.exp(-sigma)
                    if not mask.any(): mask[0,0] = True
                    current = torch.where(mask,32,native)
                    logits = raw_coarse_logits(model,seq,pad_coarse(current),sigma)
                    loss = F.cross_entropy(logits[mask].float(),native[mask])
                else:
                    prefix = choose_fine_prefix(native,item['sampled'][:,start:end],.5,rng)
                    logits = raw_fine_logits(model,seq,pad_coarse(prefix))
                    loss = F.cross_entropy(logits.float().flatten(0,1),item['fine'][:,start:end].flatten())
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite native generator loss')
            loss.backward()
            if steps < 2:
                head = model.net.output_heads.coarse_structure_head if stage == 0 else model.net.output_heads.fine_structure_head
                if not any(v.grad is not None and v.grad.abs().sum()>0 for v in head.parameters()):
                    raise RuntimeError('Expected active-head gradient missing')
                print('NATIVE_BACKWARD_COMPLETE',stage,flush=True)
            torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.)
            optimizer.step()
            steps += 1
            if args.max_steps and steps >= args.max_steps: break
        result = validate(model,validation)
        if result['score'] < best_score:
            best_score,best_epoch,best = result['score'],epoch,trainable_state(model)
        row = dict(epoch=epoch,steps=steps,validation_score=result['score'],elapsed_seconds=perf_counter()-started)
        history.append(row)
        atomic_json(args.output/'history.json',history)
        torch.save(dict(state=best,protocol=protocol,epoch=best_epoch),args.output/'best_adapter.pt')
        print(json.dumps(row),flush=True)
        if args.max_steps and steps >= args.max_steps: break
    restore_trainable_state(model,best)
    save_samples(model,diagnostic,args.output/'adapted',protocol)
    atomic_json(args.output/'summary.json',dict(protocol=protocol,steps=steps,selected_epoch=best_epoch,
        initial_validation=initial,selected_score=best_score,elapsed_seconds=perf_counter()-started,
        trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        peak_gpu_bytes=torch.cuda.max_memory_allocated()))
    print('NATIVE_GENERATOR_TRAINING_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
