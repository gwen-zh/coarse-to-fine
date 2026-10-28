"""Cache frozen generator features for development-only medium adaptation.

Both reference-coarse and sequence-generated-coarse inputs are cached. No
reference coordinates or reference tokens enter the sampled inference path.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from development_screen import (atomic_json, seed_all, load_state_dict_from_lightning_ckpt,
                                EsmSequenceTokenizer, tokenize_sequence)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(version=1, steps=5, seed=731, features='fine_head_penultimate_fp32',
                    checkpoint=str(args.checkpoint.resolve()), manifest=str(args.manifest.resolve()))
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Feature cache protocol mismatch')
    atomic_json(protocol_path, protocol)
    model = load_state_dict_from_lightning_ckpt(args.checkpoint, device='cuda:0')
    model.requires_grad_(False)
    head = model.net.output_heads.fine_structure_head[-1]
    torch.save(dict(weight=head.weight.detach().cpu(), bias=head.bias.detach().cpu()),
               args.output / 'fine_projection.pt')
    captured = []
    hook = head.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach()))
    tokenizer = EsmSequenceTokenizer()
    groups = json.loads(args.manifest.read_text())['all_groups']
    groups = sorted((t for t in groups if t['split'] == 'development'), key=lambda t: t['sequence_hash'])
    for target in groups[:args.limit]:
        path = args.output / f"{target['id']}.npz"
        if path.exists():
            with np.load(path) as a:
                assert a['sampled_h'].shape == (target['length'], head.in_features)
            continue
        ids = np.load(args.cache / f"{target['id']}.npz")
        n = target['length']
        seq = tokenize_sequence(target['sequence'], tokenizer).cuda()[None]
        lengths = torch.tensor([n], device='cuda:0')
        coarse = torch.cat([torch.tensor([34]), torch.from_numpy(ids['coarse']).long(),
                            torch.tensor([33])]).cuda()[None]
        captured.clear()
        model._model_wrapper_inference(coarse, sequence_tokens=seq,
                                       sigma=torch.zeros(1, 1, device='cuda:0'), lengths=lengths, scale=2)
        teacher = captured[-1][0, 1:n+1].cpu().numpy().copy()
        captured.clear()
        seed_all((protocol['seed'] + int(target['sequence_hash'][:8], 16)) % (2 ** 32))
        fine, sampled_coarse = model.ddpm_sample(seq, lengths=lengths, num_steps=5, return_coarse=True)
        sampled = captured[-1][0, 1:n+1].cpu().numpy().copy()
        # All normal residues must reproduce the released fine argmax exactly.
        reconstructed = torch.nn.functional.linear(torch.from_numpy(sampled).cuda(),
                                                    head.weight[:4096], head.bias[:4096]).argmax(-1)
        assert torch.equal(reconstructed, fine[0, 1:n+1])
        temporary = path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, teacher_h=teacher, sampled_h=sampled,
                            sampled_coarse=sampled_coarse[0].cpu().numpy(),
                            sampled_fine=fine[0].cpu().numpy(), fine=ids['fine'],
                            medium=ids['medium'], latent=ids['latent'])
        temporary.replace(path)
        print(json.dumps(dict(target=target['id'], length=n, cached=True)), flush=True)
    hook.remove()
    atomic_json(args.output / 'summary.json', dict(protocol=protocol, targets=len(groups[:args.limit]),
                note='Development only; teacher inputs are diagnostics, sampled inputs use sequence only.'))
    print('GENERATOR_FEATURES_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
