"""C: full899 TRAIN, pure CE versus CE+KL, fixed snapshots and no selection."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from train_residual_condition import load_item
from residual_condition import load_tail
from fine_retention import retention_kl
from coarse_lora import trainable_state
from train_residual_vq import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--index',type=int,required=True)
    args=p.parse_args();torch.set_num_threads(16);torch.manual_seed(0)
    protocol_path=args.run/'protocol.json';protocol=json.loads(protocol_path.read_text());config=protocol['C']
    if not 0<=args.index<len(config['seeds'])*len(config['arms']):raise ValueError('Invalid trajectory index')
    seed=config['seeds'][args.index//len(config['arms'])];arm=config['arms'][args.index%len(config['arms'])]
    directory=args.run/'C'/f'{arm}_seed{seed}';directory.mkdir(parents=True,exist_ok=True)
    if (directory/'progress.json').exists():raise ValueError('Existing trajectory; use a new run')
    cache=Path(protocol['tail_cache']);source=Path(protocol['mid_source'])
    if hashlib.sha256((cache/'manifest.json').read_bytes()).hexdigest()!=protocol['hashes']['tail_manifest']:
        raise ValueError('Cache changed')
    if hashlib.sha256((source/'mid.pt').read_bytes()).hexdigest()!=protocol['hashes']['mid_checkpoint']:
        raise ValueError('Fixed mid checkpoint changed')
    rows={r['id']:r for r in json.loads((cache/'manifest.json').read_text())['targets'] if r['dataset']=='corpus'}
    if any(rows[name]['subset']!='train' for name in protocol['train_targets']):raise ValueError('Nontrain optimization input')
    if any(rows[name]['subset']!='validation' for name in protocol['validation_targets']):raise ValueError('Validation mismatch')
    train=[load_item(rows[name],cache,source) for name in protocol['train_targets']]
    validation=[load_item(rows[name],cache,source) for name in protocol['validation_targets']]
    train_ce_items=[train[protocol['train_targets'].index(name)] for name in config['train_ce_targets']]
    model=load_tail(torch.load(cache/'tail.pt',map_location='cpu',weights_only=True))
    trainable={name:v for name,v in model.named_parameters() if v.requires_grad}
    if set(trainable)!={'condition.weight'} or torch.count_nonzero(model.condition.weight):
        raise ValueError('Expected original frozen tail and zero condition projection')
    parameters=list(trainable.values())
    optimizer=torch.optim.AdamW(parameters,lr=config['lr'],weight_decay=0.)
    generator=torch.Generator().manual_seed(2026+seed)
    snapshots=directory/'snapshots';snapshots.mkdir(exist_ok=True)
    predictions=directory/'predictions';predictions.mkdir(exist_ok=True)
    history=[];started=time.perf_counter();protocol_hash=hashlib.sha256(protocol_path.read_bytes()).hexdigest()

    @torch.no_grad()
    def snapshot(step):
        values=[]
        for item in validation:
            logits=model(item['prefix'],item['skip'],item['condition'])[0]
            fine=logits.argmax(-1)
            teacher=F.linear(item['teacher_h'],model.head[-1].weight[:4096],model.head[-1].bias[:4096])
            if step==0 and not np.array_equal(fine.numpy(),item['baseline_fine']):
                raise ValueError('Untrained control differs from original fine')
            row=dict(target=item['id'],ce=F.cross_entropy(logits,item['labels']).item(),
                generated_current_log_score=logits.log_softmax(-1).gather(1,fine[:,None]).mean().item(),
                generated_original_log_score=teacher.log_softmax(-1).gather(1,fine[:,None]).mean().item(),
                changed_original_tokens=int((fine.numpy()!=item['baseline_fine']).sum()))
            values.append(row)
            if item['id'] in protocol['geometry_targets']:
                np.savez_compressed(predictions/f'step{step}_{item["id"]}.npz',fine=fine.numpy(),coarse=item['coarse'])
        train_ce=float(np.mean([F.cross_entropy(model(item['prefix'],item['skip'],item['condition'])[0],item['labels']).item()
            for item in train_ce_items]))
        row=dict(step=step,epoch=step/len(train),validation_ce=float(np.mean([r['ce'] for r in values])),
            train_ce_fixed16=train_ce,validation=values,elapsed_seconds=time.perf_counter()-started)
        history.append(row)
        torch.save(dict(state=trainable_state(model),step=step,protocol_sha256=protocol_hash),snapshots/f'step{step}.pt')
        atomic_json(directory/'progress.json',dict(arm=arm,seed=seed,protocol_sha256=protocol_hash,history=history))
        print('OBJECTIVE_TRAJECTORY_SNAPSHOT',arm,seed,step,row['epoch'],row['validation_ce'],flush=True)

    snapshot(0);step=0
    for epoch in range(config['epochs']):
        for index in torch.randperm(len(train),generator=generator).tolist():
            item=train[index];optimizer.zero_grad(set_to_none=True)
            logits=model(item['prefix'],item['skip'],item['condition'])[0]
            loss=F.cross_entropy(logits,item['labels'])
            if config['beta'][arm]:
                teacher=F.linear(item['teacher_h'],model.head[-1].weight[:4096],model.head[-1].bias[:4096])
                loss=loss+config['beta'][arm]*retention_kl(logits,teacher)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite trajectory loss')
            loss.backward()
            if not all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters):raise RuntimeError('Invalid condition gradient')
            torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();step+=1
            if step in config['snapshot_steps']:snapshot(step)
            if step%256==0:print('OBJECTIVE_TRAJECTORY_STEP',arm,seed,step,'elapsed',time.perf_counter()-started,flush=True)
    if [r['step'] for r in history]!=config['snapshot_steps']:raise RuntimeError('Missing predetermined snapshots')
    if any(v.grad is not None for v in model.parameters() if not v.requires_grad):raise RuntimeError('Frozen parameter gradient')
    atomic_json(directory/'training_summary.json',dict(arm=arm,seed=seed,protocol_sha256=protocol_hash,
        steps=step,history=history,trainable_parameters=sum(v.numel() for v in parameters),
        scope='Full899 TRAIN; all predetermined snapshots retained, no selection; fixed original sampled-coarse and predicted residual mid'))
    print('OBJECTIVE_TRAJECTORY_TRAINING_COMPLETE',arm,seed,flush=True)


if __name__=='__main__':main()
