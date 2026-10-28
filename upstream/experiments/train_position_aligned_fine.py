"""Three paired fine-only arms; old coarse and structural tokenizer stay frozen."""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha, save_checkpoint, restore_checkpoint
from mixed_hierarchical_io import data, tokenizer, sample_seed
from position_supported_fine import support_rows
from position_aligned_fine import PositionedFineModel, denoise_loss, supported_cross_entropy
from compare_two_level_ae_author import metrics
from structural_metrics import ensemble_summary

RUN = Path('artifacts/position_aligned_fine_v1')


def protocol():
    p = json.loads((RUN/'protocol.json').read_text())
    for path, digest in p['source_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('Frozen alignment input changed: '+path)
    return p


def build(p, seed, arm):
    saved = torch.load(p['warmstart'][str(seed)], map_location='cpu', weights_only=False)
    old = saved['model']
    counts = np.load(p['counts'][str(seed)])
    model = PositionedFineModel(old['mean'], old['std'], counts, **p['model'], arm=arm)
    missing, unexpected = model.fine.load_state_dict({k[5:]:v for k,v in old.items() if k.startswith('fine.')}, strict=False)
    expected = [] if arm == 'support_only' else ['block_position.weight']
    if arm == 'position_coarse':
        expected.append('local_coarse_gate')
    if sorted(missing) != sorted(expected) or unexpected:
        raise RuntimeError('Warm-start fine weights differ: '+str((missing, unexpected)))
    return model


def update(model, item, optimizer, p):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, terms = denoise_loss(model, item['features'], item['coarse'], item['fine'],
                              jitter_std=p['training']['jitter_std'],
                              smoothing=p['training']['label_smoothing'])
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise RuntimeError('Nonfinite position-aligned fine update')
    optimizer.step()
    return {k:float(v) for k,v in terms.items()}


@torch.no_grad()
def likelihood(model, items, ids, seed):
    model.eval()
    before = torch.get_rng_state().clone()
    rows = []
    for item in items:
        name = item['row']['id']
        if name not in ids:
            continue
        target = item['fine']
        torch.manual_seed(sample_seed(seed, name, 900))
        mask = torch.rand(len(target)) < .5
        if not mask.any():
            mask[torch.randint(len(mask), (1,))] = True
        corrupted = torch.where(mask, model.fine.fine_size, target)
        logits = model.fine(item['features'], model.normalize(item['coarse']), corrupted, .5)
        support = support_rows(model.position_counts, len(target), target.device) > 0
        missing = ~support.gather(-1, target[:,None]).flatten()
        conditional = None
        if not missing[mask].any():
            conditional = float(supported_cross_entropy(logits[mask], target[mask], support[mask])[1])
        rows.append(dict(target=name, half_masked_conditional_nll=conditional,
                         native_tokens_outside_support=int(missing.sum()),
                         masked_targets_outside_support=int(missing[mask].sum()),
                         unrestricted_half_masked_ce=float(torch.nn.functional.cross_entropy(logits[mask], target[mask]))))
    torch.set_rng_state(before)
    if len(rows) != len(ids):
        raise RuntimeError('Incomplete paired likelihood cohort')
    finite = [r['half_masked_conditional_nll'] for r in rows if r['half_masked_conditional_nll'] is not None]
    return dict(proteins=len(rows),half_masked_conditional_nll=float(np.mean(finite)) if len(finite)==len(rows) else None,
                finite_only_conditional_nll_diagnostic=float(np.mean(finite)) if finite else None,
                proteins_with_infinite_conditional_nll=len(rows)-len(finite),
                native_tokens_outside_support=sum(r['native_tokens_outside_support'] for r in rows),
                unrestricted_half_masked_ce=float(np.mean([r['unrestricted_half_masked_ce'] for r in rows])),
                targets=rows,scope='Teacher/native coarse, fixed50% masks, no jitter or smoothing; not deployment quality or full sequence likelihood. Unsupported targets are reported, never silently excluded.')


@torch.no_grad()
def generation(model, decoder, items, p, seed, directory, samples, native=False):
    model.eval()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(p['coarse_source_run'])/'fine'/f'seed{seed}'/'final_samples'
    rows = []
    for item in items:
        name = item['row']['id']
        if name not in p['generation_probe_ids']:
            continue
        records, coords = [], []
        for sample in range(samples):
            key = sample_seed(seed, name, sample)
            if native:
                coarse = item['coarse']
            else:
                with np.load(root/f'{name}_sample{sample}.npz') as z:
                    coarse = torch.from_numpy(z['coarse'].copy()).float()
            fine = model.sample(item['features'], coarse, (key+104729)%(2**32), 16)
            backbone = decoder.decode(coarse, fine).numpy()
            records.append(dict(sample=sample, seed=key, metrics=metrics(backbone, item['bb'].numpy())))
            coords.append(backbone[:,1])
            np.savez_compressed(directory/f'{name}_sample{sample}.npz',backbone=backbone,coarse=coarse.numpy(),fine=fine.numpy())
        rows.append(dict(target=name,samples=records,spread=ensemble_summary(coords)['mean_pairwise_ca_rmsd']))
    if len(rows) != len(p['generation_probe_ids']):
        raise RuntimeError('Incomplete fixed generation cohort')
    aggregate = {k:float(np.mean([np.mean([s['metrics'][k] for s in r['samples']]) for r in rows]))
                 for k in rows[0]['samples'][0]['metrics']}
    result = dict(status='complete',proteins=len(rows),samples_per_protein=samples,
                  metrics=aggregate,spread=float(np.mean([r['spread'] for r in rows])),targets=rows,
                  scope='Native coarse oracle diagnostic' if native else 'Fixed old generated coarse, unfiltered fine samples; no reference selection or geometric acceptance.')
    atomic_json(directory/'evaluation.json',result)
    return result


def state_digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def smoke(p, seed, items, decoder):
    results=[]
    frozen=state_digest(decoder)
    for arm in p['arms']:
        torch.manual_seed(seed)
        model=build(p,seed,arm)
        optimizer=torch.optim.AdamW(model.parameters(),lr=p['training']['lr'],weight_decay=p['training']['weight_decay'])
        for item in items[:4]:
            update(model,item,optimizer,p)
        path=RUN/f'smoke_{arm}.pt';rng=np.random.default_rng(seed)
        save_checkpoint(path,model,optimizer,rng)
        expected=update(model,items[0],optimizer,p)
        weights={k:v.clone() for k,v in model.state_dict().items()}
        adam=copy.deepcopy(optimizer.state_dict());torch_rng=torch.get_rng_state().clone()
        restored=build(p,seed,arm)
        opt2=torch.optim.AdamW(restored.parameters(),lr=p['training']['lr'],weight_decay=p['training']['weight_decay'])
        restore_checkpoint(path,restored,opt2,np.random.default_rng(999),torch.device('cpu'))
        actual=update(restored,items[0],opt2,p)
        if expected!=actual or not torch.equal(torch_rng,torch.get_rng_state()):
            raise RuntimeError('Real alignment resume loss/RNG differs')
        for k,v in weights.items():
            if not torch.equal(v,restored.state_dict()[k]):raise RuntimeError('Real alignment resume model differs '+k)
        for k,values in adam['state'].items():
            for name,value in values.items():
                if not torch.equal(value,opt2.state_dict()['state'][k][name]):raise RuntimeError('Real alignment Adam differs')
        results.append(dict(arm=arm,next_update_model_adam_rng_exact=True,parameters=sum(v.numel() for v in model.parameters())))
    model.eval()
    with torch.no_grad():
        fine=model.sample(items[0]['features'],items[0]['coarse'],seed,3)
        bb=decoder.decode(items[0]['coarse'],fine)
    if not torch.isfinite(bb).all() or state_digest(decoder)!=frozen or any(v.requires_grad or v.grad is not None for v in decoder.parameters()):
        raise RuntimeError('Tokenizer freeze / generation smoke failed')
    atomic_json(RUN/'smoke.json',dict(status='passed',tests=results,tokenizer_state_exact=True))
    print('POSITION_ALIGNED_SMOKE_PASSED',json.dumps(results),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['smoke','train'],required=True);parser.add_argument('--index',type=int,default=0);args=parser.parse_args()
    p=protocol();seed=p['seeds'][0] if args.stage=='smoke' else p['seeds'][args.index//len(p['arms'])]
    arm=p['arms'][args.index%len(p['arms'])]
    torch.set_num_threads(4);torch.manual_seed(seed)
    parent=json.loads(Path(p['parent_protocol']).read_text());items,stats=data(parent,seed);train=[x for x in items if x['row']['subset']=='train'];decoder=tokenizer(parent,seed)
    if args.stage=='smoke':smoke(p,seed,train,decoder);return
    if json.loads((RUN/'smoke.json').read_text())['status']!='passed':raise RuntimeError('Smoke required')
    model=build(p,seed,arm)
    if not torch.equal(model.mean,torch.tensor(stats['mean']).float()) or not torch.equal(model.std,torch.tensor(stats['std']).float()):raise RuntimeError('Normalization mismatch')
    counts=np.zeros((model.fine.fine_size,4),dtype=np.int64)
    for item in train:np.add.at(counts,(item['fine'].numpy(),np.arange(len(item['fine']))%4),1)
    if not np.array_equal(counts,model.position_counts.numpy()):raise RuntimeError('Counts differ from TRAIN labels')
    out=RUN/arm/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Completed alignment run immutable')
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['training']['lr'],weight_decay=p['training']['weight_decay'])
    torch.manual_seed(seed+80000);rng=np.random.default_rng(seed+80000);first=0;updates=0;trace=[]
    ph=sha(RUN/'protocol.json')
    if (out/'last.pt').exists():
        saved=restore_checkpoint(out/'last.pt',model,optimizer,rng,torch.device('cpu'))
        if saved['protocol_sha256']!=ph or saved['arm']!=arm:raise RuntimeError('Alignment resume protocol mismatch')
        first,updates=saved['epoch'],saved['updates'];trace=json.loads((out/'trace.json').read_text())
    started=time.perf_counter()
    for epoch in range(first,p['training']['epochs']):
        lr=p['training']['min_lr']+.5*(p['training']['lr']-p['training']['min_lr'])*(1+math.cos(math.pi*epoch/(p['training']['epochs']-1)))
        for group in optimizer.param_groups:group['lr']=lr
        records=[];order=rng.permutation(len(train));epoch_start=time.perf_counter()
        for step,index in enumerate(order):
            records.append(update(model,train[index],optimizer,p));updates+=1
            if step%150==0:atomic_json(out/'progress.json',dict(epoch=epoch+1,epochs=p['training']['epochs'],step=step+1,updates=updates,loss=records[-1]['loss'],seconds=time.perf_counter()-started))
        record=dict(epoch=epoch+1,updates=updates,lr=lr,seconds=time.perf_counter()-epoch_start,
                    terms={k:float(np.mean([r[k] for r in records])) for k in records[0]},
                    training_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),torch_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
        trace.append(record);atomic_json(out/'trace.json',trace)
        save_checkpoint(out/'last.pt',model,optimizer,rng,model_config=model.config,epoch=epoch+1,updates=updates,seed=seed,arm=arm,protocol_sha256=ph)
        if epoch+1 in p['training']['evaluation_epochs']:
            save_checkpoint(out/f'epoch{epoch+1}.pt',model,optimizer,rng,model_config=model.config,epoch=epoch+1,updates=updates,seed=seed,arm=arm,protocol_sha256=ph)
            before=torch.get_rng_state().clone()
            nll={subset:likelihood(model,items,ids,seed) for subset,ids in [('train',p['train_probe_ids']),('validation',p['validation_ids'])]}
            atomic_json(out/f'epoch{epoch+1}_likelihood.json',nll)
            generation(model,decoder,items,p,seed,out/f'epoch{epoch+1}_samples',2)
            if not torch.equal(before,torch.get_rng_state()):raise RuntimeError('Evaluation changed training RNG')
        print('POSITION_ALIGNED_EPOCH',arm,seed,json.dumps(record),flush=True)
    save_checkpoint(out/'final.pt',model,optimizer,rng,model_config=model.config,epoch=p['training']['epochs'],updates=updates,seed=seed,arm=arm,protocol_sha256=ph)
    sampled=generation(model,decoder,items,p,seed,out/'final_samples',8)
    native=generation(model,decoder,items,p,seed,out/'native_coarse_samples',2,native=True)
    audit=dict(local_coarse_gate_raw=float(model.fine.local_coarse_gate.detach()),local_coarse_gate_tanh=float(model.fine.local_coarse_gate.detach().tanh())) if arm=='position_coarse' else {}
    if hasattr(model.fine,'block_position'):audit['block_position_norm']=float(model.fine.block_position.weight.detach().norm())
    result=dict(status='complete',seed=seed,arm=arm,epochs=p['training']['epochs'],updates=updates,parameters=sum(v.numel() for v in model.parameters()),seconds=time.perf_counter()-started,
                evaluation=sampled,native_coarse_diagnostic=native,adapter_audit=audit,protocol_sha256=ph)
    atomic_json(out/'complete.json',result);print('POSITION_ALIGNED_COMPLETE',arm,seed,json.dumps(sampled['metrics']),flush=True)


if __name__=='__main__':main()
