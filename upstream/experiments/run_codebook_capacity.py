"""Export, fit, smoke, train and report a fixed-latent fine codebook capacity test."""
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
from hierarchical_pipeline_io import sha,load_items,save_checkpoint,restore_checkpoint,usage
from mixed_hierarchical_io import tokenizer,sample_seed
from compare_two_level_ae_author import metrics
from structural_metrics import ensemble_summary
from codebook_capacity import nearest,refit_codes,counts_by_slot,paired_model,training_loss,decode_codes,state_hash

RUN=Path('artifacts/codebook_capacity_v1')


def protocol():
    p=json.loads((RUN/'protocol.json').read_text())
    for path,digest in p['source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Frozen capacity input changed: '+path)
    return p


def arm_path(seed,k):return RUN/f'seed{seed}'/f'k{k}'


def exported(p,seed,features=False):
    parent=json.loads(Path(p['parent_protocol']).read_text())
    items=load_items(parent,features=features)
    root=RUN/f'seed{seed}'/'export';manifest=json.loads((root/'manifest.json').read_text())
    for item in items:
        name=item['row']['id'];path=root/(name+'.npz')
        if sha(path)!=manifest['sha256'][name]:raise RuntimeError('Export changed: '+name)
        with np.load(path) as z:
            item.update(coarse=torch.from_numpy(z['coarse'].copy()),latent=torch.from_numpy(z['fine_latent'].copy()),original=torch.from_numpy(z['original_fine'].copy()))
    return items,parent


def aggregate(rows):
    return {key:float(np.mean([r['metrics'][key] for r in rows])) for key in rows[0]['metrics']}


def export(p,seed):
    parent=json.loads(Path(p['parent_protocol']).read_text());items=load_items(parent);decoder=tokenizer(parent,seed)
    frozen=state_hash(decoder);out=RUN/f'seed{seed}'/'export';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Export already frozen')
    hashes={};original_rows=[];continuous_rows=[];train=[];labels_root=Path(p['parent_labels'])/f'seed{seed}'
    started=time.perf_counter()
    for i,item in enumerate(items):
        row=item['row'];name=row['id'];n=len(item['bb'])
        with torch.no_grad():c,f,_=decoder.model.encode_latents(item['bb'])
        with np.load(labels_root/(name+'.npz')) as old:
            cached_c=torch.from_numpy(old['coarse'].copy()).float();cached_f=torch.from_numpy(old['fine'].copy()).long()
        torch.testing.assert_close(c,cached_c,atol=2e-5,rtol=2e-5)
        ids=nearest(f,decoder.model.fine_book.codes)
        if not torch.equal(ids,cached_f):raise RuntimeError('Original fine ID parity failed '+name)
        path=out/(name+'.npz');np.savez_compressed(path,coarse=cached_c.numpy(),fine_latent=f.numpy(),original_fine=cached_f.numpy());hashes[name]=sha(path)
        if row['subset']=='train':train.append(f)
        if row['subset']=='validation' or name in p['train_probe_ids']:
            with torch.no_grad():
                original=decoder.decode(cached_c,cached_f).numpy()
                continuous=decoder.model.decode(cached_c,f,n)['backbone'].numpy()
            original_rows.append(dict(target=name,subset=row['subset'],metrics=metrics(original,item['bb'].numpy())))
            continuous_rows.append(dict(target=name,subset=row['subset'],metrics=metrics(continuous,item['bb'].numpy())))
        if (i+1)%200==0:print('CAPACITY_EXPORT',seed,i+1,flush=True)
    all_train=torch.cat(train)
    rng=torch.Generator().manual_seed(seed+500003)
    pool=all_train[torch.randperm(len(all_train),generator=rng)[:min(len(all_train),p['fit']['pool_vectors'])]].clone()
    np.save(out/'training_pool.npy',pool.numpy())
    groups={}
    for subset in ['train','validation']:
        a=[r for r in original_rows if r['subset']==subset];b=[r for r in continuous_rows if r['subset']==subset]
        groups[subset]=dict(proteins=len(a),original1024=aggregate(a),continuous_fine=aggregate(b))
    if state_hash(decoder)!=frozen:raise RuntimeError('Frozen structure model changed')
    atomic_json(out/'reference.json',dict(status='complete',groups=groups,original_targets=original_rows,continuous_targets=continuous_rows))
    atomic_json(out/'manifest.json',dict(status='complete',sha256=hashes,pool_sha256=sha(out/'training_pool.npy'),
                pool_shape=list(pool.shape),train_vectors=len(all_train),original_fine_ids_exact=True,structure_sha256=frozen,
                reference_sha256=sha(out/'reference.json'),seconds=time.perf_counter()-started))


def fit(p,seed,k):
    out=arm_path(seed,k);out.mkdir(parents=True,exist_ok=True)
    if (out/'fit_complete.json').exists():raise RuntimeError('Fit already complete')
    items,parent=exported(p,seed);decoder=tokenizer(parent,seed);frozen=state_hash(decoder)
    source=RUN/f'seed{seed}'/'export';manifest=json.loads((source/'manifest.json').read_text())
    if sha(source/'training_pool.npy')!=manifest['pool_sha256']:raise RuntimeError('Training pool changed')
    pool=torch.from_numpy(np.load(source/'training_pool.npy'));started=time.perf_counter()
    codes,trace=refit_codes(pool,k,seed+700001,p['fit']['iterations'],p['fit']['chunk'])
    fit_seconds=time.perf_counter()-started
    torch.save(dict(codes=codes,seed=seed,k=k,trace=trace,fit_seconds=fit_seconds),out/'codes.pt')
    label_root=out/'labels';label_root.mkdir(exist_ok=True);hashes={};train_ids=[];sums={x:[] for x in ['train','validation']};rows=[]
    for item in items:
        row=item['row'];name=row['id'];ids=nearest(item['latent'],codes);path=label_root/(name+'.npy');np.save(path,ids.numpy());hashes[name]=sha(path)
        q=(item['latent']-codes[ids]).square().mean();sums[row['subset']].append(float(q))
        if row['subset']=='train':train_ids.append(ids)
        if row['subset']=='validation' or name in p['train_probe_ids']:
            with torch.no_grad():bb=decode_codes(decoder,item['coarse'],ids,codes).numpy()
            rows.append(dict(target=name,subset=row['subset'],metrics=metrics(bb,item['bb'].numpy()),latent_mse=float(q)))
    counts=counts_by_slot(train_ids,k);np.save(out/'train_slot_counts.npy',counts.numpy());allcounts=counts.sum(-1)
    groups={}
    for subset in ['train','validation']:
        selected=[r for r in rows if r['subset']==subset]
        groups[subset]=dict(proteins=len(selected),metrics=aggregate(selected),latent_mse_all_proteins=float(np.mean(sums[subset])))
    if state_hash(decoder)!=frozen:raise RuntimeError('Structure model changed during fit')
    atomic_json(out/'labels_manifest.json',dict(sha256=hashes,codes_sha256=sha(out/'codes.pt'),counts_sha256=sha(out/'train_slot_counts.npy')))
    atomic_json(out/'fit_complete.json',dict(status='complete',seed=seed,k=k,groups=groups,targets=rows,
                train_usage=usage(allcounts.numpy()),empty_training_codes=int((allcounts==0).sum()),fit_seconds=fit_seconds,trace=trace,
                structure_unchanged=True,information_rate_fine_bits=math.log2(k),
                scope='Refitted train-only codebook on frozen K1024-learned fine latents; current continuous coarse and structure decoder fixed. Reconstruction,not generation or independent confirmation.'))
    print('CAPACITY_FIT_COMPLETE',seed,k,fit_seconds,groups['validation'],flush=True)


def training_data(p,seed,k):
    items,parent=exported(p,seed,features=True);out=arm_path(seed,k);manifest=json.loads((out/'labels_manifest.json').read_text())
    if sha(out/'codes.pt')!=manifest['codes_sha256'] or sha(out/'train_slot_counts.npy')!=manifest['counts_sha256']:raise RuntimeError('Fitted book changed')
    for item in items:
        name=item['row']['id'];path=out/'labels'/(name+'.npy')
        if sha(path)!=manifest['sha256'][name]:raise RuntimeError('Capacity label changed')
        item['fine']=torch.from_numpy(np.load(path)).long()
    stats=json.loads((Path(p['parent_labels'])/f'seed{seed}'/'stats.json').read_text())
    codes=torch.load(out/'codes.pt',map_location='cpu',weights_only=False)['codes']
    return items,parent,stats,codes


def make_model(p,seed,k,stats):return paired_model(stats['mean'],stats['std'],k,seed,maximum_k=max(p['sizes']),**p['fine_model'])


def update(model,item,opt,p):
    model.train();opt.zero_grad(set_to_none=True)
    loss,terms=training_loss(model,item['features'],item['coarse'],item['fine'],p['training']['jitter_std'],p['training']['label_smoothing'])
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(loss) or not torch.isfinite(norm):raise RuntimeError('Nonfinite capacity fine update')
    opt.step();return dict(loss=float(loss.detach()),**terms)


@torch.no_grad()
def likelihood(model,items,ids,seed):
    model.eval();saved=torch.get_rng_state();rows=[]
    for item in items:
        name=item['row']['id']
        if name not in ids:continue
        torch.manual_seed(sample_seed(seed,name,900))
        target=item['fine'];mask=torch.rand(len(target))<.5
        if not mask.any():mask[0]=True
        partial=torch.where(mask,model.fine.fine_size,target)
        logits=model.fine(item['features'],model.normalize(item['coarse']),partial,.5)[mask]
        nll=torch.nn.functional.cross_entropy(logits,target[mask])
        accuracy=(logits.argmax(-1)==target[mask]).float().mean()
        rows.append(dict(target=name,nll=float(nll),accuracy=float(accuracy)))
    torch.set_rng_state(saved)
    if len(rows)!=len(ids):raise RuntimeError('Likelihood cohort incomplete')
    return dict(proteins=len(rows),nll=float(np.mean([r['nll'] for r in rows])),accuracy=float(np.mean([r['accuracy'] for r in rows])),targets=rows,
                scope='Native-coarse50%masked NLL. Target vocabularies differ across K;raw CE is not directly a cross-K quality score.')


@torch.no_grad()
def generation(model,decoder,codes,items,p,seed,out,native=False,samples=8):
    model.eval();out.mkdir(parents=True,exist_ok=True);rows=[]
    source=Path(p['coarse_source_run'])/'fine'/f'seed{seed}'/'final_samples'
    counts=counts_by_slot([item['fine'] for item in items if item['row']['subset']=='train'],len(codes))
    started=time.perf_counter()
    for item in items:
        name=item['row']['id']
        if name not in p['generation_probe_ids']:continue
        records=[];ca=[]
        for j in range(samples):
            key=sample_seed(seed,name,j)
            if native:coarse=item['coarse']
            else:
                cache=source/f'{name}_sample{j}.npz'
                if sha(cache)!=p['coarse_cache_sha256'][str(cache)]:raise RuntimeError('Frozen generated coarse cache changed')
                with np.load(cache) as z:coarse=torch.from_numpy(z['coarse'].copy())
            fine=model.sample(item['features'],coarse,(key+104729)%(2**32),p['sampling']['steps'])
            bb=decode_codes(decoder,coarse,fine,codes).numpy();ca.append(bb[:,1])
            missing=counts[fine,torch.arange(len(fine))%4]==0
            records.append(dict(sample=j,seed=key,metrics=metrics(bb,item['bb'].numpy()),train_unseen_slot_fraction=float(missing.float().mean())))
            np.savez_compressed(out/f'{name}_sample{j}.npz',backbone=bb,coarse=coarse.numpy(),fine=fine.numpy())
        rows.append(dict(target=name,samples=records,spread=ensemble_summary(ca)['mean_pairwise_ca_rmsd']))
    if len(rows)!=len(p['generation_probe_ids']):raise RuntimeError('Generation cohort incomplete')
    result=dict(status='complete',proteins=len(rows),samples_per_protein=samples,
        metrics={key:float(np.mean([np.mean([r['metrics'][key] for r in row['samples']]) for row in rows])) for key in rows[0]['samples'][0]['metrics']},
        spread=float(np.mean([r['spread'] for r in rows])),train_unseen_slot_fraction=float(np.mean([np.mean([r['train_unseen_slot_fraction'] for r in row['samples']]) for row in rows])),
        targets=rows,seconds=time.perf_counter()-started,
        scope='Native-coarse oracle diagnosis,not deployable.' if native else 'Fixed old sequence-generated coarse,AA-only fine generation,unfiltered outputs,no reference acceptance,reused development.')
    atomic_json(out/'evaluation.json',result);return result


def smoke(p):
    seed=p['seeds'][0];results=[]
    for k in p['sizes']:
        items,parent,stats,codes=training_data(p,seed,k);train=[x for x in items if x['row']['subset']=='train'];decoder=tokenizer(parent,seed);frozen=state_hash(decoder)
        torch.manual_seed(seed);model=make_model(p,seed,k,stats);opt=torch.optim.AdamW(model.parameters(),lr=p['training']['lr'],weight_decay=p['training']['weight_decay'])
        for item in train[:3]:update(model,item,opt,p)
        path=RUN/f'smoke_k{k}.pt';rng=np.random.default_rng(seed);save_checkpoint(path,model,opt,rng)
        expected=update(model,train[0],opt,p);expected_model=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict());expected_rng=torch.get_rng_state().clone()
        restored=make_model(p,seed,k,stats);opt2=torch.optim.AdamW(restored.parameters(),lr=p['training']['lr'],weight_decay=p['training']['weight_decay'])
        restore_checkpoint(path,restored,opt2,np.random.default_rng(99),torch.device('cpu'));actual=update(restored,train[0],opt2,p)
        if expected!=actual or not torch.equal(expected_rng,torch.get_rng_state()):raise RuntimeError('Resume loss/RNG differs')
        for name,v in expected_model.items():
            if not torch.equal(v,restored.state_dict()[name]):raise RuntimeError('Resume parameter differs '+name)
        for key,values in expected_opt['state'].items():
            for name,v in values.items():
                if not torch.equal(v,opt2.state_dict()['state'][key][name]):raise RuntimeError('Resume Adam differs')
        with torch.no_grad():
            restored.eval();f=restored.sample(train[0]['features'],train[0]['coarse'],seed,3);bb=decode_codes(decoder,train[0]['coarse'],f,codes)
        if not torch.isfinite(bb).all() or state_hash(decoder)!=frozen:raise RuntimeError('Decode/freeze smoke failed')
        results.append(dict(k=k,next_update_model_adam_rng_exact=True,parameters=sum(v.numel() for v in model.parameters())))
    atomic_json(RUN/'smoke.json',dict(status='passed',results=results));print('CAPACITY_SMOKE_PASSED',results,flush=True)


def train(p,seed,k):
    if json.loads((RUN/'smoke.json').read_text())['status']!='passed':raise RuntimeError('Smoke required')
    out=arm_path(seed,k)/'fine';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Fine training already complete')
    items,parent,stats,codes=training_data(p,seed,k);train_items=[x for x in items if x['row']['subset']=='train'];decoder=tokenizer(parent,seed);frozen=state_hash(decoder)
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=make_model(p,seed,k,stats)
    opt=torch.optim.AdamW(model.parameters(),lr=p['training']['lr'],weight_decay=p['training']['weight_decay'])
    first=0;updates=0;trace=[];protocol_sha=sha(RUN/'protocol.json');book_sha=sha(arm_path(seed,k)/'codes.pt');started=time.perf_counter()
    if (out/'last.pt').exists():
        state=restore_checkpoint(out/'last.pt',model,opt,rng,torch.device('cpu'))
        if state['protocol_sha256']!=protocol_sha or state['book_sha256']!=book_sha:raise RuntimeError('Resume protocol/book changed')
        first=state['epoch'];updates=state['updates'];trace=json.loads((out/'trace.json').read_text())
    for epoch in range(first,p['training']['epochs']):
        t0=time.perf_counter();order=rng.permutation(len(train_items));records=[];mask_hash=hashlib.sha256()
        lr=p['training']['min_lr']+.5*(p['training']['lr']-p['training']['min_lr'])*(1+math.cos(math.pi*epoch/max(1,p['training']['epochs']-1)))
        for group in opt.param_groups:group['lr']=lr
        for step,index in enumerate(order):
            # Stateless per-update seeds pair jitter, masks and dropout across K.
            torch.manual_seed(sample_seed(seed,train_items[index]['row']['id'],10000+epoch))
            record=update(model,train_items[index],opt,p);mask_hash.update(record['mask_sha256'].encode());records.append(record);updates+=1
            if (step+1)%200==0:atomic_json(out/'progress.json',dict(epoch=epoch+1,epochs=p['training']['epochs'],step=step+1,steps_per_epoch=len(order),updates=updates,loss=record['loss'],elapsed_seconds=time.perf_counter()-started))
        row=dict(epoch=epoch+1,updates=updates,seconds=time.perf_counter()-t0,lr=lr,loss=float(np.mean([r['loss'] for r in records])),
                 nll=float(np.mean([r['nll'] for r in records])),accuracy=float(np.mean([r['accuracy'] for r in records])),
                 order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),mask_sha256=mask_hash.hexdigest())
        trace.append(row);atomic_json(out/'trace.json',trace)
        save_checkpoint(out/'last.pt',model,opt,rng,epoch=epoch+1,updates=updates,seed=seed,k=k,model_config=model.config,protocol_sha256=protocol_sha,book_sha256=book_sha)
        print('CAPACITY_FINE_EPOCH',seed,k,json.dumps(row),flush=True)
        if epoch+1 in p['training']['evaluation_epochs']:
            atomic_json(out/f'epoch{epoch+1}_likelihood.json',dict(train=likelihood(model,items,p['train_probe_ids'],seed),validation=likelihood(model,items,p['validation_ids'],seed)))
    deployed=generation(model,decoder,codes,items,p,seed,out/'generated_coarse',samples=p['sampling']['samples'])
    native=generation(model,decoder,codes,items,p,seed,out/'native_coarse',native=True,samples=p['sampling']['native_samples'])
    if state_hash(decoder)!=frozen or any(v.requires_grad or v.grad is not None for v in decoder.parameters()):raise RuntimeError('Structure model was updated')
    save_checkpoint(out/'final.pt',model,opt,rng,epoch=p['training']['epochs'],updates=updates,seed=seed,k=k,model_config=model.config,protocol_sha256=protocol_sha,book_sha256=book_sha)
    atomic_json(out/'complete.json',dict(status='complete',seed=seed,k=k,updates=updates,parameters=sum(v.numel() for v in model.parameters()),seconds=time.perf_counter()-started,
                generated_metrics=deployed['metrics'],native_metrics=native['metrics'],structure_unchanged=True))
    print('CAPACITY_FINE_COMPLETE',seed,k,flush=True)


def report(p):
    rows=[];paired=True
    for seed in p['seeds']:
        reference=json.loads((RUN/f'seed{seed}'/'export'/'reference.json').read_text())
        signatures=[]
        for k in p['sizes']:
            out=arm_path(seed,k);fit_result=json.loads((out/'fit_complete.json').read_text());fine=json.loads((out/'fine'/'complete.json').read_text())
            trace=json.loads((out/'fine'/'trace.json').read_text());signatures.append([(r['order_sha256'],r['mask_sha256']) for r in trace])
            generated=json.loads((out/'fine'/'generated_coarse'/'evaluation.json').read_text());native=json.loads((out/'fine'/'native_coarse'/'evaluation.json').read_text())
            rows.append(dict(seed=seed,k=k,reconstruction=fit_result['groups']['validation'],usage=fit_result['train_usage'],fine=fine,
                             deployed_spread=generated['spread'],deployed_unseen_slot_fraction=generated['train_unseen_slot_fraction'],
                             original1024=reference['groups']['validation']['original1024'],continuous_fine=reference['groups']['validation']['continuous_fine']))
        paired=paired and all(x==signatures[0] for x in signatures)
    if not paired:raise RuntimeError('Data-order/mask randomness not paired across K')
    atomic_json(RUN/'comparison.json',dict(status='complete',rows=rows,paired_orders_masks=True,scope=p['scope']))
    lines=['Fixed-latent fine-codebook capacity: final development results','='*65,'',p['scope'],'',
           'seed / K / reconstruction CA RMSD / deployed CA RMSD / deployed CN% / clashes/residue / spread / training codes used','']
    for row in rows:
        m=row['fine']['generated_metrics'];lines.append(f"{row['seed']} / {row['k']} / {row['reconstruction']['metrics']['ca_rmsd']:.5f} / {m['ca_rmsd']:.5f} / {100*m['cn_rate']:.3f} / {m['ca_clashes_per_residue']:.5f} / {row['deployed_spread']:.5f} / {row['usage']['used']}")
    lines+=['','All K values are fresh refits from the identical train-only pool; original trained1024 is a separate reference. Encoder/decoder/coarse caches are fixed. Fixed final12epoch fine models; no bestepoch selection.','A larger vocabulary changes fine information rate,head/embedding parameter count and actual compute. Equal epochs/data/architecture do not mean equal FLOPs. Raw CE is not directly comparable across different target vocabularies.','This is conditional on latents previously learned with K1024,not full end-to-end K-specific tokenizer retraining. Development reused,no independent confirmation,author-matched superiority or equilibrium-ensemble claim.','No automatic weight replacement or extra epochs. Both seeds and all K are reported.']
    (RUN/'results_en.rst').write_text('\n'.join(lines)+'\n')
    atomic_json(RUN/'decision.json',dict(status='complete_capacity_comparison_requires_joint_quality_review',adopt_new_weights=False,paired_orders_masks=True,
               next='Review reconstruction and generated geometry/accuracy/spread together across both seeds before selecting any further experiment;no automatic deployment or new training.'))
    print('CAPACITY_REPORT_COMPLETE',rows,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['export','fit','smoke','train','report'],required=True);parser.add_argument('--index',type=int,default=0);args=parser.parse_args()
    p=protocol();torch.set_num_threads(p['threads'])
    if args.stage=='smoke':smoke(p);return
    if args.stage=='report':report(p);return
    if args.stage=='export':seed=p['seeds'][args.index];torch.manual_seed(seed);export(p,seed);return
    seed=p['seeds'][args.index//len(p['sizes'])];k=p['sizes'][args.index%len(p['sizes'])];torch.manual_seed(seed)
    if args.stage=='fit':fit(p,seed,k)
    else:train(p,seed,k)

if __name__=='__main__':main()
