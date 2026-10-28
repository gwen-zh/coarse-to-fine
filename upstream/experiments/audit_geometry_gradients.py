"""Measure separate CE/KL/decoder gradient norms and cosines on TRAIN data only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from train_residual_condition import load_item
from residual_condition import load_tail
from fine_retention import retention_kl
from geometry_worker_client import GeometryWorker
from gradient_diagnostics import compare_gradients
from train_residual_vq import atomic_json


def main():
    p=argparse.ArgumentParser()
    for name in ['cache','training-run','weights','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(8);torch.manual_seed(0)
    args.output.mkdir(parents=True,exist_ok=True)
    source=args.training_run/'no_geometry'
    training=json.loads((source/'condition_protocol.json').read_text())
    manifest_path=args.cache/'manifest.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=training['cache_manifest_sha256']:
        raise ValueError('Training cache manifest changed')
    corpus={r['id']:r for r in json.loads(manifest_path.read_text())['targets'] if r['dataset']=='corpus'}
    ids=training['train_targets'][:16]
    if len(ids)!=16 or any(corpus[name]['subset']!='train' for name in ids):raise ValueError('Expected16TRAIN examples')
    checkpoints=dict(original=source/'snapshots/epoch0.pt',ce_kl_epoch3=source/'snapshots/epoch3.pt',
                     geometry_epoch3=args.training_run/'geometry/snapshots/epoch3.pt')
    protocol=dict(version=1,targets=ids,selection='First16 from original fixed64 TRAIN order, without selecting by geometry',
        cache_manifest_sha256=training['cache_manifest_sha256'],
        checkpoint_sha256={label:hashlib.sha256(path.read_bytes()).hexdigest() for label,path in checkpoints.items()},
        measures='Separate CE, KL(teacher||current), hard-forward biased geometry surrogate at logits and condition projection; geometry/(CE+KL) norm and cosines',
        scope='Read-only TRAIN diagnostics. No optimizer, validation, old diagnostic set, or holdout. Cosines describe the biased surrogate, not guaranteed descent of discrete geometry.')
    protocol_path=args.output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise ValueError('Audit protocol changed')
    atomic_json(protocol_path,protocol)
    items=[load_item(corpus[name],args.cache,source) for name in ids]
    model=load_tail(torch.load(args.cache/'tail.pt',map_location='cpu',weights_only=True))
    parameters=[v for v in model.parameters() if v.requires_grad]
    flatten=lambda gradients:torch.cat([g.flatten() for g in gradients])
    records=[]
    with GeometryWorker(args.weights,args.output/'worker.log',threads=8) as worker:
        for label,path in checkpoints.items():
            snapshot=torch.load(path,map_location='cpu',weights_only=True)
            state=snapshot['state']
            if set(state)!={name for name,v in model.named_parameters() if v.requires_grad}:
                raise ValueError('Checkpoint contains unexpected trainable state')
            missing=model.load_state_dict(state,strict=False)
            if missing.unexpected_keys:raise ValueError('Unexpected checkpoint keys')
            for item in items:
                output=args.output/label/(item['id']+'.json');output.parent.mkdir(exist_ok=True)
                if output.exists():records.append(json.loads(output.read_text()));continue
                logits=model(item['prefix'],item['skip'],item['condition'])[0]
                teacher=F.linear(item['teacher_h'],model.head[-1].weight[:4096],model.head[-1].bias[:4096])
                ce=F.cross_entropy(logits,item['labels']);kl=retention_kl(logits,teacher)
                result=worker.gradient(logits)
                ce_gradient=torch.autograd.grad(ce,parameters,retain_graph=True)
                kl_gradient=torch.autograd.grad(kl,parameters,retain_graph=True)
                geometry_gradient=torch.autograd.grad(logits,parameters,grad_outputs=result['gradient'])
                with torch.no_grad():
                    probability=logits.softmax(-1)
                    ce_logit=(probability-F.one_hot(item['labels'],num_classes=4096))/len(logits)
                    kl_logit=(probability-teacher.softmax(-1))/len(logits)
                row=dict(checkpoint=label,target=item['id'],length=len(logits),
                    ce=ce.item(),kl=kl.item(),cn_loss=result['cn_loss'],clash_loss=result['clash_loss'],
                    hard_metrics=result['metrics'],
                    logits=compare_gradients(ce_logit,kl_logit,result['gradient']),
                    projection=compare_gradients(flatten(ce_gradient),flatten(kl_gradient),flatten(geometry_gradient)))
                atomic_json(output,row);records.append(row)
                print('GRADIENT_AUDIT',label,item['id'],json.dumps(row['projection']),flush=True)
    if any(v.grad is not None for v in model.parameters()):raise RuntimeError('Audit should not accumulate parameter gradients')
    summaries={}
    for label in checkpoints:
        selected=[r for r in records if r['checkpoint']==label]
        summary={}
        for space in ['logits','projection']:
            summary[space]=dict(nonzero_geometry_targets=sum(r[space]['norms']['geometry']>0 for r in selected))
            for metric in ['geometry_over_base','geometry_cosine_ce','geometry_cosine_kl','geometry_cosine_base']:
                values=[r[space][metric] for r in selected if r[space][metric] is not None]
                summary[space][metric]=dict(defined=len(values),median=float(np.median(values)) if values else None,
                    quantile10_90=np.quantile(values,[.1,.9]).tolist() if values else None,
                    negative_count=sum(value<0 for value in values))
        summaries[label]=summary
    atomic_json(args.output/'summary.json',dict(protocol=protocol,checkpoints=summaries,records=records))
    print('GEOMETRY_GRADIENT_AUDIT_COMPLETE',json.dumps(summaries),flush=True)


if __name__=='__main__':main()
