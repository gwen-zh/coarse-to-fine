"""Matched small training ablation: original residual condition +/- decoder geometry."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from train_residual_condition import load_item,validate,save_predictions
from residual_condition import load_tail
from fine_retention import retention_kl
from coarse_lora import trainable_state
from geometry_worker_client import GeometryWorker,backward_geometry_objective
from train_residual_vq import atomic_json


def main():
    p=argparse.ArgumentParser()
    for name in ['cache','mid-source','run','weights']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=['geometry','no_geometry'],required=True)
    p.add_argument('--geometry-weight',type=float,default=1.)
    p.add_argument('--weight-plan',type=Path)
    args=p.parse_args();torch.set_num_threads(8);torch.manual_seed(0)
    if not np.isfinite(args.geometry_weight) or args.geometry_weight<0:raise ValueError('Invalid geometry weight')
    directory=args.run/args.mode;directory.mkdir(parents=True,exist_ok=True)
    manifest_path=args.cache/'manifest.json';manifest=json.loads(manifest_path.read_text())
    mid_path=args.mid_source/'mid.pt';mid=torch.load(mid_path,map_location='cpu',weights_only=True)
    cache_hash=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if mid['protocol']['cache_manifest_sha256']!=cache_hash:raise ValueError('Mid/cache mismatch')
    corpus=[r for r in manifest['targets'] if r['dataset']=='corpus']
    order=lambda r:hashlib.sha256(r['id'].encode()).hexdigest()
    train_rows=sorted([r for r in corpus if r['subset']=='train'],key=order)[:64]
    val_rows=[r for r in corpus if r['subset']=='validation']
    geometry=sorted(val_rows,key=order)[:24]
    diagnostics=[r for r in manifest['targets'] if r['dataset']=='diagnostic']
    if len(train_rows)!=64 or len(val_rows)!=101 or len(geometry)!=24:raise ValueError('Unexpected pilot split')
    protocol=dict(version=1,mode=args.mode,epochs=3,snapshot_epochs=[0,1,3],lr=1e-4,beta=1.,
        geometry_weight=args.geometry_weight if args.mode=='geometry' else 0.,seed=0,samples=2,
        cache_manifest_sha256=cache_hash,mid_checkpoint_sha256=hashlib.sha256(mid_path.read_bytes()).hexdigest(),
        train_targets=[r['id'] for r in train_rows],train_proteins=64,validation_proteins=101,
        geometry_targets=[r['id'] for r in geometry],
        injection='Same residual mid checkpoint, zero 128->1536 condition projection, original fine weights frozen',
        loss='fine CE + teacher KL + lambda*(hard-forward CN[1.1,1.7] squared hinge + CA<3 squared hinge)',
        estimator='Biased straight-through softmax gradient through actual frozen decoder, separate tokenizer process',
        matched_compute='Both arms compute the same full decoder gradient each update; control discards it',
        scope='64 fixed-order new TRAIN proteins, 3 epochs, reused mid fitted on899train; small feasibility training pilot, not independent benchmark; no holdout use')
    if args.weight_plan is not None:
        plan=json.loads(args.weight_plan.read_text())
        if args.mode!='geometry' or args.geometry_weight not in [r['weight'] for r in plan['candidates']]:
            raise ValueError('Weight is not in fixed training-calibrated plan')
        if not set(plan['calibration_targets']).issubset(protocol['train_targets']):raise ValueError('Calibration must use current TRAIN only')
        if hashlib.sha256(Path(plan['audit_summary']).read_bytes()).hexdigest()!=plan['audit_summary_sha256']:
            raise ValueError('Calibration audit changed')
        protocol['weight_calibration']=dict(plan_sha256=hashlib.sha256(args.weight_plan.read_bytes()).hexdigest(),
            source='Frozen original projection gradient norms on16 unfiltered TRAIN cases only',
            baseline_geometry_over_base_median=plan['baseline_geometry_over_base_median'],
            formula=plan['formula'])
    protocol_path=directory/'condition_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise ValueError('Protocol changed')
    if (directory/'condition_history.json').exists():raise ValueError('Existing training output: use a new run directory')
    atomic_json(protocol_path,protocol)
    # Reuse existing prediction writer with an explicit link to the fixed mid cache.
    link=directory/'mid_predictions'
    if not link.exists():link.symlink_to((args.mid_source/'mid_predictions').resolve(),target_is_directory=True)
    elif link.resolve()!=(args.mid_source/'mid_predictions').resolve():raise ValueError('Mid cache link differs')
    model=load_tail(torch.load(args.cache/'tail.pt',map_location='cpu',weights_only=True))
    parameters=[v for v in model.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=protocol['lr'],weight_decay=0.)
    train=[load_item(r,args.cache,directory) for r in train_rows]
    validation=[load_item(r,args.cache,directory) for r in val_rows]
    snapshots=directory/'snapshots';snapshots.mkdir(exist_ok=True)

    def snapshot(epoch,score):
        torch.save(dict(state=trainable_state(model),epoch=epoch,validation_ce=score,protocol=protocol),snapshots/f'epoch{epoch}.pt')
        save_predictions(model,geometry+diagnostics,args.cache,directory,epoch,protocol)

    initial=validate(model,validation);snapshot(0,initial)
    history=[dict(epoch=0,validation_ce=initial)]
    atomic_json(directory/'condition_history.json',history)
    rng=torch.Generator().manual_seed(2026)
    verified_geometry_norm=None
    with GeometryWorker(args.weights,directory/'worker.log',threads=8) as worker:
        for epoch in range(1,4):
            losses=[];gradients=[];cn_losses=[];clash_losses=[]
            for step,index in enumerate(torch.randperm(len(train),generator=rng).tolist()):
                item=train[index];optimizer.zero_grad(set_to_none=True)
                logits=model(item['prefix'],item['skip'],item['condition'])[0]
                teacher=F.linear(item['teacher_h'],model.head[-1].weight[:4096],model.head[-1].bias[:4096])
                ce=F.cross_entropy(logits,item['labels']);kl=retention_kl(logits,teacher)
                result=worker.gradient(logits)
                base=ce+kl
                if not torch.isfinite(base):raise RuntimeError('Nonfinite CE/KL')
                # Chain worker dL_geometry/dlogits through the original fine tail into the condition projection.
                backward_geometry_objective(base,logits,result['gradient'],protocol['geometry_weight'])
                if not all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters):raise RuntimeError('Missing/nonfinite adapter gradients')
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
                if verified_geometry_norm is None and result['gradient'].norm()>0:
                    # Test geometry's own route to trainable parameters, including in the matched control arm.
                    optimizer.zero_grad(set_to_none=True)
                    replay=model(item['prefix'],item['skip'],item['condition'])[0]
                    replay.backward(result['gradient'])
                    geometry_norm=float(torch.sqrt(sum(v.grad.square().sum() for v in parameters)))
                    if not np.isfinite(geometry_norm) or geometry_norm<=0:raise RuntimeError('No geometry gradient reaches condition projection')
                    verified_geometry_norm=geometry_norm
                    print('GEOMETRY_TO_CONDITION_GRADIENT_VERIFIED',geometry_norm,flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    replay=model(item['prefix'],item['skip'],item['condition'])[0]
                    replay_base=F.cross_entropy(replay,item['labels'])+retention_kl(replay,teacher)
                    backward_geometry_objective(replay_base,replay,result['gradient'],protocol['geometry_weight'])
                    norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
                optimizer.step()
                losses.append(base.item());gradients.append(float(norm));cn_losses.append(result['cn_loss']);clash_losses.append(result['clash_loss'])
                if step%16==0:print('GEOMETRY_CONDITION_STEP',args.mode,epoch,step,base.item(),result['cn_loss'],flush=True)
            score=validate(model,validation)
            if epoch in protocol['snapshot_epochs']:snapshot(epoch,score)
            row=dict(epoch=epoch,validation_ce=score,mean_ce_kl=float(np.mean(losses)),mean_cn_loss=float(np.mean(cn_losses)),
                mean_clash_loss=float(np.mean(clash_losses)),mean_gradient_norm=float(np.mean(gradients)))
            history.append(row);atomic_json(directory/'condition_history.json',history)
            print('GEOMETRY_CONDITION_EPOCH',args.mode,json.dumps(row),flush=True)
    if verified_geometry_norm is None:raise RuntimeError('No nonzero geometry-to-adapter gradient verified')
    if any(v.grad is not None for v in model.parameters() if not v.requires_grad):raise RuntimeError('Original model received parameter gradients')
    atomic_json(directory/'training_summary.json',dict(protocol=protocol,history=history,trainable_parameters=sum(v.numel() for v in parameters),verified_geometry_gradient_norm=verified_geometry_norm))
    print('GEOMETRY_CONDITION_TRAINING_COMPLETE',args.mode,flush=True)


if __name__=='__main__':main()
