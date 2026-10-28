"""Fixed-budget feasibility of a jointly trained, ESM-independent two-level AE."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from two_level_autoencoder import TwoLevelAutoencoder,training_loss
from hierarchical_latent_contract import pool_ca,remove_block_translation,compose_levels
from structural_metrics import aligned_rmsd,summarize_backbone
from backbone_quality_audit import measurements
from diagnostic_io import atomic_json


def quality(output,reference):
    x=output.detach().cpu().numpy();y=reference.detach().cpu().numpy()
    if not np.isfinite(x).all():raise RuntimeError('Nonfinite output; do not exclude')
    metrics=summarize_backbone(x,[y[:,1]])
    try:angles=measurements(x,y);error=None
    except ValueError as e:angles=None;error=str(e)
    return dict(rmsd=aligned_rmsd(x[:,1],y[:,1]),cn_rate=metrics['backbone_cn_outliers']/(len(x)-1),clashes_per_residue=metrics['upstream_ca_clashes']/len(x),angles=angles,quality_error=error)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['smoke','train'],required=True);parser.add_argument('--index',type=int,default=0);a=parser.parse_args()
    run=Path('artifacts/development/two_level_autoencoder_v1');pp=run/'protocol.json';p=json.loads(pp.read_text());ph=hashlib.sha256(pp.read_bytes()).hexdigest()
    for source,digest in p['sha256'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen source changed '+source)
    if a.stage=='train' and json.loads((run/'smoke.json').read_text())['status']!='passed':raise RuntimeError('Smoke incomplete')
    seed=p['seeds'][a.index];torch.set_num_threads(p['threads']);torch.manual_seed(seed);rng=np.random.default_rng(seed);model=TwoLevelAutoencoder();optimizer=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=1e-4)
    out=run/('smoke' if a.stage=='smoke' else f'seed{seed}');out.mkdir(exist_ok=True)
    if a.stage=='train' and (out/'training.json').exists():raise RuntimeError('Existing training; explicit resume required')
    items=[]
    for row in p['targets']:
        with np.load(row['reference']) as x:bb=torch.tensor(x['backbone'],dtype=torch.float32)
        if bb.shape!=(row['length'],3,3) or not torch.isfinite(bb).all():raise RuntimeError('Reference mismatch')
        items.append(dict(row=row,bb=bb))
    train=[x for x in items if x['row']['subset']=='train'];started=time.perf_counter();trace=[];gradients=None
    @torch.no_grad()
    def evaluate(step):
        model.eval();rows=[];directory=out/f'step{step}';directory.mkdir(exist_ok=True)
        for item in items:
            row=item['row'];x=item['bb'];n=len(x);target=x-x[:,1].mean(0);centers=pool_ca(target);residual=remove_block_translation(target)
            z_c,z_f=model.encode(x);full=model.decode(z_c,z_f,n);zero_f=model.decode(z_c,torch.zeros_like(z_f),n);zero_c=model.decode(torch.zeros_like(z_c),z_f,n)
            centroid_delta=float((pool_ca(full['backbone'])-pool_ca(zero_f['backbone'])).abs().max())
            if centroid_delta>2e-5:raise RuntimeError('Fine altered coarse centers')
            direct_coarse=float((full['coarse']-centers).square().sum(-1).mean().sqrt())
            coarse_zero=float((zero_c['coarse']-centers).square().sum(-1).mean().sqrt())
            fine_error=float((full['fine_residual']-residual).square().mean());fine_zero=float((zero_f['fine_residual']-residual).square().mean())
            variants=dict(full=full['backbone'],zero_fine=zero_f['backbone'],zero_coarse=zero_c['backbone'],oracle_coarse=model.decode_with_anchors(centers,z_f,n)['backbone'],oracle_fine=compose_levels(full['coarse'],residual))
            metrics={name:quality(value,x) for name,value in variants.items()}
            # Fixed unseen orientation diagnostic; not an exactly equivariant network.
            rot=x.new_tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
            metrics['rotated_input']=quality(model(x@rot)['backbone'],x@rot)
            rows.append(dict(target=row['id'],subset=row['subset'],length=n,coarse_shape=list(z_c.shape),fine_shape=list(z_f.shape),latent_scalars=z_c.numel()+z_f.numel(),input_scalars=x.numel(),coarse_rmse=direct_coarse,zero_coarse_rmse=coarse_zero,fine_mse=fine_error,zero_fine_mse=fine_zero,centroid_invariance_max_error=centroid_delta,metrics=metrics))
            np.savez_compressed(directory/(row['id']+'.npz'),z_c=z_c.numpy(),z_f=z_f.numpy(),coarse=full['coarse'].numpy(),**{k:v.numpy() for k,v in variants.items()})
        atomic_json(directory/'evaluation.json',dict(status='complete',step=step,targets=rows,scope=p['scope']))
        return rows
    if a.stage=='train':evaluate(0)
    steps=p['smoke_steps'] if a.stage=='smoke' else p['steps'];order=[]
    for step in range(steps):
        if step%len(train)==0:order=rng.permutation(len(train)).tolist()
        item=train[order[step%len(train)]];x=item['bb']
        q,_=torch.linalg.qr(torch.randn(3,3));q[:,0]*=torch.linalg.det(q);x=x@q
        model.train();optimizer.zero_grad(set_to_none=True);output=model(x);loss,terms=training_loss(output,x,p['weights'])
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm) or norm<=0:raise RuntimeError('Invalid gradients')
        if step==0:
            gradients={name:float(sum(v.grad.square().sum() for v in getattr(model,name).parameters() if v.grad is not None).sqrt()) for name in ['coarse_encoder','coarse_decoder','fine_encoder','fine_decoder']}
            if not all(np.isfinite(v) and v>0 for v in gradients.values()):raise RuntimeError('A learned module is disconnected')
        optimizer.step()
        if (step+1)%100==0 or step==0:
            record=dict(step=step+1,loss=float(loss),terms={k:float(v) for k,v in terms.items()},elapsed_seconds=time.perf_counter()-started);trace.append(record);atomic_json(out/'progress.json',record);print('TWO_LEVEL_AE_TRAIN',seed,json.dumps(record),flush=True)
    model.eval();state={k:v.detach().clone() for k,v in model.state_dict().items()};torch.save(dict(model=state,seed=seed,steps=steps,protocol_sha256=ph),out/'final.pt')
    # Check real-data encoder->latent-only decoder and serialized checkpoint replay.
    with torch.no_grad():
        x=train[0]['bb'];original=model(x)['backbone'];c,f=model.encode(x);decoded=model.decode(c,f,len(x))['backbone'];torch.testing.assert_close(original,decoded,atol=0,rtol=0)
        reloaded=TwoLevelAutoencoder().eval();reloaded.load_state_dict(torch.load(out/'final.pt',weights_only=False)['model']);torch.testing.assert_close(original,reloaded(x)['backbone'],atol=0,rtol=0)
    training=dict(status='complete',seed=seed,steps=steps,seconds=time.perf_counter()-started,parameters=sum(v.numel() for v in model.parameters()),device='cpu',threads=p['threads'],module_gradient_norms=gradients,trace=trace)
    atomic_json(out/'training.json',training)
    if a.stage=='smoke':
        atomic_json(run/'smoke.json',dict(status='passed',module_gradient_norms=gradients,latent_only_decode_exact=True,checkpoint_replay_exact=True,steps=steps,parameters=training['parameters'],seconds=training['seconds']));print('TWO_LEVEL_AE_SMOKE_PASSED',flush=True)
    else:
        evaluate(steps);atomic_json(out/'complete.json',dict(status='complete',seed=seed,scope=p['scope']));print('TWO_LEVEL_AE_COMPLETE',seed,flush=True)

if __name__=='__main__':main()
