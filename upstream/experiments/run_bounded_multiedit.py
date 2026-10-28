"""Five proposal controls, identical decoder-call budgets, reference-free selection."""
import argparse,hashlib,json,socket,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from fine_search_constraints import evaluate_candidate
from structural_metrics import summarize_backbone
from backbone_quality_audit import measurements
from bestofn_proposals import local_proposals,apply_proposal
from bounded_multiedit_proposals import bounded_proposals,apply_multiedit
from diagnostic_io import atomic_json


def violation_proposals(backbone, current, pool):
    cn=np.linalg.norm(backbone[:-1,2]-backbone[1:,0],axis=-1)
    positions=set(np.where((cn<1)|(cn>1.8))[0].tolist())
    positions.update(i+1 for i in list(positions))
    ca=backbone[:,1]
    x,y=np.where(np.triu(np.linalg.norm(ca[:,None]-ca[None],axis=-1)<3,1))
    positions.update(x.tolist());positions.update(y.tolist())
    return [(site,int(token)) for site in sorted(positions) for token in pool[site] if token!=current[site]]


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--chunk',type=int,default=0);p.add_argument('--preflight-only',action='store_true');p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--threads',type=int,default=16);a=p.parse_args()
    torch.set_num_threads(a.threads)
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Slurm GPU required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol_path=a.run/'protocol.json';protocol=json.loads(protocol_path.read_text());protocol_hash=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    for file,digest in protocol['code_sha256'].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
    source=Path(protocol['source']);device=torch.device(a.device)
    def synchronize():
        if device.type=='cuda':torch.cuda.synchronize()
    load_start=time.perf_counter();decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30)
    state=torch.load(protocol['weights'],map_location='cpu',mmap=True,weights_only=False)
    decoder.load_state_dict(state.get('decode_proj',state),strict=True);del state
    decoder.to(device).eval().requires_grad_(False);synchronize();load_seconds=time.perf_counter()-load_start
    def decode(ids):
        return decode_embedding(decoder,decoder.embed(torch.from_numpy(ids).long().to(device))).cpu().numpy()
    targets=protocol['targets'][:1] if a.preflight_only else protocol['targets'][a.chunk::protocol['chunks']]
    for item in targets:
        directory=a.run/('preflight' if a.preflight_only else item['target']);directory.mkdir(exist_ok=True)
        if (directory/'summary.json').exists():
            old=json.loads((directory/'summary.json').read_text())
            assert old['protocol_sha256']==protocol_hash and old['status']=='complete'
            print('RESUME_COMPLETED',item['target'],flush=True);continue
        path=source/item['input'];assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol['input_sha256'][item['target']]
        with np.load(path) as z:original=z['fine'].copy();logits=torch.from_numpy(z['logits'].copy())
        assert np.array_equal(logits.argmax(-1).numpy(),original)
        pool=logits.topk(5).indices.numpy();budget=item['decoder_call_budget'];index=protocol['targets'].index(item)
        baseline=decode(original);assert np.isfinite(baseline).all();baseline_metrics=summarize_backbone(baseline)
        stream=local_proposals(pool,original,budget,20261022+index)
        multi={name:bounded_proposals(pool,original,budget,protocol['multiedit_seed']+index,protocol['edit_fraction_cap'],name) for name in ['uniform','geometric']}
        results={};arrays={};order=protocol['arms'];offset=index%len(order);order=order[offset:]+order[:offset]
        for arm in order:
            current=original.copy();best_bb=baseline.copy();best_metrics=baseline_metrics
            key=(best_metrics['backbone_cn_outliers'],best_metrics['upstream_ca_clashes'])
            proposals=[];cursor=0;trace=[];edit_hist=Counter();padding_calls=0
            reasons=dict(edit_cap_exceeded=0,displacement_exceeded=0,geometry_pareto_failed=0,eligible=0)
            synchronize();started=time.perf_counter()
            for call in range(budget):
                padding=False
                if arm=='violation_greedy':
                    if not proposals or cursor>=len(proposals):
                        proposals=violation_proposals(best_bb,current,pool);cursor=0
                    padding=not proposals
                    ids=current.copy() if padding else apply_proposal(current,proposals[cursor])
                    if not padding:cursor+=1
                elif arm.startswith('independent_multi_'):
                    proposal=multi[arm.removeprefix('independent_multi_')][call]
                    ids=apply_multiedit(original,proposal);padding=len(proposal[0])==0
                else:
                    ids=apply_proposal(current if arm=='random_greedy' else original,stream[call])
                bb=decode(ids);metrics=summarize_backbone(bb)
                if not np.isfinite(bb).all():raise ValueError('Nonfinite candidate; do not silently drop')
                gate=evaluate_candidate(ids,bb,original,baseline,metrics,baseline_metrics,protocol['delta_angstrom'],protocol['edit_fraction_cap'])
                edits=int(np.count_nonzero(ids!=original));edit_hist[str(edits)]+=1;padding_calls+=int(padding)
                reasons['edit_cap_exceeded']+=int(gate['changed_fraction']>protocol['edit_fraction_cap'])
                if arm.startswith('independent_multi_'):assert gate['changed_fraction']<=protocol['edit_fraction_cap']
                reasons['displacement_exceeded']+=int(gate['aligned_ca_displacement'] is None or gate['aligned_ca_displacement']>protocol['delta_angstrom'])
                cn,ca=metrics['backbone_cn_outliers'],metrics['upstream_ca_clashes']
                pareto=cn<=baseline_metrics['backbone_cn_outliers'] and ca<=baseline_metrics['upstream_ca_clashes'] and (cn<baseline_metrics['backbone_cn_outliers'] or ca<baseline_metrics['upstream_ca_clashes'])
                reasons['geometry_pareto_failed']+=int(not pareto);reasons['eligible']+=int(gate['eligible'])
                accepted=not padding and gate['eligible'] and (cn,ca)<key
                if accepted:
                    current=ids.copy();best_bb=bb.copy();best_metrics=metrics;key=(cn,ca)
                    if arm=='violation_greedy':proposals=[]
                trace.append(dict(call=call+1,accepted=accepted,padding=padding,proposed_total_edits=edits,gate=gate,cn_count=cn,ca_clashes=ca))
            synchronize();elapsed=time.perf_counter()-started
            assert len(trace)==budget
            arrays[arm]=(current.copy(),best_bb.copy())
            results[arm]=dict(selected=best_metrics,returned_original=bool(np.array_equal(current,original)),gate=evaluate_candidate(current,best_bb,original,baseline,best_metrics,baseline_metrics,protocol['delta_angstrom'],protocol['edit_fraction_cap']),decoder_calls=budget,padding_calls=padding_calls,search_seconds=elapsed,proposal_edit_histogram=dict(edit_hist),rejection_diagnostics_overlapping=reasons,trace=trace)
        # Reference data is opened only after every arm's proposals and selection.
        corpus=Path(protocol['corpus']);refrow=next(x for x in json.loads((corpus/'manifest.json').read_text())['targets'] if x['id']==item['protein'])
        with np.load(corpus/refrow['backbone']) as z:reference=z['backbone'].copy()
        np.savez_compressed(directory/'baseline.npz',fine=original,backbone=baseline)
        for arm,(ids,bb) in arrays.items():
            np.savez_compressed(directory/(arm+'.npz'),fine=ids,backbone=bb)
            results[arm]['posthoc_metrics']=summarize_backbone(bb,[reference[:,1]])
            results[arm]['posthoc_quality']=measurements(bb,reference)
        atomic_json(directory/'summary.json',dict(status='complete',protocol_sha256=protocol_hash,target=item['target'],protein=item['protein'],sample=item['sample'],length=len(original),budget=budget,baseline=summarize_backbone(baseline,[reference[:,1]]),baseline_quality=measurements(baseline,reference),arms=results,arm_order=order,decoder_load_seconds=load_seconds,host=socket.gethostname(),device=str(device),threads=a.threads,gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,precision='FP32;TF32disabled;all arms same host and device',scope=protocol['scope']))
        print('BOUNDED_MULTIEDIT_COMPLETE',item['target'],budget,{k:round(v['search_seconds'],3) for k,v in results.items()},flush=True)
    if a.preflight_only:
        d=json.loads((a.run/'preflight'/'summary.json').read_text());assert all(v['decoder_calls']==d['budget'] for v in d['arms'].values())
        assert all(d['arms'][arm]['rejection_diagnostics_overlapping']['edit_cap_exceeded']==0 for arm in protocol['arms'] if arm.startswith('independent_multi_'))
        atomic_json(a.run/'preflight.json',dict(status='passed',target=targets[0]['target'],protocol_sha256=protocol_hash))
    else:atomic_json(a.run/('chunk'+str(a.chunk)+'.json'),dict(status='complete',inputs=len(targets),protocol_sha256=protocol_hash))

if __name__=='__main__':main()
