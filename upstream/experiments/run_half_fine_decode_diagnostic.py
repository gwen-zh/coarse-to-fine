"""One-GPU, zero-training paired decoder interventions for half-fine checkpoints."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from diagnostic_io import atomic_json
from prepare_half_fine_tokenizer import sha
from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from train_multiscale_tokenizer import metrics,verify_sources
from half_fine_tokenizer import HalfFineTokenizer,retained_positions
from half_fine_decode_diagnostic import decode_variants


def measured(bb,reference):
    out=metrics(bb,reference);x=bb.cpu().numpy()
    cn=np.linalg.norm(x[:-1,2]-x[1:,0],axis=-1)
    out.update(cn_too_short_rate=float((cn<1).mean()),cn_too_long_rate=float((cn>1.8).mean()),
               median_cn_distance=float(np.median(cn)),median_adjacent_ca_distance=float(np.median(np.linalg.norm(np.diff(x[:,1],axis=0),axis=-1))))
    return out


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.run/'protocol.json').read_text());verify_sources(protocol)
    for path,h in protocol['source_sha256'].items():
        if sha(path)!=h:raise RuntimeError('Input hash mismatch: '+path)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    started=time.monotonic();device=torch.device('cuda');parent=Path(protocol['parent'])
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30)
    state=torch.load(protocol['weights'],map_location='cpu',mmap=True,weights_only=False);decoder.load_state_dict(state.get('decode_proj',state),strict=True);del state
    scale=decoder.embed.weight[:4096].std(0).detach().to(device);mean=decoder.embed.weight[:4096].mean(0).detach().to(device)
    decoder.to(device).eval().requires_grad_(False)
    fine_codes=torch.load(protocol['codes'],map_location='cpu',weights_only=True)['fine_codes'].float().to(device)
    models={};old={}
    for arm in protocol['arms']:
        checkpoint=torch.load(parent/arm/'final.pt',map_location='cpu',weights_only=True)
        assert checkpoint['mode']==arm and checkpoint['protocol_sha256']==protocol['parent_protocol_sha256']
        model=HalfFineTokenizer(arm).to(device);model.load_state_dict(checkpoint['model'],strict=True);model.eval().requires_grad_(False);models[arm]=model
        old[arm]={r['target']:r for r in json.loads((parent/arm/'evaluation.json').read_text())['targets']}
    directory=a.run/'decoded';directory.mkdir(exist_ok=True);records=[];calls=0
    for row in protocol['targets']:
        name=row['target']
        with np.load(row['tokens']) as data:fine=torch.as_tensor(data['fine'].copy(),device=device).long()
        with np.load(row['reference']) as data:reference=torch.as_tensor(data['backbone'].copy(),device=device)
        n=len(fine);keep=retained_positions(n,device)
        original_embedding=decoder.embed(fine)
        original=decode_embedding(decoder,original_embedding);calls+=1
        # Only retained fine IDs enter inference; true hidden labels remain evaluation-only.
        visible_codes=fine_codes.new_zeros(n,128);visible_codes[keep]=fine_codes[fine[keep]]
        visible_embeddings=mean.expand(n,-1).clone();visible_embeddings[keep]=decoder.embed(fine[keep])
        arrays={'original':original.cpu().numpy(),'retained':keep.cpu().numpy()}
        record=dict(target=name,length=n,original=measured(original,reference),arms={})
        for arm,model in models.items():
            with np.load(parent/arm/(name+'.npz')) as saved:
                coarse_ids=torch.as_tensor(saved['coarse_ids'].copy(),device=device).long()
                assert np.array_equal(saved['retained'],keep.cpu().numpy())
                prior=saved['trained'].copy()
            assert coarse_ids.shape==(n,) and bool(((coarse_ids>=0)&(coarse_ids<32)).all())
            coarse=model.quantizer.codes[coarse_ids]
            predicted=model.fusion(visible_codes,visible_embeddings,keep,coarse,scale,mean)
            variants,hidden_ids=decode_variants(predicted,original_embedding,keep,decoder.embed.weight[:4096],scale,mean)
            final_ids=fine.clone();final_ids[~keep]=hidden_ids
            assert torch.equal(final_ids[keep],fine[keep])
            torch.testing.assert_close(variants['hidden_snapped'],decoder.embed(final_ids),rtol=0,atol=0)
            results={};replay={};arrays[arm+'_snapped_fine_ids']=final_ids.cpu().numpy();arrays[arm+'_coarse_ids']=coarse_ids.cpu().numpy()
            for variant in protocol['variants']:
                bb=decode_embedding(decoder,variants[variant]);calls+=1
                results[variant]=measured(bb,reference);arrays[arm+'_'+variant]=bb.cpu().numpy()
                if variant=='continuous':
                    replay=dict(max_raw_coordinate_difference=float(np.abs(bb.cpu().numpy()-prior).max()),
                        ca_rmsd_difference=results[variant]['ca_rmsd']-old[arm][name]['methods']['trained']['ca_rmsd'],
                        cn_rate_difference=results[variant]['cn_rate']-old[arm][name]['methods']['trained']['cn_rate'])
            counts=torch.bincount(hidden_ids,minlength=4096).cpu().numpy()
            record['arms'][arm]=dict(methods=results,replay=replay,hidden_token_accuracy=float((hidden_ids==fine[~keep]).float().mean()),hidden_codes_used=int((counts>0).sum()))
        np.savez_compressed(directory/(name+'.npz'),**arrays);atomic_json(directory/(name+'.json'),record);records.append(record)
        print('DECODE_COMPLETE',name,'calls',calls,flush=True)
    assert calls==24*7
    parity={arm:{key:max(abs(r['arms'][arm]['replay'][key]) for r in records) for key in ('max_raw_coordinate_difference','ca_rmsd_difference','cn_rate_difference')} for arm in protocol['arms']}
    parity_ok=all(v['ca_rmsd_difference']<=.05 and v['cn_rate_difference']<=.01 for v in parity.values())
    atomic_json(a.run/'evaluation.json',dict(status='complete',scope=protocol['scope'],proteins=len(records),decoder_calls=calls,targets=records,replay=parity,replay_passed=parity_ok,device=torch.cuda.get_device_name(),elapsed_seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
    if not parity_ok:raise RuntimeError('Continuous replay beyond predefined tolerance; investigate before interpreting intervention')
    print('HALF_FINE_DECODE_COMPLETE',calls,parity,flush=True)

if __name__=='__main__':main()
