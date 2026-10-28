import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from spatial_packing import segments,pooled_broadcast
from structural_metrics import summarize_backbone,aligned_rmsd
from backbone_quality_audit import measurements
from diagnostic_io import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--chunk',type=int,required=True);a=p.parse_args();torch.set_num_threads(4);assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device=torch.device('cuda');protocol=json.loads((a.run/'protocol.json').read_text())
    for f,h in protocol['code_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
    codes=torch.load(protocol['codes'],map_location='cpu',weights_only=True)['fine_codes'].float().to(device);decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30);state=torch.load(protocol['weights'],map_location='cpu',mmap=True,weights_only=False);decoder.load_state_dict(state.get('decode_proj',state),strict=True);del state;decoder.to(device).eval().requires_grad_(False)
    def snap(x):return torch.cdist(torch.as_tensor(x,device=device).float(),codes).argmin(-1)
    for item in protocol['targets'][a.chunk::protocol['chunks']]:
        with np.load(item['latent']) as z:latent=z['latent'].copy();ids=z['fine'].copy()
        assert np.array_equal(snap(latent).cpu().numpy(),ids)
        with np.load(item['reference']) as z:reference=z['backbone'].copy()
        n=len(ids);fixed=[(i,min(n,i+4)) for i in range(0,n,4)];parts=segments(item['ss3']);capped=segments(item['ss3'],4);origin=decoder.embed(torch.from_numpy(ids).long().to(device));embeddings={'original':origin};counts={'original':n}
        for arm,partition in [('fixed4_snap',fixed),('dssp_snap',parts),('dssp_cap4_snap',capped)]:embeddings[arm]=decoder.embed(snap(pooled_broadcast(latent,partition)));counts[arm]=len(partition)
        embeddings['dssp_embedding_average']=torch.from_numpy(pooled_broadcast(origin.cpu().numpy(),parts)).to(device);counts['dssp_embedding_average']=len(parts);results={};arrays={}
        for arm in protocol['arms']:
            bb=decode_embedding(decoder,embeddings[arm]).cpu().numpy();arrays[arm]=bb
            try:quality=measurements(bb,reference);error=None
            except ValueError as e:quality=None;error=str(e)
            results[arm]=dict(segments=counts[arm],compression_ratio=n/counts[arm],metrics=summarize_backbone(bb,[reference[:,1]]),quality=quality,quality_error=error,ca_displacement_from_original=aligned_rmsd(bb[:,1],arrays['original'][:,1]))
        directory=a.run/item['target'];directory.mkdir(exist_ok=True);np.savez_compressed(directory/'structures.npz',**arrays);atomic_json(directory/'summary.json',dict(status='complete',target=item['target'],length=n,arms=results,scope=protocol['scope']));print('PACKING_COMPLETE',item['target'],flush=True)
    atomic_json(a.run/f'chunk{a.chunk}.json',dict(status='complete'))


if __name__=='__main__':main()
