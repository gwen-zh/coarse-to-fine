import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from structural_metrics import summarize_backbone,aligned_rmsd
from backbone_quality_audit import measurements
from diagnostic_io import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--arm',choices=['ce','repa'],required=True);a=p.parse_args();torch.set_num_threads(4);assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device=torch.device('cuda');protocol=json.loads((a.run/'protocol.json').read_text());out=a.run/a.arm
    for attempt in range(31):
        if (out/'summary.json').exists():break
        if attempt==30:raise RuntimeError('Training completion marker unavailable')
        time.sleep(2)
    assert json.loads((out/'summary.json').read_text())['status']=='complete'
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30);state=torch.load('artifacts/tokenizer/decoder_checkpoint_110.pth',map_location='cpu',mmap=True,weights_only=False);decoder.load_state_dict(state.get('decode_proj',state),strict=True);del state;decoder.to(device).eval().requires_grad_(False);corpus=Path('artifacts/pdb_training_v2');rows={r['id']:r for r in json.loads((corpus/'manifest.json').read_text())['targets']};directory=out/'decoded';directory.mkdir(exist_ok=True)
    for name in protocol['geometry_targets']:
        with np.load(corpus/rows[name]['backbone']) as z:reference=z['backbone'].copy()
        arrays={};samples=[]
        for index in range(8):
            with np.load(out/'predictions'/f'{name}_sample{index}.npz') as z:ids={key:z[key].copy() for key in ['fine','native_fine']}
            record={}
            for key,value in ids.items():
                bb=decode_embedding(decoder,decoder.embed(torch.from_numpy(value).long().to(device))).cpu().numpy();arrays[f'{key}_{index}']=bb
                try:quality=measurements(bb,reference);error=None
                except ValueError as e:quality=None;error=str(e)
                record[key]=dict(metrics=summarize_backbone(bb,[reference[:,1]]),quality=quality,quality_error=error)
            record['fine_change_fraction']=float(np.mean(ids['fine']!=ids['native_fine']));samples.append(record)
        spreads={key:float(np.mean([aligned_rmsd(arrays[f'{key}_{i}'][:,1],arrays[f'{key}_{j}'][:,1]) for i in range(8) for j in range(i+1,8)])) for key in ['fine','native_fine']}
        np.savez_compressed(directory/(name+'.npz'),**arrays);atomic_json(directory/(name+'.json'),dict(status='complete',target=name,length=len(reference),samples=samples,spread=spreads));print('REPA_DECODE_COMPLETE',a.arm,name,flush=True)
    atomic_json(out/'decode_summary.json',dict(status='complete',proteins=24,samples=192))


if __name__=='__main__':main()
