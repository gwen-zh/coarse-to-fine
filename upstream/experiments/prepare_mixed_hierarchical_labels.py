"""Export frozen mixed labels; normalization strictly uses training proteins."""
import json,time
import numpy as np
import torch
from pathlib import Path
from mixed_hierarchical_io import RUN,protocol,tokenizer,sha
from hierarchical_pipeline_io import load_items
from diagnostic_io import atomic_json


@torch.no_grad()
def main():
 p=protocol();torch.set_num_threads(1);items=load_items(p);started=time.perf_counter();summaries=[]
 for seed in p['seeds']:
  root=RUN/'labels'/f'seed{seed}';root.mkdir(parents=True,exist_ok=True)
  if (root/'manifest.json').exists():raise RuntimeError('Immutable label export already complete')
  decoder=tokenizer(p,seed);rows=[];means=[];squares=[];parity=0
  for item in items:
   name=item['row']['id'];c,f=decoder.encode(item['bb']);centers=decoder.coarse_centers(c,len(f));path=root/(name+'.npz')
   np.savez_compressed(path,coarse=c.numpy(),fine=f.numpy(),centers=centers.numpy());rows.append(dict(id=name,subset=item['row']['subset'],sha256=sha(path)))
   if item['row']['subset']=='train':means.append(c.double().mean(0).numpy());squares.append(c.double().square().mean(0).numpy())
   if item['row']['subset']=='validation':
    old=Path(p['hybrid_diagnostic'])/f'seed{seed}'/(name+'.npz')
    with np.load(old) as z:
     if not np.array_equal(c.numpy(),z['coarse_latent']) or not np.array_equal(f.numpy(),z['fine']):raise RuntimeError('Hybrid labels changed '+name)
     torch.testing.assert_close(decoder.decode(c,f),torch.from_numpy(z['backbone']),atol=0,rtol=0)
    parity+=1
  mean=np.mean(means,0);std=np.sqrt(np.maximum(np.mean(squares,0)-mean**2,1e-6))
  stats=dict(mean=mean.astype(np.float32).tolist(),std=std.astype(np.float32).tolist(),training_proteins=len(means),definition='Protein-equal channel moments,training899only;std floor1e-3')
  atomic_json(root/'stats.json',stats)
  result=dict(status='complete',targets=rows,stats_sha256=sha(root/'stats.json'),seed=seed,validation_hybrid_replays=parity,protocol_sha256=sha(RUN/'protocol.json'))
  atomic_json(root/'manifest.json',result);summaries.append(dict(seed=seed,targets=len(rows),validation_replays=parity));print('MIXED_LABELS_COMPLETE',seed,len(rows),parity,flush=True)
 atomic_json(RUN/'labels_complete.json',dict(status='complete',seeds=summaries,seconds=time.perf_counter()-started));print('MIXED_LABEL_PREPARATION_COMPLETE',flush=True)


if __name__=='__main__':main()
