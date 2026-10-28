"""Sequence-feature-only sampling with the experimental trained mixed generator.

No reference structure or encoded native coarse/fine labels are read here.
AA features must be computed using the matching AA-only extractor. Existing
experimental generation quality is limited; this is a research entry point.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from project_model import ROOT,load_tokenizer,load_generator
from structural_metrics import summarize_backbone

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--generator',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/demo_generation')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--samples',type=int,default=1)
    p.add_argument('--coarse-steps',type=int,default=16);p.add_argument('--fine-steps',type=int,default=16)
    args=p.parse_args()
    if args.samples<1 or args.threads<1:raise ValueError('Positive sample/thread counts required')
    torch.set_num_threads(args.threads)
    with np.load(args.features) as z:h=torch.from_numpy(z['h'].copy()).float().to(args.device)
    if h.ndim!=2 or h.shape[1]!=1536 or not torch.isfinite(h).all():raise ValueError('Expected finite AA-only h[N,1536]')
    decoder=load_tokenizer(args.tokenizer,args.device);generator=load_generator(args.generator,args.device)
    if generator.config['fine_size']!=decoder.model.fine_book.size:raise ValueError('Tokenizer/generator fine semantics must match')
    args.output.mkdir(parents=True,exist_ok=True);reports=[]
    for i in range(args.samples):
        seed=(args.seed+i)%(2**32);started=time.perf_counter()
        c,f=generator.sample(h,seed,args.coarse_steps,args.fine_steps);bb=decoder.decode(c,f)
        if args.device=='cuda':torch.cuda.synchronize()
        seconds=time.perf_counter()-started;xyz=bb.cpu().numpy()
        np.savez_compressed(args.output/f'sample{i}.npz',backbone=xyz,coarse=c.cpu().numpy(),fine=f.cpu().numpy())
        reports.append(dict(sample=i,seed=seed,seconds=seconds,geometry=summarize_backbone(xyz)))
    report=dict(residues=len(h),samples=reports,reference_structure_read=False,scope='Experimental sequence-feature-only sampling;not a confirmed improvement over the author model.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
