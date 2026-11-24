"""New full PDB-only pretraining regime, not continuation of mixed MD probes."""
import argparse,copy,hashlib,json
from pathlib import Path
import torch,numpy as np
import run_block_frame_probe as engine
from block_stable_frame_tokenizer import StableBlockFrameFineTokenizer,stable_frame_context
from block_centroid_transport_frames import centroid_transport_frames
from compact_training_trace import CompactTrainingTrace,restore_trace_prefix
from mixed_hierarchical_generator import HybridTokenizer
from hierarchical_pipeline_io import sha
from diagnostic_io import atomic_json

RUN=Path('artifacts/block_frame_full_pdb_v1')
PARENT=Path('artifacts/block_stable_frame_tokenizer_v1')


def load_cache():
    p=json.loads((RUN/'cache_manifest.json').read_text())
    if sha(RUN/'labels.pt')!=p['cache_sha256']:raise RuntimeError('Changed PDB-pretraining cache')
    return torch.load(RUN/'labels.pt',map_location='cpu',weights_only=True)


def configure():
    engine.RUN=RUN;engine.load_cache=load_cache;engine.BlockFrameFineTokenizer=StableBlockFrameFineTokenizer


@torch.no_grad()
def prepare():
    RUN.mkdir(exist_ok=True)
    if (RUN/'preparation_protocol.json').exists():raise RuntimeError('Use a new full-pretraining namespace')
    parent=json.loads((PARENT/'protocol.json').read_text());parent_result=json.loads((PARENT/'report.json').read_text())
    for path,digest in parent['source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Changed parent input '+path)
    old=torch.load(PARENT/'labels.pt',map_location='cpu',weights_only=True);manifest=Path('artifacts/pdb_training_v2/manifest.json');all_targets=json.loads(manifest.read_text())['targets'];development=[r for r in all_targets if r['subset']=='validation']
    references=[dict(id=r['id'],sequence=r['sequence'],path=str(manifest.parent/r['backbone'])) for r in development]
    sources=[Path(__file__),Path('upstream/experiments/compact_training_trace.py'),Path('upstream/tests/test_compact_training_trace.py'),Path('upstream/scripts/block_frame_full_pdb.sbatch'),PARENT/'protocol.json',PARENT/'report.json',PARENT/'labels.pt',manifest,*[Path(r['path']) for r in references]]
    atomic_json(RUN/'preparation_protocol.json',dict(scope='New PDB-only representation-pretraining experiment. Both coordinate conventions start fresh with identical seed/model/data/update counts. Mixed-probe checkpoints are never extended. Coarse maps stay frozen. MD R2 is evaluated only; no MD sample enters optimizer batches or kmeans initialization. No R3 reuse.',source_sha256={str(x):sha(x) for x in sources}))
    base=HybridTokenizer.from_checkpoints(**parent['tokenizer']);lookup={r['id']:r for r in old['PDB_development']};groups={k:v for k,v in old.items() if k!='train_probe'};groups['PDB_development']=[]
    for row in references:
        if row['id'] in lookup:case=lookup[row['id']]
        else:
            with np.load(row['path'],allow_pickle=False) as z:x=torch.from_numpy(z['backbone'].copy()).float()
            c,_,target=base.model.encode_latents(x);centers=base.coarse_centers(c,len(x));frames,metadata=centroid_transport_frames(centers);contexts={arm:stable_frame_context(centers,len(x),arm,frames=frames,metadata=metadata) for arm in parent['arms']};case=dict(id=row['id'],target=target,coarse=c,contexts=contexts)
        groups['PDB_development'].append(case)
    initialization=sorted(groups['PDB_train'],key=lambda x:hashlib.sha256(('block-full-PDB-kmeans-v1:'+x['id']).encode()).hexdigest())[:512]
    groups['train_probe']=initialization
    if any(r['id'].startswith('R1:') or r['id'].startswith('R2:') for r in initialization):raise RuntimeError('MD leaked into PDB codebook initialization')
    torch.save(groups,RUN/'labels.pt');atomic_json(RUN/'cache_manifest.json',dict(cache_sha256=sha(RUN/'labels.pt'),groups={k:len(v) for k,v in groups.items()},training_domain='PDB_train only',kmeans_domain='512 PDB_train proteins only',R3_loaded=False))
    p=copy.deepcopy(parent);n=len(groups['PDB_train']);p['warmup_updates']=20*n;p['quantized_updates']=80*n;p['batch']={'MD_R1':0,'PDB_train':1};p['references']['development']=references;p['parent_trial']=str(PARENT/'protocol.json');p['training_regime']='PDB-only with replacement, 89900 draws = 100 corpus-size-equivalent passes; not literal permutation epochs. 20-equivalent-pass warmup and 80 quantized. No MD gradient or kmeans examples.';p['logging']='Append-only JSONL, compact JSON materialized on resume/final completion; checkpoints at fixed 512 steps. No checkpoint quality selection.';p['scope']='Matched representation-pretraining comparison only. All 101 PDB and MD R2 development cases are reused; no independent confirmation or new-sequence generation claim.'
    p['source_sha256'].update({str(x):sha(x) for x in [*sources,RUN/'preparation_protocol.json',RUN/'labels.pt',RUN/'cache_manifest.json']});atomic_json(RUN/'protocol.json',p)
    print('FULL_PDB_PRETRAIN_READY',json.dumps(dict(warmup=p['warmup_updates'],quantized=p['quantized_updates'],PDB_train=n,PDB_development=len(references),kmeans_proteins=len(initialization))),flush=True)


def train(p,arm,seed):
    out=RUN/arm/f'seed{seed}';checkpoint=out/'last.pt';start=0
    if checkpoint.exists():start=torch.load(checkpoint,map_location='cpu',weights_only=False)['updates']
    restored=restore_trace_prefix(out,start);engine.atomic_json=CompactTrainingTrace(p['warmup_updates']+p['quantized_updates'],restored)
    engine.train(p,arm,seed)


def report(p):
    engine.report(p);allowed={r['id'] for r in load_cache()['PDB_train']}
    for arm in p['arms']:
        for seed in p['seeds']:
            trace=json.loads((RUN/arm/f'seed{seed}/trace.json').read_text())
            if any(len(r['batch'])!=1 or r['batch'][0] not in allowed for r in trace):raise RuntimeError('Non-PDB sample found in optimizer trace')
    result=json.loads((RUN/'report.json').read_text());result['PDB_only_optimizer_domain_verified']=True;result['MD_used_for_kmeans']=False;result['scope']=p['scope'];atomic_json(RUN/'report.json',result)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',required=True,choices=['prepare','train','report']);a.add_argument('--arm',choices=['reference','local']);a.add_argument('--seed',type=int,default=2026);args=a.parse_args();torch.set_num_threads(2)
    if args.stage=='prepare':prepare()
    else:
        configure();p=engine.protocol()
        if args.stage=='train':
            if args.arm not in p['arms'] or args.seed not in p['seeds']:raise RuntimeError('Undeclared full-pretraining arm')
            train(p,args.arm,args.seed)
        else:report(p)
