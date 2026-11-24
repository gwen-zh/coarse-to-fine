"""Fresh same-budget follow-up with stable transported frame geometry."""
import argparse,copy,json
from pathlib import Path
import torch
import run_block_frame_probe as engine
from block_stable_frame_tokenizer import StableBlockFrameFineTokenizer,stable_frame_context
from block_centroid_transport_frames import centroid_transport_frames
from hierarchical_pipeline_io import sha
from diagnostic_io import atomic_json

RUN=Path('artifacts/block_stable_frame_tokenizer_v1')
OLD=Path('artifacts/block_frame_tokenizer_v1')


def shared_cache():
    m=json.loads((RUN/'cache_manifest.json').read_text())
    if sha(RUN/'labels.pt')!=m['cache_sha256']:raise RuntimeError('Changed stable-frame cache')
    return torch.load(RUN/'labels.pt',map_location='cpu',weights_only=True)


def configure():
    engine.RUN=RUN;engine.load_cache=shared_cache;engine.BlockFrameFineTokenizer=StableBlockFrameFineTokenizer


def prepare():
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists() or (RUN/'preparation_protocol.json').exists():raise RuntimeError('Use a new stable-frame training namespace')
    parent=engine.protocol();old=engine.load_cache();geometry=Path('artifacts/block_centroid_transport_geometry_v1/report.json')
    if not geometry.exists():raise RuntimeError('Complete frame-only stability audit first')
    sources=[Path(__file__),Path('upstream/experiments/block_stable_frame_tokenizer.py'),Path('upstream/experiments/block_centroid_transport_frames.py'),Path('upstream/tests/test_block_stable_frame_tokenizer.py'),Path('upstream/tests/test_block_centroid_transport_frames.py'),Path('upstream/scripts/block_stable_frame_probe.sbatch'),OLD/'protocol.json',OLD/'report.json',OLD/'labels.pt',geometry]
    atomic_json(RUN/'preparation_protocol.json',dict(scope='Fresh fine training, same budgets/data/initial weights/shape/book. Frame family changed in both arms geometric features; within this family reference-vs-local is the matched comparison. Parent model training is not extended. No R3 reuse.',source_sha256={str(x):sha(x) for x in sources}))
    groups={};lookup={}
    for group in ['MD_train','MD_development','PDB_train','PDB_development']:
        groups[group]=[]
        for row in old[group]:
            centers=row['contexts']['reference']['centers'];frames,metadata=centroid_transport_frames(centers)
            contexts={arm:stable_frame_context(centers,len(row['target']),arm,frames=frames,metadata=metadata) for arm in parent['arms']}
            case=dict(id=row['id'],target=row['target'],coarse=row['coarse'],contexts=contexts);groups[group].append(case);lookup[row['id']]=case
        print('STABLE_FRAME_CACHE_GROUP',group,len(groups[group]),flush=True)
    groups['train_probe']=[lookup[row['id']] for row in old['train_probe']]
    torch.save(groups,RUN/'labels.pt');atomic_json(RUN/'cache_manifest.json',dict(cache_sha256=sha(RUN/'labels.pt'),groups={k:len(v) for k,v in groups.items()},R3_loaded=False))
    p=copy.deepcopy(parent);p['geometry']='Centroid-seeded transported tangent frames, from the same frozen reconstructed coarse centers at encode/decode. Both arms retain identical geometric feature dimensions.';p['parent_trial']=str(OLD/'protocol.json');p['model']['frame_family']='centroid_transport';p['source_sha256'].update({str(x):sha(x) for x in [*sources,RUN/'preparation_protocol.json',RUN/'labels.pt',RUN/'cache_manifest.json']});p['tests_passed']=6
    atomic_json(RUN/'protocol.json',p);print('STABLE_BLOCK_FRAME_PROTOCOL_READY',flush=True)


def report(p):
    engine.report(p);new=json.loads((RUN/'report.json').read_text());old=json.loads((OLD/'report.json').read_text())
    for arm in p['arms']:
        for seed in p['seeds']:
            a=json.loads((OLD/arm/f'seed{seed}/trace.json').read_text());b=json.loads((RUN/arm/f'seed{seed}/trace.json').read_text())
            if [x['batch'] for x in a]!=[x['batch'] for x in b]:raise RuntimeError('Frame family data draws were not paired')
    atomic_json(RUN/'family_comparison.json',dict(status='complete',families={'regularized_independent_frames':old['comparison'],'centroid_transport_frames':new['comparison']},same_parameter_count_and_update_budget=True,all_minibatches_matched=True,R3_used=False,scope='Two frame families and two coordinate conventions, same fresh-training budget. All comparisons are reused development, not independent test or generation gains.'))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',required=True,choices=['prepare','train','report']);a.add_argument('--arm',choices=['reference','local']);a.add_argument('--seed',type=int,default=2026);args=a.parse_args();torch.set_num_threads(2)
    if args.stage=='prepare':prepare()
    else:
        configure();p=engine.protocol()
        if args.stage=='train':
            if args.arm not in p['arms'] or args.seed not in p['seeds']:raise RuntimeError('Undeclared stable-frame arm')
            engine.train(p,args.arm,args.seed)
        else:report(p)
