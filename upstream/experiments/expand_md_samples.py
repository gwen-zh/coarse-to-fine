"""Frozen5-target MD sample expansion, paired original/search, three seed families."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json
from gpu_latency_io import unpack


def sample_seed(family,target,sample):
    return int(hashlib.sha256((f'md-pilot-v1:{family}:{target}_{sample}').encode()).hexdigest()[:8],16)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,required=True);a=parser.parse_args();run=Path('artifacts/md_emulation_expansion_v1');path=run/'protocol.json';p=json.loads(path.read_text());ph=hashlib.sha256(path.read_bytes()).hexdigest()
    for name,digest in p['sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Changed source '+name)
    if not 0<=a.index<len(p['targets'])*len(p['families']):raise ValueError('Array index')
    family=p['families'][a.index//len(p['targets'])];target=p['targets'][a.index%len(p['targets'])];name=target['id'];directory=run/f'family{family}'/name;directory.mkdir(parents=True,exist_ok=True)
    records=[];workers={};logs={};metadata={}
    for i in range(32 if family==2026 else 0):
        parent=Path(p['parent'])/'outputs'/(name+'_'+str(i)+'.npz');expected=p['parent_outputs_sha256'][str(parent)]
        if hashlib.sha256(parent.read_bytes()).hexdigest()!=expected:raise RuntimeError('Parent output changed')
        records.append(dict(sample=i,seed=sample_seed(family,name,i),source=str(parent),sha256=expected,reused=True))
    def receive(role):
        line=workers[role].stdout.readline()
        if not line:raise RuntimeError(role+' worker exited, inspect per-target log')
        return json.loads(line)
    def request(role,message):workers[role].stdin.write(json.dumps(message)+'\n');workers[role].stdin.flush();return receive(role)
    try:
        pending=[]
        for sample in range(len(records),p['samples_per_protein']):
            meta=directory/(str(sample)+'.json')
            if meta.exists():
                row=json.loads(meta.read_text())
                if row['protocol_sha256']!=ph or row['seed']!=sample_seed(family,name,sample) or hashlib.sha256(Path(row['source']).read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Resume mismatch')
                records.append(row)
            else:pending.append(sample)
        if pending:
            for role in ['generator','decoder']:
                logs[role]=(directory/(role+'_worker.log')).open('a');workers[role]=subprocess.Popen([sys.executable,'-u','upstream/experiments/gpu_latency_worker.py','--role',role,'--protocol',str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=logs[role],text=True,bufsize=1,env=os.environ.copy());metadata[role]=receive(role)
                if metadata[role]['status']!='ready':raise RuntimeError('Worker initialization')
            atomic_json(directory/'startup.json',dict(protocol_sha256=ph,workers=metadata,reused_parent_startup=str(Path(p['parent'])/'startup.json')))
        for sample in pending:
            seed=sample_seed(family,name,sample);started=time.perf_counter();g=request('generator',dict(sequence=target['sequence'],seed=seed,logits=True));b=request('decoder',dict(mode='original',fine=g['fine'],max_calls=0));s=request('decoder',dict(mode='fixed_calls',fine=g['fine'],logits=g['logits'],max_calls=p['max_calls']))
            if b['original_fine']!=s['original_fine'] or b['original_fine']!=g['fine']:raise RuntimeError('Pair mismatch')
            original=unpack(b['backbone']);selected=unpack(s['backbone'])
            if original.shape!=(target['length'],3,3) or selected.shape!=original.shape or not np.isfinite(original).all() or not np.isfinite(selected).all():raise RuntimeError('Invalid output: stop, do not drop sample')
            dest=directory/(str(sample)+'.npz');temp=dest.with_suffix('.tmp.npz');np.savez_compressed(temp,original=original,selected=selected,original_fine=np.array(g['fine']),selected_fine=np.array(s['selected_fine']),coarse=np.array(g['coarse']));temp.replace(dest)
            row=dict(sample=sample,seed=seed,source=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),reused=False,protocol_sha256=ph,decoder_calls=s['decoder_calls'],search_seconds=s['search_seconds'],generation_seconds=g['generation_seconds'],paired_seconds=time.perf_counter()-started,gate=s['gate'])
            atomic_json(directory/(str(sample)+'.json'),row);records.append(row);atomic_json(directory/'progress.json',dict(status='sampling',completed=len(records),expected=p['samples_per_protein']))
            if sample%16==0:print('MD_EXPANSION',family,name,sample,flush=True)
    finally:
        for worker in workers.values():
            if worker.poll() is None:
                try:worker.stdin.write('{"action":"stop"}\n');worker.stdin.flush();worker.wait(timeout=10)
                except Exception:worker.terminate();worker.wait(timeout=10)
        for log in logs.values():log.close()
    records=sorted(records,key=lambda x:x['sample'])
    if [r['sample'] for r in records]!=list(range(p['samples_per_protein'])):raise RuntimeError('Missing/duplicate samples')
    atomic_json(directory/'manifest.json',dict(status='complete',family=family,target=name,protocol_sha256=ph,records=records));atomic_json(directory/'progress.json',dict(status='complete',completed=len(records),expected=p['samples_per_protein']));print('MD_EXPANSION_TARGET_COMPLETE',family,name,flush=True)

if __name__=='__main__':main()
