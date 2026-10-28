"""Paired frozen-model MD projection pilot; no reference-guided sample selection."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json
from gpu_latency_io import unpack
from md_projection_adapter import ca_trajectory


def main():
    run=Path('artifacts/md_emulation_pilot_v1');path=run/'sampling_protocol.json';protocol=json.loads(path.read_text())
    for name,digest in protocol['sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Changed source: '+name)
    workers={};logs={};records=[];metadata={};outputs=run/'outputs';outputs.mkdir(exist_ok=True)
    def receive(role):
        line=workers[role].stdout.readline()
        if not line:raise RuntimeError(role+' worker exited; inspect worker log')
        return json.loads(line)
    def request(role,message):
        workers[role].stdin.write(json.dumps(message)+'\n');workers[role].stdin.flush();return receive(role)
    try:
        for role in ['generator','decoder']:
            logs[role]=(run/(role+'_worker.log')).open('w')
            workers[role]=subprocess.Popen([sys.executable,'-u','upstream/experiments/gpu_latency_worker.py','--role',role,'--protocol',str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=logs[role],text=True,bufsize=1,env=os.environ.copy())
            metadata[role]=receive(role)
            if metadata[role]['status']!='ready':raise RuntimeError('Worker initialization failed')
        atomic_json(run/'startup.json',metadata)
        for target in protocol['targets']:
            for sample in range(protocol['samples_per_protein']):
                name=target['id']+'_'+str(sample);dest=outputs/(name+'.npz')
                # No resume ambiguity: outputs are written only after the full paired record.
                if dest.exists():raise RuntimeError('Existing output; explicit resume audit required')
                seed=int(hashlib.sha256(('md-pilot-v1:2026:'+name).encode()).hexdigest()[:8],16)
                started=time.perf_counter()
                g=request('generator',dict(sequence=target['sequence'],seed=seed,logits=True))
                b=request('decoder',dict(mode='original',fine=g['fine'],max_calls=0))
                s=request('decoder',dict(mode='fixed_calls',fine=g['fine'],logits=g['logits'],max_calls=protocol['max_calls']))
                if b['original_fine']!=s['original_fine'] or b['original_fine']!=g['fine']:raise RuntimeError('Pair mismatch')
                original=unpack(b['backbone']);selected=unpack(s['backbone'])
                if not np.isfinite(original).all() or not np.isfinite(selected).all():
                    atomic_json(run/'failure.json',dict(target=name,reason='Nonfinite output; do not filter silently'))
                    raise RuntimeError('Nonfinite output')
                np.savez_compressed(dest,original=original,selected=selected,original_fine=np.array(g['fine']),selected_fine=np.array(s['selected_fine']),coarse=np.array(g['coarse']))
                row=dict(target=target['id'],sample=sample,seed=seed,decoder_calls=s['decoder_calls'],padding_calls=s.get('padding_calls',0),search_seconds=s['search_seconds'],generation_seconds=g['generation_seconds'],base_decode_seconds=b['baseline_decode_seconds'],paired_wall_seconds=time.perf_counter()-started,returned_original=s['returned_original'],final_gate=s['gate'])
                records.append(row);atomic_json(run/'progress.json',dict(status='sampling',completed=len(records),expected=len(protocol['targets'])*protocol['samples_per_protein'],records=records))
                print('MD_PAIRED_SAMPLE',name,flush=True)
    finally:
        for worker in workers.values():
            if worker.poll() is None:
                try:worker.stdin.write('{"action":"stop"}\n');worker.stdin.flush();worker.wait(timeout=10)
                except Exception:worker.terminate();worker.wait(timeout=10)
        for log in logs.values():log.close()
    # Isolated workers have exited before importing the evaluator or reading MD outcomes.
    sys.path.insert(0,str(Path('upstream/bioemu-benchmarks').resolve()))
    from bioemu_benchmarks.eval.md_emulation.projections import project_samples
    from bioemu_benchmarks.eval.md_emulation.utils import load_projection_parameters,load_reference_projections
    from bioemu_benchmarks.eval.md_emulation.state_metric import compute_state_metrics
    from structural_metrics import summarize_backbone
    projections={};raw_geometry=[]
    for arm in ['original','selected']:
        trajectories={}
        for target in protocol['targets']:
            backbones=[]
            for sample in range(protocol['samples_per_protein']):
                with np.load(outputs/(target['id']+'_'+str(sample)+'.npz')) as a:bb=a[arm].copy()
                backbones.append(bb);raw_geometry.append(dict(arm=arm,target=target['id'],sample=sample,metrics=summarize_backbone(bb)))
            trajectories[target['id']]=ca_trajectory(target['sequence'],np.stack(backbones)[:,:,1,:])
        projections[arm]=project_samples(trajectories,load_projection_parameters())
        np.savez_compressed(run/(arm+'_projections.npz'),**projections[arm])
        metrics=compute_state_metrics(projections[arm],load_reference_projections(),temperature_K=300.,random_seed=2026)
        metrics.to_csv(run/(arm+'_md_metrics.csv'))
    atomic_json(run/'raw_geometry.json',raw_geometry)
    atomic_json(run/'progress.json',dict(status='complete',completed=len(records),records=records,scope='5 proteins x32 paired outputs, one seed family, exploratory sample-size pilot; no filtering. Fixed64 decoder proposals, not the historical30s confirmation budget. No thermodynamic superiority or kinetic claim.'))
    print('MD_PILOT_COMPLETE',flush=True)

if __name__=='__main__':main()
