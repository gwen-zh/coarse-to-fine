"""Actual paired wall latency with both frozen models resident on one GPU."""
import argparse,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json
from gpu_latency_io import unpack,paired_summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();path=a.run/'protocol.json';protocol=json.loads(path.read_text());ph=hashlib.sha256(path.read_bytes()).hexdigest()
    for name,digest in protocol['code_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    workers={};logs={};records=[];metadata={};cold_started=time.perf_counter()
    def receive(role):
        line=workers[role].stdout.readline()
        if not line:raise RuntimeError(role+' worker exited; inspect '+str(a.run/(role+'_worker.log')))
        return json.loads(line)
    def request(role,message):
        workers[role].stdin.write(json.dumps(message)+'\n');workers[role].stdin.flush();return receive(role)
    def pipeline(item,mode,calls=None):
        started=time.perf_counter();g=request('generator',dict(sequence=item['sequence'],seed=item['seed'],logits=mode!='original'))
        message=dict(mode=mode,fine=g['fine'],max_calls=item['decoder_call_budget'] if calls is None else calls)
        if mode!='original':message['logits']=g['logits']
        d=request('decoder',message);elapsed=time.perf_counter()-started
        assert d['original_fine']==g['fine']
        return dict(wall_seconds=elapsed,generation_seconds=g['generation_seconds'],decoder=d,initial_fine=g['fine'],coarse=g['coarse'])
    try:
        for role in ['generator','decoder']:
            logs[role]=(a.run/(role+'_worker.log')).open('w')
            workers[role]=subprocess.Popen([sys.executable,'-u','upstream/experiments/gpu_latency_worker.py','--role',role,'--protocol',str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=logs[role],text=True,bufsize=1,env=os.environ.copy())
            metadata[role]=receive(role);assert metadata[role]['status']=='ready';print('WORKER_READY',role,metadata[role],flush=True)
        assert metadata['generator']['host']==metadata['decoder']['host']==socket.gethostname()
        assert metadata['generator']['visible_devices']==metadata['decoder']['visible_devices']
        cold_seconds=time.perf_counter()-cold_started
        atomic_json(a.run/'startup.json',dict(protocol_sha256=ph,workers=metadata,cold_process_startup_seconds=cold_seconds,scope='Includes imports and both model loads; filesystem caches not flushed. Both workers remain resident.'))
        # Warm all fixed lengths and complete both IPC paths, including a proposal.
        for item in protocol['targets']:
            b=pipeline(item,'original');s=pipeline(item,'fixed_calls',calls=1)
            assert b['initial_fine']==s['initial_fine'] and b['coarse']==s['coarse']
        atomic_json(a.run/'preflight.json',dict(status='passed',protocol_sha256=ph,paired_seed_replay=True,targets=len(protocol['targets'])))
        references={}
        for mode,repeats in [('fixed_calls',protocol['fixed_repeats']),('wall30',protocol['wall_repeats'])]:
            for repeat in range(repeats):
                for index,item in enumerate(protocol['targets']):
                    outputs={};order=['original',mode] if (repeat+index)%2==0 else [mode,'original']
                    for arm in order:outputs[arm]=pipeline(item,arm,calls=100000 if arm=='wall30' else None)
                    b,s=outputs['original'],outputs[mode]
                    if b['initial_fine']!=s['initial_fine'] or b['coarse']!=s['coarse']:raise ValueError('Paired original tokens differ; timing comparison invalid')
                    if mode=='fixed_calls':
                        previous=references.setdefault(item['target'],s['decoder']['selected_fine'])
                        if previous!=s['decoder']['selected_fine']:raise ValueError('Fixed-call repeated outputs differ')
                    row=dict(target=item['target'],protein=item['protein'],mode=mode,repeat=repeat,order=order,original_wall_seconds=b['wall_seconds'],refined_wall_seconds=s['wall_seconds'],original_generation_seconds=b['generation_seconds'],refined_generation_seconds=s['generation_seconds'],original_decode_seconds=b['decoder']['baseline_decode_seconds'],refined_initial_decode_seconds=s['decoder']['baseline_decode_seconds'],search_seconds=s['decoder']['search_seconds'],decoder_calls=s['decoder']['decoder_calls'],padding_calls=s['decoder'].get('padding_calls',0),initial_token_sha256=hashlib.sha256(np.asarray(b['initial_fine'],np.int64).tobytes()).hexdigest(),selected_token_sha256=hashlib.sha256(np.asarray(s['decoder']['selected_fine'],np.int64).tobytes()).hexdigest())
                    records.append(row);atomic_json(a.run/'progress.json',dict(status='running',protocol_sha256=ph,records=records))
                    out=a.run/'outputs';out.mkdir(exist_ok=True);np.savez_compressed(out/(item['target']+'_'+mode+'_'+str(repeat)+'.npz'),initial_fine=np.array(b['initial_fine']),selected_fine=np.array(s['decoder']['selected_fine']),original=unpack(b['decoder']['backbone']),selected=unpack(s['decoder']['backbone']))
                    print('GPU_PAIRED_TIMING',mode,repeat,item['target'],round(b['wall_seconds'],3),round(s['wall_seconds'],3),s['decoder']['decoder_calls'],flush=True)
        memory={role:request(role,{'action':'stats'}) for role in workers}
        result=dict(status='complete',protocol_sha256=ph,workers=metadata,cold_process_startup_seconds=cold_seconds,memory_by_worker=memory,modes={mode:paired_summary([r for r in records if r['mode']==mode]) for mode in ['fixed_calls','wall30']},records=records,scope=protocol['scope'],limitations='4fixed development proteins x2outputs; timing not efficacy. Actual parent wall includes sequence tokenization, logits preparation, CPU/GPU transfers, base64 IPC and returned coordinates. Both model processes resident same allocated GPU; not singleprocess optimizeddeployment. Fixed-call budget inherited from CPU development; GPU generation may differ from priorCPU samples. Wall30 uses more/variable calls, reported separately. Filesystemcache not flushed; memory peaks perprocess are not an exact combinedGPUpeak.')
        atomic_json(a.run/'comparison.json',result)
        lines=['同 GPU 完整管线计时','='*50,'','固定4蛋白×2输出；两模型在同GPU常驻隔离进程。计时包含IPC及坐标返回，加载单列，不是拼接不同硬件的组件时间。','']
        for mode,x in result['modes'].items():lines.append(f"{mode}: 原路径 {x['mean_original_seconds']:.4f}s → 搜索路径 {x['mean_refined_seconds']:.4f}s；总耗时 {x['ratio_of_means']:.3f}×，增加 {x['added_percent']:.1f}%")
        lines+=['',f'进程启动及模型加载总计 {cold_seconds:.3f}s（未清空文件缓存）','',result['limitations']]
        (a.run/'results_zh.rst').write_text('\n'.join(lines)+'\n');print('GPU_PIPELINE_TIMING_COMPLETE',json.dumps(result['modes']),flush=True)
    finally:
        for role,worker in workers.items():
            if worker.poll() is None:
                try:worker.stdin.write('{"action":"stop"}\n');worker.stdin.flush();worker.wait(timeout=10)
                except Exception:worker.terminate()
        for f in logs.values():f.close()

if __name__=='__main__':main()
