"""Persistent workers isolate the incompatible tokenizer/generator ESM packages."""
import argparse,contextlib,json,os,socket,sys,time
from pathlib import Path
import numpy as np
from gpu_latency_io import pack,unpack


def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['generator','decoder'],required=True);p.add_argument('--protocol',type=Path,required=True);a=p.parse_args();protocol=json.loads(a.protocol.read_text())
    start=time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        torch.set_num_threads(protocol['threads'])
        if not torch.cuda.is_available():raise RuntimeError('Slurm GPU required')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        device=torch.device('cuda:0')
        if a.role=='generator':
            from train_pretrained_coarse import load_state_dict_from_lightning_ckpt,EsmSequenceTokenizer,tokenize_sequence
            model=load_state_dict_from_lightning_ckpt(Path(protocol['generator']),device=device).float().eval().requires_grad_(False)
            tokenizer=EsmSequenceTokenizer();head=model.net.output_heads.fine_structure_head[-1];capture=[];decision=[]
            handle=head.register_forward_pre_hook(lambda module,args:capture.append(args[0].detach()))
            native_wrapper=model._model_wrapper_inference
            def capture_decision(*args,**kwargs):
                result=native_wrapper(*args,**kwargs)
                if kwargs.get('scale')==2:decision.append(result[0].detach().clone())
                return result
            model._model_wrapper_inference=capture_decision
        else:
            from evaluate_residual_vq import ProjectedDecoder,decode_embedding
            from structural_metrics import summarize_backbone
            from fine_search_constraints import evaluate_candidate
            model=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30)
            state=torch.load(protocol['decoder'],map_location='cpu',mmap=True,weights_only=False);model.load_state_dict(state.get('decode_proj',state),strict=True);del state
            model.to(device).float().eval().requires_grad_(False)
    torch.cuda.synchronize();load=time.perf_counter()-start
    def reply(value):print(json.dumps(value,allow_nan=False),flush=True)
    reply(dict(status='ready',role=a.role,host=socket.gethostname(),gpu=torch.cuda.get_device_name(0),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),load_seconds=load))
    with torch.no_grad():
        for line in sys.stdin:
            req=json.loads(line)
            if req.get('action')=='stop':break
            if req.get('action')=='stats':
                reply(dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()));continue
            with contextlib.redirect_stdout(sys.stderr):
                if a.role=='generator':
                    torch.manual_seed(req['seed']);torch.cuda.manual_seed_all(req['seed']);np.random.seed(req['seed'])
                    seq=tokenize_sequence(req['sequence'],tokenizer)[None].to(device);n=len(req['sequence']);lengths=torch.tensor([n],device=device);capture.clear();decision.clear()
                    torch.cuda.synchronize();t=time.perf_counter()
                    fine,coarse=model.ddpm_sample(seq,lengths=lengths,num_steps=protocol['coarse_steps'],return_coarse=True)
                    torch.cuda.synchronize();generation=time.perf_counter()-t
                    ids=fine[0,1:n+1].cpu().numpy();response=dict(fine=ids.tolist(),coarse=coarse[0,1:n+1].cpu().tolist(),generation_seconds=generation)
                    if req.get('logits'):
                        t=time.perf_counter();logits=torch.nn.functional.linear(capture[-1][0,1:n+1].float(),head.weight[:4096],head.bias[:4096]);torch.cuda.synchronize()
                        value=logits.cpu().numpy()
                        exact=decision[-1][0,1:n+1,:4096].cpu().numpy()
                        if not np.array_equal(exact.argmax(-1),ids):raise ValueError('Captured native decision scores do not replay argmax')
                        mismatch=int(np.count_nonzero(value.argmax(-1)!=ids))
                        centered_error=float(np.max(np.abs((value-value.max(-1,keepdims=True))-(exact-exact.max(-1,keepdims=True)))))
                        response['score_extraction_audit']=dict(legacy_argmax_mismatches=mismatch,centered_max_abs_difference=centered_error,mode='native_decision_fallback' if mismatch else 'legacy_replay')
                        if mismatch:value=exact
                        response.update(logits=pack(value),fine_head_extract_and_transport_prepare_seconds=time.perf_counter()-t)
                    capture.clear();decision.clear()
                else:
                    ids=np.asarray(req['fine'],dtype=np.int64)
                    def decode(tokens):return decode_embedding(model,model.embed(torch.from_numpy(tokens).long().to(device))).cpu().numpy()
                    torch.cuda.synchronize();t=time.perf_counter();baseline=decode(ids);base_seconds=time.perf_counter()-t
                    response=dict(baseline_decode_seconds=base_seconds,original_fine=ids.tolist())
                    if req['mode']=='original':
                        response.update(selected_fine=ids.tolist(),backbone=pack(baseline),decoder_calls=0,search_seconds=0.)
                    else:
                        logits=unpack(req['logits']);pool=torch.from_numpy(logits).topk(5).indices.numpy()
                        if not np.array_equal(logits.argmax(-1),ids):raise ValueError('Transport mismatch')
                        bm=summarize_backbone(baseline);current=ids.copy();bbest=baseline.copy();best=bm;key=(bm['backbone_cn_outliers'],bm['upstream_ca_clashes'])
                        proposals=[];cursor=0;calls=0;padding=0;trace=[];torch.cuda.synchronize();t=time.perf_counter()
                        while calls<req['max_calls']:
                            if req['mode']=='wall30' and time.perf_counter()-t>=protocol['search_seconds']:break
                            if not proposals or cursor>=len(proposals):
                                cn=np.linalg.norm(bbest[:-1,2]-bbest[1:,0],axis=-1);positions=set(np.where((cn<1)|(cn>1.8))[0].tolist());positions.update(i+1 for i in list(positions))
                                ca=bbest[:,1];x,y=np.where(np.triu(np.linalg.norm(ca[:,None]-ca[None],axis=-1)<3,1));positions.update(x.tolist());positions.update(y.tolist())
                                proposals=[(i,int(token)) for i in sorted(positions) for token in pool[i] if token!=current[i]];cursor=0
                            if not proposals and req['mode']=='wall30':break
                            candidate=current.copy();is_padding=not proposals
                            if is_padding:padding+=1
                            else:i,token=proposals[cursor];cursor+=1;candidate[i]=token
                            bb=decode(candidate);calls+=1;metrics=summarize_backbone(bb)
                            gate=evaluate_candidate(candidate,bb,ids,baseline,metrics,bm,protocol['delta_angstrom'],protocol['edit_fraction_cap'])
                            candidate_key=(metrics['backbone_cn_outliers'],metrics['upstream_ca_clashes']);accepted=not is_padding and gate['eligible'] and candidate_key<key
                            if accepted:current=candidate.copy();bbest=bb.copy();best=metrics;key=candidate_key;proposals=[]
                            trace.append(dict(call=calls,accepted=accepted,padding=is_padding,gate=gate))
                        torch.cuda.synchronize();elapsed=time.perf_counter()-t
                        if req['mode']=='fixed_calls':assert calls==req['max_calls']
                        response.update(selected_fine=current.tolist(),backbone=pack(bbest),decoder_calls=calls,padding_calls=padding,search_seconds=elapsed,returned_original=bool(np.array_equal(current,ids)),gate=evaluate_candidate(current,bbest,ids,baseline,best,bm,protocol['delta_angstrom'],protocol['edit_fraction_cap']),trace=trace)
            reply(response)

if __name__=='__main__':main()
