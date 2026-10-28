"""Real checkpoint sampler equivalence and equal-NFE corrector comparison."""
import argparse,hashlib,json,socket,time
from pathlib import Path
import numpy as np
import torch
from train_native_generator import load_state_dict_from_lightning_ckpt,EsmSequenceTokenizer,tokenize_sequence
from discrete_flow_sampler import mask_flow_sample
from diagnostic_io import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--chunk",type=int,default=0);p.add_argument("--preflight-only",action="store_true");a=p.parse_args();torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError("Allocate Slurm GPU")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=json.loads((a.run/"protocol.json").read_text())
    for file,digest in protocol["source_code_sha256"].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest
    f=protocol["generator"];st=Path(f["path"]).stat();assert st.st_size==f["size"] and st.st_mtime_ns==f["mtime_ns"]
    device=torch.device("cuda");load_started=time.perf_counter();model=load_state_dict_from_lightning_ckpt(Path(f["path"]),device=device);model.eval().requires_grad_(False);torch.cuda.synchronize();load_seconds=time.perf_counter()-load_started
    if not model.noise_removal or model.coarse_mask_index!=32:raise ValueError("Unexpected native interface")
    tokenizer=EsmSequenceTokenizer();counter={"coarse":0,"fine":0}
    def count_hook(module,args,kwargs,output):counter["coarse" if kwargs.get("scale",0)==0 else "fine"]+=1
    hook=model.net.register_forward_hook(count_hook,with_kwargs=True)
    def generate(seq,n,steps,arm,seed):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);counter.update(coarse=0,fine=0);lengths=torch.tensor([n],device=device);torch.cuda.synchronize();start=time.perf_counter()
        if arm["name"]=="native":fine,coarse=model.ddpm_sample(seq,lengths=lengths,num_steps=steps,return_coarse=True);trace={}
        else:fine,coarse,trace=mask_flow_sample(model,seq,lengths,steps,arm["beta"],protocol["eps"])
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        if counter!={"coarse":steps+1,"fine":1}:raise ValueError(f"Unexpected model budget: {counter}")
        ids=fine[0,1:n+1].cpu().numpy();plan=coarse[0,1:n+1].cpu().numpy()
        if np.any((ids<0)|(ids>=4096)) or np.any((plan<0)|(plan>=32)):raise ValueError("Invalid normal-residue token")
        return ids,plan,dict(seed=seed,seconds=elapsed,network_calls=dict(counter),trace=trace)
    targets=protocol["targets"][:2] if a.preflight_only else protocol["targets"][a.chunk::protocol["chunks"]]
    if a.preflight_only:
        checks=[];native=protocol["arms"][0];base=protocol["arms"][1]
        for item in targets:
            seq=tokenize_sequence(item["sequence"],tokenizer).to(device)[None]
            for steps in protocol["coarse_steps"]:
                for seed in [3107,3108]:
                    f0,c0,s0=generate(seq,item["length"],steps,native,seed);f1,c1,s1=generate(seq,item["length"],steps,base,seed)
                    if not np.array_equal(f0,f1) or not np.array_equal(c0,c1):raise RuntimeError("Native/basic-FM exact token replay failed")
                    checks.append(dict(target=item["id"],steps=steps,seed=seed,exact_fine=True,exact_coarse=True,calls=s0["network_calls"]))
                    print("MASK_FM_EXACT_REPLAY",item["id"],steps,seed,flush=True)
        item=targets[0];seq=tokenize_sequence(item["sequence"],tokenizer).to(device)[None]
        _,_,trace=generate(seq,item["length"],20,protocol["arms"][3],3107)
        if trace["trace"]["remask_events"]<=0:raise RuntimeError("Corrector did not reintroduce any masks")
        atomic_json(a.run/"preflight.json",dict(status="passed",checks=checks,corrector_trace=trace,noise_class=type(model.noise).__name__,time_conditioning=bool(model.time_conditioning),model_parameters=sum(p.numel() for p in model.parameters()),host=socket.gethostname(),gpu=torch.cuda.get_device_name(),load_seconds=load_seconds))
        hook.remove();print("DISCRETE_FLOW_PREFLIGHT_PASSED",flush=True);return
    output=a.run/"predictions";output.mkdir(exist_ok=True)
    # Untimed warmup, distinct from all scored seeds.
    item=targets[0];seq=tokenize_sequence(item["sequence"],tokenizer).to(device)[None];generate(seq,item["length"],5,protocol["arms"][0],91)
    for item in targets:
        name=item["id"];n=item["length"];seq=tokenize_sequence(item["sequence"],tokenizer).to(device)[None]
        for steps in protocol["coarse_steps"]:
            arrays={x["name"]:dict(fine=[],coarse=[],samples=[]) for x in protocol["arms"]}
            for sample in range(protocol["samples"]):
                seed=int(hashlib.sha256(("discrete-flow-v1:"+name+":"+str(sample)).encode()).hexdigest()[:8],16);arms=protocol["arms"];offset=sample%len(arms);order=arms[offset:]+arms[:offset]
                for arm in order:
                    fine,coarse,info=generate(seq,n,steps,arm,seed);arrays[arm["name"]]["fine"].append(fine);arrays[arm["name"]]["coarse"].append(coarse);arrays[arm["name"]]["samples"].append(dict(sample=sample,**info))
                for key in ["fine","coarse"]:
                    if not np.array_equal(arrays["native"][key][-1],arrays["base_fm"][key][-1]):raise RuntimeError("Full-run native/basic replay failure")
            for arm,values in arrays.items():
                stem=f"{name}_steps{steps}_{arm}";np.savez_compressed(output/(stem+".npz"),fine=np.stack(values["fine"]),coarse=np.stack(values["coarse"]))
                atomic_json(output/(stem+".json"),dict(target=name,steps=steps,arm=arm,length=n,samples=values["samples"],host=socket.gethostname(),gpu=torch.cuda.get_device_name(),precision="FP32; TF32 disabled; all parameters frozen",model_load_seconds=load_seconds))
            print("DISCRETE_FLOW_TARGET_BUDGET_COMPLETE",name,steps,flush=True)
    hook.remove()
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise RuntimeError("Model freeze violated")
    atomic_json(a.run/("sampling_chunk"+str(a.chunk)+".json"),dict(status="complete",targets=[x["id"] for x in targets],native_base_exact_replay=True))


if __name__=="__main__":main()
