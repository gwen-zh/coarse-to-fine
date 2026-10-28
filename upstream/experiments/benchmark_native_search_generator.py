"""Warm original generation latency; decoder timing runs in another process on same node."""
import argparse,hashlib,json,time,socket
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train_native_generator import load_state_dict_from_lightning_ckpt,EsmSequenceTokenizer,tokenize_sequence
from diagnostic_io import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();torch.set_num_threads(16);r=a.run;protocol=json.loads((r/"timing_protocol.json").read_text())
    corpus=Path(protocol["corpus"]);rows={x["id"]:x for x in json.loads((corpus/"manifest.json").read_text())["targets"]}
    load=time.perf_counter();model=load_state_dict_from_lightning_ckpt(Path(protocol["generator"]),device=torch.device("cpu"));model.eval().requires_grad_(False);load=time.perf_counter()-load
    tokenizer=EsmSequenceTokenizer();head=model.net.output_heads.fine_structure_head[-1];captured=[];hook=head.register_forward_pre_hook(lambda m,i:captured.append(i[0].detach()));source=r/"inputs";source.mkdir(exist_ok=True);items=[];times=[]
    first=rows[protocol["proteins"][0]];seq=tokenize_sequence(first["sequence"],tokenizer)[None];model.ddpm_sample(seq,lengths=torch.tensor([first["length"]]),num_steps=5,return_coarse=True);captured.clear()
    for protein in protocol["proteins"]:
        row=rows[protein];seq=tokenize_sequence(row["sequence"],tokenizer)[None];n=row["length"]
        with np.load(corpus/row["backbone"]) as z:ref=z["backbone"][:,1].copy()
        for sample in range(2):
            name=protein+"_sample"+str(sample);seed=int(hashlib.sha256(("search-latency-v1:"+name).encode()).hexdigest()[:8],16);torch.manual_seed(seed);np.random.seed(seed);captured.clear();started=time.perf_counter();fine,coarse=model.ddpm_sample(seq,lengths=torch.tensor([n]),num_steps=5,return_coarse=True);elapsed=time.perf_counter()-started
            logits=F.linear(captured[-1][0,1:n+1].float(),head.weight[:4096],head.bias[:4096]);ids=fine[0,1:n+1].numpy()
            assert np.array_equal(logits.argmax(-1).numpy(),ids)
            path=source/(name+".npz");np.savez_compressed(path,logits=logits.numpy(),fine=ids,coarse=coarse[0,1:n+1].numpy(),reference_ca=ref)
            items.append(dict(target=name,protein=protein,sample=sample,input=path.name));times.append(dict(target=name,generation_seconds=elapsed));print("LATENCY_ORIGINAL_COMPLETE",name,elapsed,flush=True)
    hook.remove();atomic_json(source/"manifest.json",dict(targets=items))
    search=json.loads(Path("artifacts/development/bounded_fine_search_ensemble_30s_v1/protocol.json").read_text());search.update(source=str(source),arms=["discrete_topk"],scope="Same-node CPU16 component-latency benchmark, four fixed development proteins x2 generated draws; not an efficacy or thermodynamic benchmark",input_sha256={x["target"]:hashlib.sha256((source/x["input"]).read_bytes()).hexdigest() for x in items})
    atomic_json(r/"protocol.json",search);atomic_json(r/"generator_timing.json",dict(host=socket.gethostname(),threads=16,load_seconds=load,records=times,warmup="One untimed full original generation",scope="Generator checkpoint loading excluded from warm per-output inference; fine hidden-state capture hook retained"))


if __name__=="__main__":main()
