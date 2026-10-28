"""Original sequence-only coarse diffusion, eight independent full generations."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train_native_generator import load_state_dict_from_lightning_ckpt, EsmSequenceTokenizer, tokenize_sequence
from diagnostic_io import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--chunk",type=int,required=True);a=p.parse_args()
    torch.set_num_threads(16)
    protocol=json.loads((a.run/"sampling_protocol.json").read_text())
    if not 0<=a.chunk<protocol["chunks"]:raise ValueError("Invalid chunk")
    source=Path(protocol["source"]);source.mkdir(exist_ok=True)
    model=load_state_dict_from_lightning_ckpt(Path(protocol["generator"]),device=torch.device("cpu"));model.eval().requires_grad_(False)
    head=model.net.output_heads.fine_structure_head[-1];captured=[]
    hook=head.register_forward_pre_hook(lambda m,i:captured.append(i[0].detach()))
    tokenizer=EsmSequenceTokenizer()
    for target in protocol["targets"][a.chunk::protocol["chunks"]]:
        assert target["subset"]=="confirmation"
        sequence=tokenize_sequence(target["sequence"],tokenizer)[None];n=target["length"]
        # Reference is stored for later evaluation only; never provided to the model.
        with np.load(Path(protocol["corpus"])/target["backbone"]) as z:ref=z["backbone"][:,1].copy()
        for sample in range(protocol["samples_per_protein"]):
            name=target["id"]+"_sample"+str(sample);path=source/(name+".npz")
            if path.exists():continue
            seed=int(hashlib.sha256(("bounded-fine-ensemble-v1:"+name).encode()).hexdigest()[:8],16)
            torch.manual_seed(seed);np.random.seed(seed);captured.clear();started=time.perf_counter()
            fine,coarse=model.ddpm_sample(sequence,lengths=torch.tensor([n]),num_steps=protocol["coarse_steps"],return_coarse=True)
            elapsed=time.perf_counter()-started
            logits=F.linear(captured[-1][0,1:n+1].float(),head.weight[:4096],head.bias[:4096])
            expected=fine[0,1:n+1].numpy()
            if not np.array_equal(logits.argmax(-1).numpy(),expected):raise ValueError("Original fine argmax replay differs")
            temporary=path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary,logits=logits.numpy(),fine=expected,coarse=coarse[0,1:n+1].numpy(),reference_ca=ref)
            temporary.replace(path)
            atomic_json(source/(name+".json"),dict(protein=target["id"],sample=sample,seed=seed,generation_seconds=elapsed,policy="Unmodified original ddpm_sample five coarse steps, original fine argmax, CPU FP32"))
            print("ENSEMBLE_ORIGINAL_GENERATED",name,elapsed,flush=True)
    hook.remove()
    atomic_json(a.run/("sampling_chunk"+str(a.chunk)+".json"),dict(status="complete",chunk=a.chunk))


if __name__=="__main__":main()
