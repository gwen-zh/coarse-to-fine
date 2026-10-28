"""Fixed-feature FM versus deterministic latent regression, development only."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from latent_flow_head import LatentHead,flow_path,protein_mean_mse,euler_sample
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--mode",choices=["fm","regression"],required=True);p.add_argument("--smoke",action="store_true");a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("Slurm GPU allocation required")
    torch.set_num_threads(4);torch.manual_seed(20261025);np.random.seed(20261025);device=torch.device("cuda")
    protocol=json.loads((a.run/"protocol.json").read_text());feature_dir=Path(protocol["features"]);tokens_dir=Path(protocol["tokens"])
    for key,path in [("feature_manifest_sha256",feature_dir/"manifest.json"),("token_manifest_sha256",tokens_dir/"manifest.json")]:assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol[key]
    assert hashlib.sha256(Path(protocol["calibration"]).read_bytes()).hexdigest()==protocol["calibration_sha256"]
    for file,digest in protocol["code_sha256"].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest
    with np.load(protocol["calibration"]) as z:mean=torch.from_numpy(z["mean"].astype(np.float32)).to(device);scale=float(z["centered_rms"]);codes=torch.from_numpy(z["fine_codes"].copy()).to(device)
    features={x["id"]:x for x in json.loads((feature_dir/"manifest.json").read_text())["targets"] if x["dataset"]=="corpus"}
    rows=json.loads((tokens_dir/"manifest.json").read_text())["targets"];train=sorted([x for x in rows if x["subset"]=="train"],key=lambda x:x["id"]);val=sorted([x for x in rows if x["subset"]=="validation"],key=lambda x:x["id"])
    assert len(train)==899 and len(val)==101
    if a.smoke:train=train[:8];val=val[:4]
    cache={}
    for row in train+val:
        f=features[row["id"]];assert f["subset"]==row["subset"]
        with np.load(feature_dir/f["features"]) as z:h=z["h"].copy();native=z["fine"].copy()
        with np.load(tokens_dir/row["tokens"]) as z:target=z["latent"].copy()
        assert h.shape==(row["length"],1536) and target.shape==(row["length"],128) and np.isfinite(h).all() and np.isfinite(target).all()
        cache[row["id"]]=(h,target,native)
    model=LatentHead(width=protocol["width"],layers=protocol["layers"]).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=protocol["lr"],weight_decay=protocol["weight_decay"])
    output=a.run/("smoke_"+a.mode if a.smoke else a.mode);output.mkdir(exist_ok=True)
    def batch(group):
        length=max(x["length"] for x in group);condition=np.zeros((len(group),length,1536),np.float32);target=np.zeros((len(group),length,128),np.float32);valid=np.zeros((len(group),length),bool)
        for i,row in enumerate(group):
            h,z,_=cache[row["id"]];n=len(z);condition[i,:n]=h;target[i,:n]=z;valid[i,:n]=True
        h=torch.from_numpy(condition).to(device);z=(torch.from_numpy(target).to(device)-mean)/scale;mask=torch.from_numpy(valid).to(device)
        return h,z,mask
    def val_loss():
        model.eval();losses=[]
        with torch.no_grad():
            for row in val:
                h,z,mask=batch([row])
                if a.mode=="regression":loss=protein_mean_mse(model(torch.zeros_like(z),torch.zeros(1,device=device),h,mask),z,mask).item()
                else:
                    seed=int(hashlib.sha256(("latent-flow-validation:"+row["id"]).encode()).hexdigest()[:8],16);rng=torch.Generator(device=device).manual_seed(seed);noise=torch.randn(z.shape,device=device,generator=rng);parts=[]
                    for t0 in [.1,.35,.65,.9]:
                        t=torch.tensor([t0],device=device);state,velocity=flow_path(z,noise,t);parts.append(protein_mean_mse(model(state,t,h,mask),velocity,mask).item())
                    loss=float(np.mean(parts))
                losses.append(loss)
        return float(np.mean(losses))
    initial_loss=val_loss();history=[dict(epoch=0,validation_objective=initial_loss)];best=initial_loss
    atomic_json(output/"initial.json",dict(mode=a.mode,validation_objective=initial_loss,parameters=sum(p.numel() for p in model.parameters()),train_proteins=len(train),validation_proteins=len(val),smoke=a.smoke))
    torch.save(dict(state=model.state_dict(),epoch=0,mode=a.mode,protocol=protocol),output/"best.pt")
    steps=0;started=time.perf_counter();epochs=20 if a.smoke else protocol["epochs"];rng=np.random.default_rng(protocol["seed"])
    for epoch in range(1,epochs+1):
        model.train();order=rng.permutation(len(train));losses=[]
        for start in range(0,len(train),protocol["batch_size"]):
            group=[train[i] for i in order[start:start+protocol["batch_size"]]];h,z,mask=batch(group);optimizer.zero_grad(set_to_none=True)
            if a.mode=="fm":
                t=torch.rand(len(group),device=device);noise=torch.randn_like(z);state,target=flow_path(z,noise,t);prediction=model(state,t,h,mask)
            else:target=z;prediction=model(torch.zeros_like(z),torch.zeros(len(group),device=device),h,mask)
            loss=protein_mean_mse(prediction,target,mask)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite training objective")
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm) or norm<=0:raise RuntimeError("Nonfinite or zero gradient")
            optimizer.step();steps+=1;losses.append(loss.item())
        value=val_loss();row=dict(epoch=epoch,steps=steps,train_objective=float(np.mean(losses)),validation_objective=value,elapsed_seconds=time.perf_counter()-started);history.append(row);atomic_json(output/"history.json",history)
        if value<best:best=value;torch.save(dict(state=model.state_dict(),epoch=epoch,mode=a.mode,protocol=protocol),output/"best.pt")
        print("LATENT_HEAD_EPOCH",a.mode,json.dumps(row),flush=True)
    if a.smoke:
        h,z,mask=batch([val[0]]);model.eval();sample=euler_sample(model,torch.randn_like(z),h,mask,steps=16)
        if not torch.isfinite(sample).all():raise RuntimeError("Nonfinite ODE sample")
        atomic_json(output/"summary.json",dict(status="smoke_passed",mode=a.mode,steps=steps,initial_validation=initial_loss,final_validation=history[-1]["validation_objective"]))
        print("LATENT_FLOW_SMOKE_PASSED",flush=True);return
    best_checkpoint=torch.load(output/"best.pt",map_location=device,weights_only=False);model.load_state_dict(best_checkpoint["state"]);model.eval();prediction_dir=output/"predictions";prediction_dir.mkdir(exist_ok=True)
    selected={x["id"]:x for x in val}
    for name in protocol["geometry_targets"]:
        row=selected[name];h,z,mask=batch([row]);sampled=[]
        for sample_index in range(2 if a.mode=="fm" else 1):
            seed=int(hashlib.sha256(("latent-flow-sample:"+name+":"+str(sample_index)).encode()).hexdigest()[:8],16)
            with torch.no_grad():
                if a.mode=="fm":
                    generator=torch.Generator(device=device).manual_seed(seed);start=torch.randn(z.shape,device=device,generator=generator);result=euler_sample(model,start,h,mask,steps=protocol["ode_steps"])
                else:result=model(torch.zeros_like(z),torch.zeros(1,device=device),h,mask)
            if not torch.isfinite(result).all():raise RuntimeError("Nonfinite predicted latent")
            sampled.append((result[0,:row["length"]]*scale+mean).detach().cpu().numpy())
        sampled=np.stack(sampled);np.savez_compressed(prediction_dir/(name+".npz"),latent=sampled,native_fine=cache[name][2])
    atomic_json(output/"summary.json",dict(status="complete",mode=a.mode,train_proteins=len(train),validation_proteins=len(val),steps=steps,selected_epoch=best_checkpoint["epoch"],initial_validation=initial_loss,best_validation=best,training_seconds=time.perf_counter()-started,parameters=sum(p.numel() for p in model.parameters()),scope=protocol["scope"]))


if __name__=="__main__":main()
