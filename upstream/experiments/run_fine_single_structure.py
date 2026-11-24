"""Training-only memorization check: one fixed structure per fine model.

A point target should be learnable before a diverse conditional distribution.
Success here is a debugging result, never evidence for ensemble quality.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from compare_two_level_ae_author import metrics
from diagnostic_io import atomic_json
from fine_geometry_flow import fine_flow_loss
from hierarchical_ensemble_flow import component_seed
from hierarchical_latent_contract import pool_ca
from hierarchical_pipeline_io import sha
from joint_sequence_predictor import encode_aa
from multisequence_flow_data import check_case
from run_sequence_multiscale_ensemble import load_codec
from sequence_feature_ensemble_flow import SequenceFeatureEnsembleFlow
from structural_metrics import aligned_rmsd

RUN=Path("artifacts/fine_single_structure_v1")
PARENT=Path("artifacts/generator_curriculum_v1")
FINE_TRIAL=Path("artifacts/fine_geometry_flow_v1")


def key_seed(seed,case_id,phase,index):
    return int(hashlib.sha256(f"fine-single-structure-v1:{seed}:{case_id}:{phase}:{index}".encode()).hexdigest()[:8],16)


def verify(p):
    for name,digest in p["source_sha256"].items():
        if sha(name)!=digest:raise RuntimeError("Changed single-structure input: "+name)


def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/"protocol.json").exists():raise RuntimeError("Single-structure plan already frozen")
    parent=json.loads((PARENT/"protocol.json").read_text());verify(parent)
    result=json.loads((FINE_TRIAL/"assessment.json").read_text())
    if result["status"]!="complete_no_material_generation_gain_at_fixed_weight_and_budget":
        raise RuntimeError("Preceding result not assessed")
    groups=torch.load(PARENT/"labels.pt",map_location="cpu",weights_only=True)
    chosen=[next(c for c in groups["PDB_train"] if c["id"]==parent["pretrain_probe_ids"][0]),
            groups["train"]["1b2s_E"][0]]
    for c in chosen:check_case(c,"train")
    torch.save(dict(PDB_single=chosen[0],MD_single_frame=chosen[1]),RUN/"cases.pt")
    p=copy.deepcopy(parent)
    p.update(cases=["PDB_single","MD_single_frame"],case_ids=[c["id"] for c in chosen],
        updates=2048,batch_size=4,lr=3e-4,min_lr=3e-5,samples=8,
        checkpoints={str(s):str(PARENT/"interleaved"/f"seed{s}"/"last.pt") for s in [2026,2027]},
        objective="Pure fine flow MSE; each model sees only one fixed AA/native-coarse/native-fine target with fresh Gaussian/time draws. No geometric loss, coarse generation, quantization or decoder updates.",
        evaluation="Initial and final fixed checkpoints only; eight paired Gaussian draws at fine Euler16 and Euler64, plus a fixed 5-time by 8-noise teacher-velocity check with separate seeds. New noise is not a biological validation split. Report all trials, no sample selection.",
        interpretation="This intentionally collapses a single-label conditional distribution. It is a trainability check, not an ensemble model or a diversity-preservation result. Supplied native coarse makes it an oracle diagnostic. MD case uses one frozen R1 training frame, not a trajectory target.",
        stop_rule="Exactly 2048 updates, no extension, best epoch, hyperparameter sweep, biological validation/test or R3 coordinate evaluation.",
        scope="Training-only engineering diagnostic on one preselected PDB structure and one preselected MD R1 frame, two initial parent seeds.")
    codec=load_codec(p,"centers3")
    for c in chosen:
        coarse,fine=codec.encode(c["target"])
        torch.testing.assert_close(coarse,c["coarse"],atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(fine,c["fine_continuous"],atol=1e-5,rtol=1e-5)
    model=make_model(p,chosen[0],2026)
    loss,_=fine_flow_loss(SimpleNamespace(fine_model=model),encode_aa(chosen[0]["sequence"]),
        chosen[0]["coarse"],chosen[0]["fine_continuous"],torch.Generator().manual_seed(7251))
    if not torch.isfinite(loss):raise RuntimeError("Nonfinite single-case integration")
    loss.backward()
    gradient=model.fine_head[-1].weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or gradient.norm()<=0:
        raise RuntimeError("No single-case fine gradient")
    atomic_json(RUN/"integration_check.json",dict(status="passed",native_labels_replayed_for_both_cases=True,
        real_checkpoint_and_single_sequence_feature_mapping_verified=True,
        gradient_norm=float(gradient.norm()),optimizer_updates=0))
    sources=[Path(__file__),Path("upstream/scripts/fine_single_structure.sbatch"),
        Path("upstream/experiments/fine_geometry_flow.py"),RUN/"cases.pt",RUN/"integration_check.json",PARENT/"protocol.json",
        FINE_TRIAL/"assessment.json",FINE_TRIAL/"checkpoint_update_audit.json"]
    sources += [Path(v) for v in p["checkpoints"].values()]
    p["source_sha256"].update({str(path):sha(path) for path in sources})
    atomic_json(RUN/"protocol.json",p)
    print("FINE_SINGLE_STRUCTURE_READY",p["case_ids"],flush=True)


def make_model(p,case,seed):
    saved=torch.load(p["checkpoints"][str(seed)],map_location="cpu",weights_only=False)
    if saved["updates"]!=12288 or saved["protocol_sha256"]!=sha(PARENT/"protocol.json"):
        raise RuntimeError("Wrong parent checkpoint")
    model=SequenceFeatureEnsembleFlow(**p["normalization"],**p["model"],coarse_dim=3,
        feature_mode="ESM3_context",feature_dim=p["feature_dimension"])
    model.load_state_dict(saved["model"]);model.coarse_head.requires_grad_(False)
    row=json.loads(Path(p["feature_index"]).read_text())[case["sequence"]]
    if sha(row["path"])!=row["sha256"]:raise RuntimeError("Changed fixed sequence features")
    with np.load(row["path"],allow_pickle=False) as z:h=torch.from_numpy(z["h"].copy()).float()
    model.attach_feature_bank({case["sequence"]:h})
    return model


@torch.no_grad()
def evaluate(p,model,codec,case,seed,folder):
    model.eval();aa=encode_aa(case["sequence"]);c=case["coarse"];f=case["fine_continuous"]
    native=codec.decode(c,f);rows=[]
    for nfe in [16,64]:
        values=[];all_bb=[];latent=[]
        for index in range(p["samples"]):
            key=component_seed(key_seed(seed,case["id"],"evaluation",index),"fine")
            prediction=model.sample_fine(aa,c,key,nfe);bb=codec.decode(c,prediction)
            torch.testing.assert_close(pool_ca(bb),c,atol=2e-5,rtol=2e-5)
            values.append(metrics(bb.numpy(),case["target"].numpy()));all_bb.append(bb.numpy())
            latent.append(float((prediction-f).square().mean().sqrt()))
        arrays=np.stack(all_bb)
        spread=[aligned_rmsd(a[:,1],b[:,1]) for i,a in enumerate(arrays) for b in arrays[i+1:]]
        rows.append(dict(nfe=nfe,metrics={key:float(np.mean([v[key] for v in values])) for key in values[0]},
            mean_fine_latent_rmse=float(np.mean(latent)),mean_pairwise_ca_rmsd_A=float(np.mean(spread))))
        np.savez_compressed(folder/f"fine{nfe}.npz",backbone=arrays)
    teacher=[];normal=model.normalize_fine(f);adapter=SimpleNamespace(fine_model=model)
    for t in [.05,.25,.5,.75,.95]:
        scores=[]
        for index in range(8):
            rng=torch.Generator().manual_seed(key_seed(seed,case["id"],"teacher_evaluation",index))
            noise=torch.randn(f.shape,generator=rng);state=(1-t)*noise+t*normal
            velocity=model.fine_velocity(aa,state,t,model.normalize_coarse(c))
            scores.append(float((velocity-(normal-noise)).square().mean()))
        teacher.append(dict(time=t,velocity_mse=float(np.mean(scores))))
    result=dict(status="complete",rows=rows,teacher=teacher,
        mean_teacher_velocity_mse=float(np.mean([r["velocity_mse"] for r in teacher])),
        native_reconstruction=metrics(native.numpy(),case["target"].numpy()))
    atomic_json(folder/"evaluation.json",result)
    return result


def train(p,name,seed):
    case=torch.load(RUN/"cases.pt",map_location="cpu",weights_only=True)[name];check_case(case,"train")
    model=make_model(p,case,seed);codec=load_codec(p,"centers3")
    params=[v for v in model.parameters() if v.requires_grad]
    buffers={k:v.clone() for k,v in model.named_buffers()}
    codec_state={k:v.clone() for k,v in codec.state_dict().items()}
    out=RUN/name/f"seed{seed}";out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists() or (out/"progress.json").exists():raise RuntimeError("Fresh diagnostic only")
    for phase in ["initial","final"]:(out/phase).mkdir(exist_ok=True)
    initial=evaluate(p,model,codec,case,seed,out/"initial")
    aa=encode_aa(case["sequence"]);adapter=SimpleNamespace(fine_model=model)
    opt=torch.optim.AdamW(params,lr=p["lr"],weight_decay=p["weight_decay"])
    trace=[];start=time.perf_counter()
    for step in range(p["updates"]):
        model.train();opt.zero_grad(set_to_none=True)
        rate=p["min_lr"]+.5*(p["lr"]-p["min_lr"])*(1+math.cos(math.pi*step/(p["updates"]-1)))
        for group in opt.param_groups:group["lr"]=rate
        value=0.
        for slot in range(p["batch_size"]):
            rng=torch.Generator().manual_seed(key_seed(seed,case["id"],"training",step*4+slot))
            loss,_=fine_flow_loss(adapter,aa,case["coarse"],case["fine_continuous"],rng)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite memorization loss")
            (loss/p["batch_size"]).backward();value+=float(loss.detach())/p["batch_size"]
        grad=torch.nn.utils.clip_grad_norm_(params,p["gradient_clip"])
        if not torch.isfinite(grad):raise RuntimeError("Nonfinite memorization gradient")
        opt.step();trace.append(value)
        if (step+1)%128==0 or step+1==p["updates"]:
            progress=dict(case=name,seed=seed,updates=step+1,total=p["updates"],loss=value,seconds=time.perf_counter()-start)
            atomic_json(out/"progress.json",progress);print("SINGLE_FINE_TRAIN",json.dumps(progress),flush=True)
    torch.save(dict(model=model.state_dict(),updates=p["updates"],protocol_sha256=sha(RUN/"protocol.json"),
        case_id=case["id"],seed=seed),out/"last.pt")
    final=evaluate(p,model,codec,case,seed,out/"final")
    for k,v in model.named_buffers():torch.testing.assert_close(v,buffers[k],atol=0,rtol=0)
    for k,v in codec.state_dict().items():torch.testing.assert_close(v,codec_state[k],atol=0,rtol=0)
    verify(p)
    atomic_json(out/"complete.json",dict(status="complete",case=name,case_id=case["id"],seed=seed,
        initial=initial,final=final,trace=trace,seconds=time.perf_counter()-start,
        codec_and_normalization_frozen=True,training_only=True,no_ensemble_claim=True,
        test_coordinates_opened=False,R3_opened=False))


def report(p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=[]
    for name in p["cases"]:
        for seed in p["seeds"]:
            row=json.loads((RUN/name/f"seed{seed}"/"complete.json").read_text())
            if len(row["trace"])!=p["updates"]:raise RuntimeError("Incomplete memorization")
            rows.append(row)
    verify(p)
    atomic_json(RUN/"report.json",dict(status="complete",rows=rows,scope=p["scope"],
        interpretation=p["interpretation"],training_only=True,no_ensemble_claim=True,
        model_default_replaced=False,test_coordinates_opened=False,R3_opened=False))
    lines=[p["scope"],p["interpretation"],""]
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),constrained_layout=True)
    for i,r in enumerate(rows):
        for phase,marker in [("initial","o"),("final","^")]:
            v=next(x for x in r[phase]["rows"] if x["nfe"]==16)
            for ax,key,scale in [(axes[0],"ca_rmsd",1.),(axes[1],"cn_rate",100.)]:
                ax.scatter(i+(-.08 if phase=="initial" else .08),scale*v["metrics"][key],marker=marker,
                    color="#777777" if phase=="initial" else "#24679e")
            axes[2].scatter(i+(-.08 if phase=="initial" else .08),r[phase]["mean_teacher_velocity_mse"],
                marker=marker,color="#777777" if phase=="initial" else "#24679e")
            lines.append(f"{r['case']} seed{r['seed']} {phase}: CA {v['metrics']['ca_rmsd']:.5f} A; CN {100*v['metrics']['cn_rate']:.4f}%; latent RMSE {v['mean_fine_latent_rmse']:.5f}; teacher MSE {r[phase]['mean_teacher_velocity_mse']:.5f}")
    labels=[f"{r['case']}\nseed{r['seed']}" for r in rows]
    for ax,label in zip(axes,["CA error (Å)","C–N abnormalities (%)","Fine velocity MSE (new noise)"]):
        ax.set_xticks(range(len(rows)),labels,fontsize=7);ax.set_ylabel(label);ax.grid(axis="y",alpha=.2)
    fig.suptitle("Can the fine flow fit one fixed training structure? Gray: initial; blue: final")
    folder=Path("paper/figures/fine_single_structure_v1");folder.mkdir(parents=True,exist_ok=True)
    for ext in ["png","pdf","svg"]:fig.savefig(folder/("Fine_Single_Structure."+ext),dpi=180)
    plt.close(fig);(RUN/"result_summary.txt").write_text("\n".join(lines)+"\n")
    print("FINE_SINGLE_STRUCTURE_COMPLETE",len(rows),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["prepare","train","report"],required=True)
    parser.add_argument("--case",choices=["PDB_single","MD_single_frame"]);parser.add_argument("--seed",type=int,default=2026)
    args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=="prepare":prepare()
    else:
        p=json.loads((RUN/"protocol.json").read_text());verify(p)
        if args.stage=="report":report(p)
        else:
            if args.case not in p["cases"] or args.seed not in p["seeds"]:raise RuntimeError("Undeclared case")
            train(p,args.case,args.seed)
