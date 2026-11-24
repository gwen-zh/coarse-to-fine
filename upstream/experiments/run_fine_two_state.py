"""Same-AA two-state trainability: only native coarse communicates state.

Selected training probes, not ensemble coverage or new-sequence performance.
"""
import argparse,copy,json,math,time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from diagnostic_io import atomic_json
from fine_geometry_flow import fine_flow_loss
from hierarchical_ensemble_flow import component_seed
from hierarchical_pipeline_io import sha
from joint_sequence_predictor import encode_aa
from run_fine_single_structure import make_model,evaluate,key_seed
from run_sequence_multiscale_ensemble import load_codec

RUN=Path("artifacts/fine_two_state_v1")
AUDIT=Path("artifacts/fine_coarse_condition_audit_v1")
BASE=Path("artifacts/generator_curriculum_v1")
PROTEINS=["1b2s_E","1ef1_D"]


def verify(p):
    for name,digest in p["source_sha256"].items():
        if sha(name)!=digest:raise RuntimeError("Changed two-state input: "+name)


def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/"protocol.json").exists():raise RuntimeError("Two-state plan already frozen")
    p=copy.deepcopy(json.loads((BASE/"protocol.json").read_text()));verify(p)
    audit=json.loads((AUDIT/"assessment.json").read_text())
    if audit["status"]!="complete_coarse_response_often_weak_or_misaligned_in_selected_pairs":raise RuntimeError("Missing response audit")
    pairs={r["protein"]:r for r in torch.load(AUDIT/"pairs.pt",map_location="cpu",weights_only=True)}
    chosen={name:pairs[name] for name in PROTEINS}
    for pair in chosen.values():
        if pair["split"]!="train" or pair["a"]["sequence"]!=pair["b"]["sequence"]:raise RuntimeError("Training same-AA pairs required")
        if torch.equal(pair["a"]["coarse"],pair["b"]["coarse"]):raise RuntimeError("Distinct coarse states required")
    torch.save(chosen,RUN/"pairs.pt")
    p.update(proteins=PROTEINS,updates=2048,batch_size=4,samples=8,lr=3e-4,min_lr=3e-5,
        checkpoints={str(s):str(BASE/"interleaved"/f"seed{s}"/"last.pt") for s in [2026,2027]},
        training="Fresh original-parent fine models, one protein per model. Two fixed native MD-training frames from the earlier response audit, same AA, two noise draws per state per optimizer update. Native coarse is the only input communicating the state. Pure fine flow loss; no geometric loss or decoder update.",
        evaluation="Initial and final only; both states share the same eight fine Gaussian seeds. Evaluate matched state geometry and normalized-latent nearest-state assignment, plus direction/magnitude of the predicted conditional-mean fine change. Native coarse is supplied in all evaluations. Euler16 is primary, Euler64 a solver diagnostic.",
        selection="Selected TRAIN probes 1b2s_E (previous one-frame trainability probe) and 1ef1_D (known difficult training protein); same first/last cached native frames as the response audit. This is not a representative sample or a performance test cohort.",
        scope="Two-state conditional learnability only, two selected training proteins and two parent seeds. No biological validation, independent test or R3 evaluation. Success does not establish full-trajectory ensemble or unseen-sequence quality.",
        stop_rule="Exactly 2048 updates, no best checkpoint, extension, state/sample selection, test/R3 use or model promotion.")
    codec=load_codec(p,"centers3")
    for pair in chosen.values():
        model=make_model(p,pair["a"],2026)
        for case in [pair["a"],pair["b"]]:
            c,f=codec.encode(case["target"])
            torch.testing.assert_close(c,case["coarse"],atol=1e-5,rtol=1e-5)
            torch.testing.assert_close(f,case["fine_continuous"],atol=1e-5,rtol=1e-5)
            model.zero_grad(set_to_none=True)
            loss,_=fine_flow_loss(SimpleNamespace(fine_model=model),encode_aa(case["sequence"]),c,f,torch.Generator().manual_seed(714))
            loss.backward();g=model.fine_head[-1].weight.grad
            if g is None or not torch.isfinite(g).all() or g.norm()<=0:raise RuntimeError("Two-state gradient integration failed")
    atomic_json(RUN/"integration_check.json",dict(status="passed",native_states_checked=4,real_checkpoint_forward_backward=True,optimizer_updates=0))
    sources=[Path(__file__),Path("upstream/scripts/fine_two_state.sbatch"),
        Path("upstream/experiments/run_fine_single_structure.py"),Path("upstream/experiments/fine_geometry_flow.py"),
        AUDIT/"protocol.json",AUDIT/"report.json",AUDIT/"assessment.json",RUN/"pairs.pt",RUN/"integration_check.json"]
    sources += [Path(v) for v in p["checkpoints"].values()]
    p["source_sha256"].update({str(v):sha(v) for v in sources});atomic_json(RUN/"protocol.json",p)
    print("FINE_TWO_STATE_READY",PROTEINS,flush=True)


@torch.no_grad()
def evaluate_pair(p,model,codec,pair,seed,folder):
    model.eval();rows=[];common="two-state-evaluation:"+pair["protein"]
    cases=[pair["a"],pair["b"]]
    for side,case in enumerate(cases):
        target_folder=folder/f"state{side}";target_folder.mkdir(parents=True,exist_ok=True)
        result=evaluate(p,model,codec,dict(case,id=common),seed,target_folder)
        rows.append(dict(state=side,native_case_id=case["id"],evaluation=result))
    labels=[model.normalize_fine(c["fine_continuous"]) for c in cases]
    generated=[[],[]];assignment=[];aa=encode_aa(cases[0]["sequence"])
    for index in range(p["samples"]):
        key=component_seed(key_seed(seed,common,"evaluation",index),"fine")
        for side,case in enumerate(cases):
            f=model.normalize_fine(model.sample_fine(aa,case["coarse"],key,16));generated[side].append(f)
            distances=[float((f-native).square().mean()) for native in labels]
            assignment.append(.5 if distances[0]==distances[1] else float(int(np.argmin(distances))==side))
    native_delta=labels[1]-labels[0];delta=torch.stack(generated[1]).mean(0)-torch.stack(generated[0]).mean(0)
    cosine=float((delta*native_delta).sum()/(delta.norm()*native_delta.norm())) if delta.norm()>1e-8 else None
    result=dict(status="complete",states=rows,state_assignment_accuracy=float(np.mean(assignment)),
        native_fine_change_rms=float(native_delta.square().mean().sqrt()),predicted_fine_change_rms=float(delta.square().mean().sqrt()),
        fine_change_direction_cosine=cosine)
    atomic_json(folder/"evaluation.json",result);return result


def train(p,name,seed):
    pair=torch.load(RUN/"pairs.pt",map_location="cpu",weights_only=True)[name]
    cases=[pair["a"],pair["b"]];model=make_model(p,cases[0],seed);codec=load_codec(p,"centers3")
    out=RUN/name/f"seed{seed}";out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists() or (out/"progress.json").exists():raise RuntimeError("Fresh two-state diagnostic only")
    for phase in ["initial","final"]:(out/phase).mkdir(exist_ok=True)
    initial=evaluate_pair(p,model,codec,pair,seed,out/"initial")
    frozen={k:v.clone() for k,v in codec.state_dict().items()};buffers={k:v.clone() for k,v in model.named_buffers()}
    params=[v for v in model.parameters() if v.requires_grad];opt=torch.optim.AdamW(params,lr=p["lr"],weight_decay=p["weight_decay"])
    aa=encode_aa(cases[0]["sequence"]);adapter=SimpleNamespace(fine_model=model);trace=[];start=time.perf_counter()
    for step in range(p["updates"]):
        model.train();opt.zero_grad(set_to_none=True);value=0.
        rate=p["min_lr"]+.5*(p["lr"]-p["min_lr"])*(1+math.cos(math.pi*step/(p["updates"]-1)))
        for group in opt.param_groups:group["lr"]=rate
        for slot in range(4):
            case=cases[(step+slot)%2]
            generator=torch.Generator().manual_seed(key_seed(seed,name,"two-state-training",step*4+slot))
            loss,_=fine_flow_loss(adapter,aa,case["coarse"],case["fine_continuous"],generator)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite two-state loss")
            (loss/4).backward();value+=float(loss.detach())/4
        grad=torch.nn.utils.clip_grad_norm_(params,p["gradient_clip"])
        if not torch.isfinite(grad):raise RuntimeError("Nonfinite two-state gradient")
        opt.step();trace.append(value)
        if (step+1)%128==0 or step+1==p["updates"]:
            progress=dict(protein=name,seed=seed,updates=step+1,total=p["updates"],loss=value,seconds=time.perf_counter()-start)
            atomic_json(out/"progress.json",progress);print("TWO_STATE_TRAIN",json.dumps(progress),flush=True)
    torch.save(dict(model=model.state_dict(),updates=p["updates"],protocol_sha256=sha(RUN/"protocol.json")),out/"last.pt")
    final=evaluate_pair(p,model,codec,pair,seed,out/"final")
    for k,v in codec.state_dict().items():torch.testing.assert_close(v,frozen[k],atol=0,rtol=0)
    for k,v in model.named_buffers():torch.testing.assert_close(v,buffers[k],atol=0,rtol=0)
    verify(p);atomic_json(out/"complete.json",dict(status="complete",protein=name,seed=seed,
        initial=initial,final=final,trace=trace,seconds=time.perf_counter()-start,
        codec_and_normalization_frozen=True,training_only=True,no_ensemble_claim=True,test_coordinates_opened=False,R3_opened=False))


def report(p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=[];lines=[p["scope"],p["selection"],""]
    for name in PROTEINS:
        for seed in p["seeds"]:
            row=json.loads((RUN/name/f"seed{seed}"/"complete.json").read_text())
            if len(row["trace"])!=p["updates"]:raise RuntimeError("Incomplete two-state training")
            rows.append(row)
    verify(p);atomic_json(RUN/"report.json",dict(status="complete",rows=rows,scope=p["scope"],
        training_only=True,no_ensemble_claim=True,model_default_replaced=False,test_coordinates_opened=False,R3_opened=False))
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),constrained_layout=True)
    for index,row in enumerate(rows):
        for phase,offset,color,marker in [("initial",-.08,"#777777","o"),("final",.08,"#24679e","^")]:
            d=row[phase];values=[next(v for v in r["evaluation"]["rows"] if v["nfe"]==16) for r in d["states"]]
            ca=float(np.mean([v["metrics"]["ca_rmsd"] for v in values]));cn=float(np.mean([v["metrics"]["cn_rate"] for v in values]))
            for ax,value in zip(axes,[ca,100*cn,100*d["state_assignment_accuracy"]]):ax.scatter(index+offset,value,color=color,marker=marker)
            lines.append(f"{row['protein']} seed{row['seed']} {phase}: CA {ca:.5f} A; CN {100*cn:.4f}%; correct fine state {100*d['state_assignment_accuracy']:.2f}%; delta cosine {d['fine_change_direction_cosine']}")
    labels=[f"{r['protein']}\nseed{r['seed']}" for r in rows]
    for ax,label in zip(axes,["CA error (Å)","C–N abnormalities (%)","Correct fine-state assignment (%)"]):
        ax.set_xticks(range(len(rows)),labels,fontsize=8);ax.set_ylabel(label);ax.grid(axis="y",alpha=.2)
    axes[2].axhline(50.,ls="--",color=".5",lw=1)
    fig.suptitle("Same sequence, two native coarse states: training probes only")
    folder=Path("paper/figures/fine_two_state_v1");folder.mkdir(parents=True,exist_ok=True)
    for ext in ["pdf","png","svg"]:fig.savefig(folder/("Fine_Two_State_Conditioning."+ext),dpi=180)
    plt.close(fig);(RUN/"result_summary.txt").write_text("\n".join(lines)+"\n")
    print("FINE_TWO_STATE_COMPLETE",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["prepare","train","report"],required=True)
    parser.add_argument("--protein",choices=PROTEINS);parser.add_argument("--seed",type=int,default=2026)
    args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=="prepare":prepare()
    else:
        p=json.loads((RUN/"protocol.json").read_text());verify(p)
        if args.stage=="report":report(p)
        else:
            if args.protein not in PROTEINS or args.seed not in p["seeds"]:raise RuntimeError("Undeclared two-state run")
            train(p,args.protein,args.seed)
