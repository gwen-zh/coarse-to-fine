"""Fine-flow trainability on eight fixed native-coarse PDB targets."""
import argparse,copy,json,time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from compact_training_trace import CompactTrainingTrace,restore_trace_prefix
from diagnostic_io import atomic_json
from few_target_flow import event,updates,learning_rate
from fine_geometry_flow import fine_flow_loss
from hierarchical_pipeline_io import sha,save_checkpoint,restore_checkpoint
from joint_sequence_predictor import encode_aa
from multisequence_flow_data import check_case
from run_sequence_multiscale_ensemble import load_codec
from run_fine_single_structure import make_model as initial_model,evaluate,key_seed

RUN=Path("artifacts/fine_few_targets_v1")
PARENT=Path("artifacts/generator_curriculum_v1")
SINGLE=Path("artifacts/fine_single_structure_v1")
ARMS=["fixed_total","matched_presentations"]


def verify(p):
    for name,digest in p["source_sha256"].items():
        if sha(name)!=digest:raise RuntimeError("Changed few-target input: "+name)


def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/"protocol.json").exists():raise RuntimeError("Few-target plan already frozen")
    parent=json.loads((PARENT/"protocol.json").read_text());verify(parent)
    result=json.loads((SINGLE/"assessment.json").read_text())
    if result["status"]!="complete_single_structure_fine_flow_can_fit_training_targets":raise RuntimeError("Single-target check missing")
    groups=torch.load(PARENT/"labels.pt",map_location="cpu",weights_only=True)
    case_map={c["id"]:c for c in groups["PDB_train"]};cases=[case_map[name] for name in parent["pretrain_probe_ids"]]
    if len(cases)!=8 or len(set(c["id"] for c in cases))!=8:raise RuntimeError("Eight unique fixed probes required")
    for c in cases:check_case(c,"train")
    torch.save(cases,RUN/"cases.pt")
    p=copy.deepcopy(json.loads((SINGLE/"protocol.json").read_text()))
    p.pop("updates",None)
    p.update(arms=ARMS,cases=[c["id"] for c in cases],case_ids=[c["id"] for c in cases],samples=8,
        training="Fresh fine branches from the same parent interleaved checkpoints, not the single-target fitted weights. Eight static training structures, native coarse. One target per minibatch, four fresh noises; cycle the same fixed eight-target order.",
        budgets={"fixed_total":dict(updates=2048,examples_per_target=1024),
                 "matched_presentations":dict(updates=16384,examples_per_target=8192)},
        learning_rate="Fixed-total arm uses the same LR at global step as the completed 2048-step single-target run. Matched-presentations arm uses the same LR at EACH TARGET's local exposure step; every target receives 2048 minibatches with the original cosine schedule.",
        pairing="Per-case Gaussian/time seeds are reused from the completed single-target diagnostic: case ID + local example counter. For the first PDB case, the long arm sees exactly the same 8192 noise/time events and per-case LR sequence as single-target training, with seven other targets interleaved. The two eight-target budgets share their first example/noise prefix but have different LR schedules; they cannot isolate step count from LR progression.",
        objective="Pure fine flow MSE; no geometry loss, quantization, coarse prediction or decoder updates.",
        evaluation="Initial and final endpoints only; same eight-noise Euler16/64 evaluation as the single-target diagnostic on ALL eight training structures. Native coarse supplied. No biological validation or test evaluation; single-target reference is valid only for the first common PDB target.",
        scope="Training-only multi-target learnability diagnostic, not new-sequence generation or ensemble fidelity.",
        stop_rule="Fixed two budgets and two parent seeds. No best epoch, automatic extension, weight scan, sample rejection, test/R3 access or model promotion.")
    codec=load_codec(p,"centers3")
    for c in cases:
        coarse,fine=codec.encode(c["target"])
        torch.testing.assert_close(coarse,c["coarse"],atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(fine,c["fine_continuous"],atol=1e-5,rtol=1e-5)
    model=make_model(p,cases,2026);adapter=SimpleNamespace(fine_model=model)
    for case in [cases[0],cases[-1]]:
        model.zero_grad(set_to_none=True)
        loss,_=fine_flow_loss(adapter,encode_aa(case["sequence"]),case["coarse"],case["fine_continuous"],
            torch.Generator().manual_seed(6322))
        if not torch.isfinite(loss):raise RuntimeError("Nonfinite few-target integration")
        loss.backward();gradient=model.fine_head[-1].weight.grad
        if gradient is None or not torch.isfinite(gradient).all() or gradient.norm()<=0:
            raise RuntimeError("Missing few-target gradient")
    atomic_json(RUN/"integration_check.json",dict(status="passed",native_labels_checked=8,
        frozen_AA_feature_entries=len(model.feature_bank),real_checkpoint_cases_checked=[cases[0]["id"],cases[-1]["id"]],optimizer_updates=0))
    sources=[Path(__file__),Path("upstream/experiments/few_target_flow.py"),Path("upstream/tests/test_few_target_flow.py"),
        Path("upstream/scripts/fine_few_targets.sbatch"),RUN/"cases.pt",RUN/"integration_check.json",SINGLE/"assessment.json",SINGLE/"report.json"]
    p["source_sha256"].update({str(v):sha(v) for v in sources});atomic_json(RUN/"protocol.json",p)
    print("FINE_FEW_TARGETS_READY",p["budgets"],flush=True)


def make_model(p,cases,seed):
    model=initial_model(p,cases[0],seed);bank={};index=json.loads(Path(p["feature_index"]).read_text())
    for case in cases:
        row=index[case["sequence"]]
        if sha(row["path"])!=row["sha256"]:raise RuntimeError("Changed sequence feature")
        with np.load(row["path"],allow_pickle=False) as z:bank[case["sequence"]]=torch.from_numpy(z["h"].copy()).float()
    model.attach_feature_bank(bank);return model


def evaluation(p,model,codec,cases,seed,out,phase):
    rows=[]
    for case in cases:
        folder=out/phase/case["protein"];folder.mkdir(parents=True,exist_ok=True)
        rows.append(dict(case_id=case["id"],protein=case["protein"],evaluation=evaluate(p,model,codec,case,seed,folder)))
    atomic_json(out/(phase+".json"),dict(status="complete",rows=rows));return rows


def train(p,arm,seed):
    cases=torch.load(RUN/"cases.pt",map_location="cpu",weights_only=True)
    for c in cases:check_case(c,"train")
    model=make_model(p,cases,seed);codec=load_codec(p,"centers3");adapter=SimpleNamespace(fine_model=model)
    codec_before={k:v.clone() for k,v in codec.state_dict().items()};buffers={k:v.clone() for k,v in model.named_buffers()}
    opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=3e-4,weight_decay=p["weight_decay"])
    out=RUN/arm/f"seed{seed}";out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists():raise RuntimeError("Few-target arm already complete")
    start=0;rng=np.random.default_rng(seed);digest=sha(RUN/"protocol.json")
    if (out/"last.pt").exists():
        if not (out/"initial.json").exists():raise RuntimeError("Cannot resume without the frozen initial diagnostic")
        saved=restore_checkpoint(out/"last.pt",model,opt,rng,torch.device("cpu"))
        if saved["protocol_sha256"]!=digest:raise RuntimeError("Resume source mismatch")
        start=saved["updates"]
    if not (out/"initial.json").exists():evaluation(p,model,codec,cases,seed,out,"initial")
    restore_trace_prefix(out,start);trace=[json.loads(v) for v in (out/"trace.jsonl").read_text().splitlines()]
    writer=CompactTrainingTrace(updates(arm),start);begin=time.perf_counter()
    for step in range(start,updates(arm)):
        idx,indices=event(step);case=cases[idx];aa=encode_aa(case["sequence"])
        model.train();opt.zero_grad(set_to_none=True);value=0.
        rate=learning_rate(arm,step)
        for group in opt.param_groups:group["lr"]=rate
        for counter in indices:
            generator=torch.Generator().manual_seed(key_seed(seed,case["id"],"training",counter))
            loss,_=fine_flow_loss(adapter,aa,case["coarse"],case["fine_continuous"],generator)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite few-target loss")
            (loss/4).backward();value+=float(loss.detach())/4
        norm=torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],p["gradient_clip"])
        if not torch.isfinite(norm):raise RuntimeError("Nonfinite few-target gradient")
        opt.step();trace.append(dict(update=step+1,case_id=case["id"],noise_counters=indices,lr=rate,loss=value))
        if (step+1)%128==0 or step+1==updates(arm):
            writer(out/"trace.json",trace)
            progress=dict(arm=arm,seed=seed,updates=step+1,total=updates(arm),seconds=time.perf_counter()-begin,loss=value)
            atomic_json(out/"progress.json",progress);print("FEW_FINE_TRAIN",json.dumps(progress),flush=True)
        if (step+1)%2048==0:
            save_checkpoint(out/"last.pt",model,opt,rng,updates=step+1,protocol_sha256=digest)
    final=evaluation(p,model,codec,cases,seed,out,"final")
    for k,v in model.named_buffers():torch.testing.assert_close(v,buffers[k],atol=0,rtol=0)
    for k,v in codec.state_dict().items():torch.testing.assert_close(v,codec_before[k],atol=0,rtol=0)
    verify(p)
    atomic_json(out/"complete.json",dict(status="complete",arm=arm,seed=seed,final=final,
        seconds=time.perf_counter()-begin,training_only=True,no_ensemble_claim=True,
        codec_and_normalization_frozen=True,test_coordinates_opened=False,R3_opened=False))


def report(p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=[]
    for arm in ARMS:
        for seed in p["seeds"]:
            out=RUN/arm/f"seed{seed}"
            done=json.loads((out/"complete.json").read_text());trace=json.loads((out/"trace.json").read_text())
            if len(trace)!=updates(arm):raise RuntimeError("Incomplete few-target training")
            counts={name:0 for name in p["case_ids"]}
            for r in trace:counts[r["case_id"]]+=len(r["noise_counters"])
            if set(counts.values())!={p["budgets"][arm]["examples_per_target"]}:raise RuntimeError("Wrong presentation budget")
            initial=json.loads((out/"initial.json").read_text())["rows"]
            rows.append(dict(arm=arm,seed=seed,initial=initial,final=done["final"],presentations=counts))
    verify(p);atomic_json(RUN/"report.json",dict(status="complete",rows=rows,scope=p["scope"],
        pairing=p["pairing"],training_only=True,no_ensemble_claim=True,model_default_replaced=False))
    lines=[p["scope"],p["pairing"],""]
    fig,axes=plt.subplots(1,3,figsize=(12,5),constrained_layout=True)
    for index,row in enumerate(rows):
        for phase,color,offset in [("initial","#777777",-.08),("final","#24679e",.08)]:
            vals=[next(v for v in x["evaluation"]["rows"] if v["nfe"]==16) for x in row[phase]]
            ca=float(np.mean([v["metrics"]["ca_rmsd"] for v in vals]));cn=float(np.mean([v["metrics"]["cn_rate"] for v in vals]))
            mse=float(np.mean([x["evaluation"]["mean_teacher_velocity_mse"] for x in row[phase]]))
            for ax,value in zip(axes,[ca,100*cn,mse]):ax.scatter(index+offset,value,color=color,marker="o" if phase=="initial" else "^")
            lines.append(f"{row['arm']} seed{row['seed']} {phase}: mean CA {ca:.5f} A; CN {100*cn:.4f}%; teacher MSE {mse:.6f}")
    labels=[f"{r['arm']}\nseed{r['seed']}" for r in rows]
    for ax,label in zip(axes,["CA error (Å)","C–N abnormalities (%)","Fine velocity MSE"]):
        ax.set_xticks(range(len(rows)),labels,fontsize=7);ax.set_ylabel(label);ax.grid(axis="y",alpha=.2)
    fig.suptitle("Eight fixed training targets: total updates versus per-target presentations")
    folder=Path("paper/figures/fine_few_targets_v1");folder.mkdir(parents=True,exist_ok=True)
    for ext in ["png","pdf","svg"]:fig.savefig(folder/("Fine_Eight_Targets."+ext),dpi=180)
    plt.close(fig);(RUN/"result_summary.txt").write_text("\n".join(lines)+"\n")
    print("FINE_EIGHT_TARGETS_COMPLETE",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["prepare","train","report"],required=True)
    parser.add_argument("--arm",choices=ARMS);parser.add_argument("--seed",type=int,default=2026)
    args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=="prepare":prepare()
    else:
        p=json.loads((RUN/"protocol.json").read_text());verify(p)
        if args.stage=="report":report(p)
        else:
            if args.seed not in p["seeds"] or args.arm not in p["arms"]:raise RuntimeError("Undeclared few-target run")
            train(p,args.arm,args.seed)
