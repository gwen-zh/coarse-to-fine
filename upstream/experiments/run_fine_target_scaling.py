"""32 fixed training targets plus reused AA-only development evaluation.

Reuse frozen training code through process-local configuration, without editing
any completed experiment source. No independent test coordinates are opened.
"""
import argparse,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from diagnostic_io import atomic_json
from fine_geometry_flow import FixedCoarseFineFlow,fine_flow_loss
from hierarchical_latent_contract import pool_ca
from hierarchical_pipeline_io import sha
from joint_sequence_predictor import encode_aa,AA_ALPHABET
from multisequence_flow_data import check_case
from run_multisequence_ensemble_flow import evaluate as development_evaluate
from sequence_feature_ensemble_flow import SequenceFeatureEnsembleFlow
import run_fine_few_targets as engine
from target_scaling_flow import updates,event,learning_rate

RUN=Path("artifacts/fine_target_scaling_v1")
PREVIOUS=Path("artifacts/fine_few_targets_v1")
BASE=Path("artifacts/generator_curriculum_v1")
ARMS=["fixed_total","matched_presentations"]


def verify(p):
    for path,digest in p["source_sha256"].items():
        if sha(path)!=digest:raise RuntimeError("Changed scaling input: "+path)


def configure():
    engine.RUN=RUN;engine.ARMS=ARMS
    engine.updates=updates;engine.event=event;engine.learning_rate=learning_rate


def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/"protocol.json").exists():raise RuntimeError("Scaling plan already frozen")
    parent=json.loads((PREVIOUS/"protocol.json").read_text());verify(parent)
    result=json.loads((PREVIOUS/"assessment.json").read_text())
    if result["status"]!="complete_eight_training_targets_fit_near_native_reconstruction":raise RuntimeError("Eight-target result missing")
    groups=torch.load(BASE/"labels.pt",map_location="cpu",weights_only=True)
    index={c["id"]:c for c in groups["PDB_train"]}
    first=parent["case_ids"]
    other=sorted((c for c in groups["PDB_train"] if c["id"] not in first),
        key=lambda c:hashlib.sha256(("fine-32-training-v1:"+c["id"]).encode()).hexdigest())[:24]
    cases=[index[name] for name in first]+other
    if len(cases)!=32 or len(set(c["id"] for c in cases))!=32:raise RuntimeError("Expected 32 nested targets")
    for c in cases:check_case(c,"train")
    torch.save(cases,RUN/"cases.pt")
    p=copy.deepcopy(parent)
    p.update(cases=[c["id"] for c in cases],case_ids=[c["id"] for c in cases],samples=4,development_samples=64,
        training="Fresh fine branches from the same original parent checkpoints; 32 fixed static PDB training structures, native coarse. One target per four-noise minibatch, cycling all 32 in fixed order. Structural maps and normalization are frozen.",
        budgets={"fixed_total":dict(updates=2048,examples_per_target=256),
                 "matched_presentations":dict(updates=8192,examples_per_target=1024)},
        selection="Previous eight preselected PDB training cases plus 24 other PDB training cases ranked by fixed SHA256 of fine-32-training-v1:case_id. No selection by fit or development metrics.",
        learning_rate="Fixed-total arm uses the same LR at every global step as the eight-target short arm. Matched-presentations arm maps each target's local update to the eight-target short LR sequence; the first eight common targets get the exact same per-target LR/noise events and 1024 examples, with 24 extra targets interleaved.",
        pairing="Per-case Gaussian/time seeds are inherited unchanged from single/eight-target training. Short/long 32-target arms share an example/noise prefix but differ in LR progression and exposure count. This is not a duration-only or matched-compute comparison.",
        evaluation="Four retained Gaussian draws per TRAINING structure, fine Euler16/64; same draw prefix as earlier eight-sample probes. Separately, 64 retained raw samples on each of the two reused development sequences, using a frozen parent coarse model and this adapted fine model. Report full AA-only and native-coarse diagnostic outlets, with the original parent as reference.",
        primary="Report all 32 training targets and both reused development sequences. Separate native-coarse fine diagnostics from full AA-only generation; use the same parent coarse model, reference frames, sample seeds and histogram bins for development comparisons.",
        scope="32-target trainability and exploratory performance on two repeatedly used development sequences. Native-coarse training fits are not deployed ensembles. ESM3 pretraining exposure is unknown. No independent test or R3 access.",
        stop_rule="Exactly two budgets, two seeds, fixed endpoints; no checkpoint selection, rejection, test/R3 opening, automatic extension or model promotion.")
    codec=engine.load_codec(p,"centers3")
    for c in cases:
        coarse,fine=codec.encode(c["target"])
        torch.testing.assert_close(coarse,c["coarse"],atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(fine,c["fine_continuous"],atol=1e-5,rtol=1e-5)
    model=engine.make_model(p,cases,2026)
    for case in [cases[0],cases[-1]]:
        model.zero_grad(set_to_none=True)
        loss,_=fine_flow_loss(SimpleNamespace(fine_model=model),encode_aa(case["sequence"]),
            case["coarse"],case["fine_continuous"],torch.Generator().manual_seed(424))
        if not torch.isfinite(loss):raise RuntimeError("Nonfinite scaling integration")
        loss.backward();g=model.fine_head[-1].weight.grad
        if g is None or not torch.isfinite(g).all() or g.norm()<=0:raise RuntimeError("Missing scaling gradient")
    atomic_json(RUN/"integration_check.json",dict(status="passed",native_training_labels_checked=32,
        feature_bank_entries=len(model.feature_bank),real_checkpoint_forward_backward_cases=2))
    sources=[Path(__file__),Path("upstream/experiments/target_scaling_flow.py"),
        Path("upstream/tests/test_target_scaling_flow.py"),Path("upstream/scripts/fine_target_scaling.sbatch"),
        RUN/"cases.pt",RUN/"integration_check.json",PREVIOUS/"protocol.json",PREVIOUS/"report.json",PREVIOUS/"assessment.json",BASE/"labels.pt"]
    p["source_sha256"].update({str(v):sha(v) for v in sources});atomic_json(RUN/"protocol.json",p)
    print("FINE_32_TARGETS_READY",p["budgets"],flush=True)


@torch.no_grad()
def development(p,arm,seed):
    out=RUN/arm/f"seed{seed}";destination=out/"development";destination.mkdir(exist_ok=True)
    if (out/"development_complete.json").exists():return
    cases=torch.load(RUN/"cases.pt",map_location="cpu",weights_only=True)
    fine=engine.make_model(p,cases,seed)
    saved=torch.load(out/"last.pt",map_location="cpu",weights_only=False)
    if saved["updates"]!=updates(arm) or saved["protocol_sha256"]!=sha(RUN/"protocol.json"):
        raise RuntimeError("Wrong fixed scaling checkpoint")
    fine.load_state_dict(saved["model"])
    labels=torch.load(BASE/"labels.pt",map_location="cpu",weights_only=True)
    groups=dict(validation=labels["validation"],train={},reused_diagnostic={})
    features=json.loads(Path(p["feature_index"]).read_text());bank=dict(fine.feature_bank)
    for sequence_cases in groups["validation"].values():
        sequence=sequence_cases[0]["sequence"];aa=encode_aa(sequence)
        shuffled=aa[torch.randperm(len(aa),generator=torch.Generator().manual_seed(719))]
        for seq in [sequence,"".join(AA_ALPHABET[i] for i in shuffled.tolist())]:
            row=features[seq]
            if sha(row["path"])!=row["sha256"]:raise RuntimeError("Changed development AA features")
            with np.load(row["path"],allow_pickle=False) as z:bank[seq]=torch.from_numpy(z["h"].copy()).float()
    fine.attach_feature_bank(bank)
    coarse=SequenceFeatureEnsembleFlow(**p["normalization"],**p["model"],coarse_dim=3,
        feature_mode="ESM3_context",feature_dim=p["feature_dimension"])
    coarse.load_state_dict(torch.load(p["checkpoints"][str(seed)],map_location="cpu",weights_only=False)["model"])
    coarse.feature_bank=fine.feature_bank
    model=FixedCoarseFineFlow(coarse,fine).eval();codec=engine.load_codec(p,"centers3")
    ep=dict(p,samples=p["development_samples"],primary=p["scope"])
    development_evaluate(ep,model,codec,groups,destination,seed)
    maximum=0.
    for name in groups["validation"]:
        with np.load(BASE/"interleaved"/f"seed{seed}"/name/"full.npz") as z:before=z["backbone"].copy()
        with np.load(destination/name/"full.npz") as z:after=z["backbone"].copy()
        if before.shape!=after.shape:raise RuntimeError("Development sample counts changed")
        for a,b in zip(before,after):
            ac=pool_ca(torch.from_numpy(a));bc=pool_ca(torch.from_numpy(b));maximum=max(maximum,float((ac-bc).abs().max()))
            torch.testing.assert_close(ac,bc,atol=2e-5,rtol=2e-5)
    verify(p);atomic_json(out/"development_complete.json",dict(status="complete",raw_samples_per_protein=64,
        parent_coarse_sampler_preserved=True,maximum_coarse_center_roundoff_A=maximum,
        independent_test_claim=False,test_coordinates_opened=False,R3_opened=False))


def report(p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    training=[];trials=[];macro=[]
    for seed in p["seeds"]:
        trials.append(dict(arm="parent",seed=seed,evaluation=json.loads((BASE/"interleaved"/f"seed{seed}"/"evaluation.json").read_text())))
        for arm in ARMS:
            out=RUN/arm/f"seed{seed}"
            done=json.loads((out/"complete.json").read_text());dev=json.loads((out/"development_complete.json").read_text())
            if not done["codec_and_normalization_frozen"] or not dev["parent_coarse_sampler_preserved"]:raise RuntimeError("Frozen component mismatch")
            trace=json.loads((out/"trace.json").read_text());counts={name:0 for name in p["case_ids"]}
            if len(trace)!=updates(arm):raise RuntimeError("Incomplete scaling trace")
            for row in trace:counts[row["case_id"]]+=len(row["noise_counters"])
            if set(counts.values())!={p["budgets"][arm]["examples_per_target"]}:raise RuntimeError("Wrong exposure counts")
            phases={}
            for phase in ["initial","final"]:
                rows=json.loads((out/(phase+".json")).read_text())["rows"]
                vals=[next(v for v in row["evaluation"]["rows"] if v["nfe"]==16) for row in rows]
                if len(vals)!=32:raise RuntimeError("Missing training target")
                phases[phase]=dict(ca_rmsd=float(np.mean([v["metrics"]["ca_rmsd"] for v in vals])),
                    cn_rate=float(np.mean([v["metrics"]["cn_rate"] for v in vals])),
                    teacher_mse=float(np.mean([v["evaluation"]["mean_teacher_velocity_mse"] for v in rows])))
            training.append(dict(arm=arm,seed=seed,presentations=counts,**phases))
            trials.append(dict(arm=arm,seed=seed,evaluation=json.loads((out/"development"/"evaluation.json").read_text())))
    for t in trials:
        for outlet in ["full","oracle_coarse"]:
            vals=[v for v in t["evaluation"]["rows"] if v["split"]=="validation" and v["outlet"]==outlet]
            if len(vals)!=2:raise RuntimeError("Expected two reused development proteins")
            stats={key:float(np.mean([v[key] for v in vals])) for key in [
                "mean_sample_nearest_reference_ca_rmsd_A","mean_reference_nearest_sample_ca_rmsd_A","js_pairwise_distance_divergence_nats"]}
            stats["cn_rate"]=float(np.mean([v["geometry"]["cn_rate"] for v in vals]))
            stats["spread_ratio"]=float(np.mean([v["mean_pairwise_ca_rmsd_A"]/v["reference_mean_pairwise_ca_rmsd_A"] for v in vals]))
            macro.append(dict(arm=t["arm"],seed=t["seed"],outlet=outlet,**stats))
    verify(p);atomic_json(RUN/"report.json",dict(status="complete",training=training,development=trials,
        macro_per_seed=macro,scope=p["scope"],model_default_replaced=False,independent_test_claim=False,
        test_coordinates_opened=False,R3_opened=False))
    (RUN/"result_summary.txt").write_text("\n".join([p["scope"],p["pairing"],"TRAINING"]+[json.dumps(v) for v in training]+["DEVELOPMENT"]+[json.dumps(v) for v in macro])+"\n")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,"svg.fonttype":"none"})
    fig,axes=plt.subplots(2,3,figsize=(12,7.5),constrained_layout=True)
    names=["parent"]+ARMS;labels=["Parent","2,048 updates","8,192 updates"]
    for seed,color,marker in [(2026,"#24679e","o"),(2027,"#c27b24","^")]:
        train=[next(v for v in training if v["seed"]==seed and v["arm"]==arm) for arm in ARMS]
        axes[0,0].plot([0,1],[v["final"]["ca_rmsd"] for v in train],marker=marker,color=color,label=f"Seed {seed}")
        axes[1,0].plot([0,1],[100*v["final"]["cn_rate"] for v in train],marker=marker,color=color)
        for column,outlet in [(1,"oracle_coarse"),(2,"full")]:
            rows=[next(v for v in macro if v["seed"]==seed and v["arm"]==arm and v["outlet"]==outlet) for arm in names]
            axes[0,column].plot(range(3),[v["mean_sample_nearest_reference_ca_rmsd_A"] for v in rows],marker=marker,color=color)
            axes[1,column].plot(range(3),[100*v["cn_rate"] for v in rows],marker=marker,color=color)
    for col,title in enumerate(["Training: native coarse","Development: native coarse","Development: AA-only"]):
        axes[0,col].set_title(title);axes[0,col].set_ylabel("CA error (Å)");axes[1,col].set_ylabel("C–N abnormalities (%)")
        for ax in axes[:,col]:ax.set_xticks(range(2) if col==0 else range(3),labels[1:] if col==0 else labels);ax.grid(axis="y",alpha=.18)
    axes[0,0].legend(frameon=False);fig.suptitle("32-target fine training: fit versus reused development performance")
    folder=Path("paper/figures/fine_target_scaling_v1");folder.mkdir(parents=True,exist_ok=True)
    for ext in ["pdf","png","svg"]:fig.savefig(folder/("Fine_32_Targets_Development."+ext),dpi=180)
    plt.close(fig);print("FINE_32_TARGETS_COMPLETE",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["prepare","train","report"],required=True)
    parser.add_argument("--arm",choices=ARMS);parser.add_argument("--seed",type=int,default=2026);args=parser.parse_args()
    torch.set_num_threads(2);configure()
    if args.stage=="prepare":prepare()
    else:
        p=json.loads((RUN/"protocol.json").read_text());verify(p)
        if args.stage=="report":report(p)
        else:
            if args.arm not in ARMS or args.seed not in p["seeds"]:raise RuntimeError("Undeclared scaling trial")
            out=RUN/args.arm/f"seed{args.seed}"
            if not (out/"complete.json").exists():engine.train(p,args.arm,args.seed)
            development(p,args.arm,args.seed)
