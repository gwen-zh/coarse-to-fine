"""Bounded matched geometric correction on the immutable parent coarse flow."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from coarse_relative_geometry import CoarseRelativeGeometry, GeometryCorrectedEnsemble, update
from coarse_full_cohort_evaluation import evaluate, summarize_rows
from coarse_full_cohort_flow import batch_rng, identity_record, sample_training_batch
from coarse_coverage_mechanism import bridge_endpoint
from coarse_generation_diagnostic import weighted_rmsd_matrix
from compact_training_trace import CompactTrainingTrace, restore_trace_prefix
from diagnostic_io import atomic_json
from hierarchical_ensemble_flow import component_seed
from hierarchical_pipeline_io import restore_checkpoint, save_checkpoint, sha
from joint_sequence_predictor import encode_aa
from run_coarse_coverage_mechanism import diagnostic_key
from run_coarse_few_targets import make_model as parent_model
from run_coarse_full_cohort import model_cases
from run_coarse_generation_diagnostic import project_centers
from run_fine_geometry_flow import state_digest
from run_multiframe_early_fine import attach_development_features
from run_multisequence_ensemble_flow import sample_seed
from run_sequence_multiscale_ensemble import load_codec


ROOT=Path("artifacts/coarse_relative_geometry_v1")
MECHANISM=Path("artifacts/coarse_coverage_mechanism_v1")
PREVIOUS=Path("artifacts/coarse_coverage_repair_v1")
FULL=Path("artifacts/coarse_full_cohort_v1")
FINE=Path("artifacts/multiframe_coarse_context_v1")
LABELS=Path("artifacts/multiframe_early_fine_v1")
ARMS=["cartesian","radial"]


def load(p):return json.loads(Path(p).read_text())


def verify(p):
    for name,digest in p["source_sha256"].items():
        if sha(name)!=digest:raise RuntimeError("Changed geometric-correction input: "+name)


def make_model(p,cases,seed,arm):
    original=parent_model(p,cases,seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+62461)
        coarse=CoarseRelativeGeometry(original.coarse_model,arm,hidden=p["adapter_hidden"])
    return GeometryCorrectedEnsemble(coarse,original.fine_model)


def frozen(model,codec,expected):
    actual=[state_digest(model.coarse_model.base),state_digest(model.fine_model),state_digest(codec)]
    if actual!=expected:raise RuntimeError("Parent coarse, fine or structure maps changed")
    if any(v.grad is not None for v in model.coarse_model.base.parameters()) or any(v.grad is not None for v in model.fine_model.parameters()):
        raise RuntimeError("A frozen generator received a gradient")


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/"protocol.json").exists():raise RuntimeError("Fresh correction namespace required")
    if load(MECHANISM/"assessment.json")["status"]!="complete_reviewed_late_flow_contraction_and_unseen_sequence_denoising_degradation_no_unique_loss_attribution":
        raise RuntimeError("Review the read-only mechanism study first")
    old=load(PREVIOUS/"protocol.json")
    keys=["normalization","model","tokenizer","fine_checkpoint","feature_index","all_feature_index",
          "feature_dimension","checkpoints","ca_step_cap_A","feature_cost"]
    p={k:old[k] for k in keys}
    p.update(seeds=[2026,2027],arms=ARMS,updates=8192,adapter_hidden=64,correction_penalty=.01,
        batch_size=4,lr=3e-4,min_lr=3e-5,weight_decay=1e-4,gradient_clip=1.,checkpoint_every=1024,progress_every=256,
        samples=64,train_diagnostic_samples=64,coarse_nfe=16,fine_nfe=16,bridge_times=[.1,.5,.9],bridge_frames=8,bridge_draws=2,
        PDB_events=24576,MD_events=8192,
        hypothesis="A frozen coarse parent avoids full-trunk forgetting. A small learned relative-geometry correction may improve state-dependent denoising on new sequences. Explicit radial geometry is compared to a parameter-matched Cartesian-relative-coordinate correction; no gain is assumed.",
        architecture="Own original12288-update coarse velocity field frozen; new29k-class edge MLP added to its velocity. Fine geometry-aware8192 candidate and own structure maps stay frozen. Every correction uses frozen AA block features, frozen time features, sequence separation and three spatial scalars. Aggregate bounded scalar weights times physical relative-center vectors, remove residue-count-weighted translation and multiply by t^2. Then convert physical velocity to the original normalized coordinates.",
        arms_definition={"cartesian":"Three normalized physical relative Cartesian coordinates as edge state features.",
                         "radial":"Three invariant radial features: sqrt(r^2+1e-8), log1p(r^2), r^2/(1+r^2)."},
        symmetry="Only the radial correction is translation invariant and rotation equivariant in physical velocity. The frozen parent and complete model are not claimed SE(3) equivariant. All coordinates retain the original gauge/normalization; no rotation augmentation or fine-interface change.",
        initialization="Both arms have identical adapter weight tensors/counts. Zero final layer yields exactly original-parent coarse velocities and samplers at update0, including fine outputs. Each seed starts fresh; not from regressed131072 or PDB-rehearsal weights.",
        training="Exactly8192 AdamW updates. Every update3 uniformly drawn static PDB structures plus1 uniformly chosen MD-training protein/frame, using the same event order, Gaussian/time keys and LR in both arms. Only edge-MLP parameters train. No development optimization, new MD download or pretrained-feature update.",
        objective="Uniform-time normalized native coarse FM MSE plus fixed0.01 mean squared normalized correction velocity, equally averaged over four slots. No spread-matching target, geometry reward, altered fine objective or inference sample selection.",
        evaluation="Final8192 endpoint only, both arms/seeds. Same64 raw AA-only samples and128 native references on5 seen MD proteins and2 reused development proteins; full, native-coarse fine oracle and shuffled-AA controls. Same frozen fine sampler/noise. Native interpolation endpoints at times0.1/0.5/0.9 use8 fixed reference frames and2 noises, matching the read-only mechanism study; these use native coordinates and are diagnostic only.",
        matching="Parameter-matched architecture ablation: same edge-MLP tensors/counts, zero initial function, native cases, base draws, LR, optimizer steps, frozen parent/fine, inference NFE. Radial and Cartesian feature arithmetic differs and actual time is measured. The parent lacks the adapter overhead; no inference-cost equivalence claim.",
        primary="Exploratory radial-versus-Cartesian native-neighborhood denoising and full AA-only accuracy/coverage. A bridge diagnostic alone cannot qualify a deployable improvement; report geometry, distribution and per-protein spread for every arm/seed. Compare with cached parent and author25/5 only in the same reused development cohort, without claiming matched pretraining/capacity.",
        scope="Bounded light coarse-adapter pilot;899 static PDB targets,2264 native MD frames from5 training proteins. Two repeatedly reused development sequences. No independent test/R3 access, model promotion, hyperparameter scan or calibrated-thermodynamics claim.",
        failure_policy="Retain every requested sample and all fine-frame failures. Any failed fine sample makes that sequence's full metrics unavailable. Nonfinite coarse intermediate states stop the job; never replace/filter candidates or silently fall back.",
        stop_rule="Exactly8192 updates per arm/seed, final endpoint only. Intermediate checkpoints are recovery snapshots, not candidates for best-checkpoint selection. No adaptive extension, weight scan or automatic default promotion.")
    train=torch.load(LABELS/"train_labels.pt",map_location="cpu",weights_only=True)
    dev=torch.load(LABELS/"development_labels.pt",map_location="cpu",weights_only=True)
    checks=[];initial_states={};codec=load_codec(p,"centers3")
    for seed in p["seeds"]:
        initial_states[str(seed)]={}
        for arm in ARMS:
            m=make_model(p,model_cases(train),seed,arm).eval()
            expected=[state_digest(m.coarse_model.base),state_digest(m.fine_model),state_digest(codec)]
            old_initial=load(FULL/f"seed{seed}/initial.json")
            if expected!=[old_initial[k] for k in ["coarse_sha256","fine_sha256","codec_sha256"]]:
                raise RuntimeError("Wrong immutable base/fine/codec")
            initial_states[str(seed)][arm]=state_digest(m.coarse_model.adapter)
            bank=m.coarse_model.feature_bank
            try:
                attach_development_features(p,m,dev)
                for name,cases in dev["validation"].items():
                    aa=encode_aa(cases[0]["sequence"]);key=sample_seed(seed,name,0)
                    c=project_centers(m.sample_coarse(aa,component_seed(key,"coarse"),16),len(aa))
                    base=project_centers(m.coarse_model.base.sample_coarse(aa,component_seed(key,"coarse"),16),len(aa))
                    torch.testing.assert_close(c,base,atol=0,rtol=0)
                    with torch.no_grad():bb=codec.decode(c,m.sample_fine(aa,c,component_seed(key,"fine"),16)).numpy()
                    with np.load(FULL/f"seed{seed}/step0/ensembles/{name}/full.npz") as z:cached=z["backbone"][0]
                    np.testing.assert_allclose(bb,cached,atol=2e-4,rtol=2e-5)
            finally:m.coarse_model.feature_bank=bank;m.fine_model.feature_bank=bank
            batch=sample_training_batch(train,batch_rng(seed));case=batch[-1];aa=encode_aa(case["sequence"])
            before=m.sample_fine(aa,case["coarse"],813,16)
            opt=torch.optim.AdamW(m.trainable_parameters(),lr=p["lr"],weight_decay=p["weight_decay"])
            clock=time.perf_counter();row=update(m,opt,batch,seed,0,p);seconds=time.perf_counter()-clock
            if row["gradient_norm"]<=0 or torch.count_nonzero(m.coarse_model.adapter[-1].weight)==0:raise RuntimeError("Missing real adapter update")
            torch.testing.assert_close(before,m.sample_fine(aa,case["coarse"],813,16),atol=0,rtol=0)
            frozen(m,codec,expected)
            checks.append(dict(seed=seed,arm=arm,trainable_parameters=sum(v.numel() for v in m.trainable_parameters()),
                zero_coarse_sampler_exact=True,frozen_fine_sampler_exact=True,discarded_preflight_updates=1,one_training_update_seconds=seconds))
        if initial_states[str(seed)][ARMS[0]]!=initial_states[str(seed)][ARMS[1]]:raise RuntimeError("Unequal adapter initialization")
    if len({c["trainable_parameters"] for c in checks})!=1:raise RuntimeError("Parameter matching failed")
    p["trainable_parameters"]=checks[0]["trainable_parameters"]
    atomic_json(ROOT/"integration_check.json",dict(status="passed",checks=checks,actual_trial_optimizer_updates=0))
    paths=[Path(__file__),Path("upstream/experiments/coarse_relative_geometry.py"),Path("upstream/tests/test_coarse_relative_geometry.py"),
        Path("upstream/scripts/coarse_relative_geometry.sbatch"),MECHANISM/"protocol.json",MECHANISM/"report.json",MECHANISM/"review.json",MECHANISM/"assessment.json",
        PREVIOUS/"protocol.json",PREVIOUS/"report.json",PREVIOUS/"review.json",FINE/"protocol.json",ROOT/"integration_check.json",
        LABELS/"train_labels.pt",LABELS/"development_labels.pt",Path(p["feature_index"]),Path(p["all_feature_index"]),Path(p["fine_checkpoint"])]
    paths += [Path(v) for v in p["tokenizer"].values()]
    for seed in p["seeds"]:
        paths += [Path(p["checkpoints"][str(seed)]),FINE/f"local_geometry/seed{seed}/last.pt",FULL/f"seed{seed}/initial.json"]
        for name in dev["validation"]:
            for outlet in ["full","oracle_coarse"]:paths.append(FULL/f"seed{seed}/step0/ensembles/{name}/{outlet}.npz")
    for entry in load(p["all_feature_index"]).values():paths.append(Path(entry["path"]))
    workspace=Path.cwd().resolve()
    for module in list(sys.modules.values()):
        name=getattr(module,"__file__",None)
        if name:
            q=Path(name).resolve()
            if q.suffix==".py" and q.is_relative_to(workspace/"upstream"):paths.append(q)
    p["source_sha256"]={str(q):sha(q) for q in dict.fromkeys(paths)}
    atomic_json(ROOT/"protocol.json",p)
    print("COARSE_RELATIVE_GEOMETRY_READY",json.dumps(checks),flush=True)


@torch.no_grad()
def bridges(p,model,train,dev,seed,out):
    rows=[]
    for split,groups in [("MD_train",train["train"]),("development",dev["validation"])]:
        for name,cases in sorted(groups.items()):
            refs=[cases[int(i)]["coarse"] for i in np.linspace(0,len(cases)-1,min(128,len(cases)),dtype=int)]
            aa=encode_aa(cases[0]["sequence"])
            for t in p["bridge_times"]:
                values=[]
                for i in np.linspace(0,len(refs)-1,min(len(refs),p["bridge_frames"]),dtype=int):
                    target=refs[int(i)]
                    for draw in range(p["bridge_draws"]):
                        noise=torch.randn(target.shape,generator=torch.Generator().manual_seed(diagnostic_key(seed,name,f"bridge_{i}",draw)))
                        pred=bridge_endpoint(model.coarse_model,aa,target,noise,t)
                        values.append(float(weighted_rmsd_matrix(pred.numpy()[None],target.numpy()[None],len(aa))[0,0]))
                rows.append(dict(split=split,protein=name,time=t,oracle_bridge_endpoint_center_RMSD_A=float(np.mean(values))))
    atomic_json(out/"native_bridge.json",dict(status="complete",rows=rows,note="Native structure is used only in this diagnostic, not raw AA-only sampling."))


def train(p,seed,arm,resume):
    groups=torch.load(LABELS/"train_labels.pt",map_location="cpu",weights_only=True)
    m=make_model(p,model_cases(groups),seed,arm);codec=load_codec(p,"centers3")
    expected=[state_digest(m.coarse_model.base),state_digest(m.fine_model),state_digest(codec)]
    opt=torch.optim.AdamW(m.trainable_parameters(),lr=p["lr"],weight_decay=p["weight_decay"])
    rng=batch_rng(seed);out=ROOT/arm/f"seed{seed}";out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists():raise RuntimeError("Correction arm already complete")
    start=0;trace=[];elapsed=0.;identity=hashlib.sha256()
    if resume:
        saved=restore_checkpoint(out/"last.pt",m,opt,rng,torch.device("cpu"))
        if saved["protocol_sha256"]!=sha(ROOT/"protocol.json") or saved["arm"]!=arm or saved["seed"]!=seed:raise RuntimeError("Wrong resume")
        start=saved["updates"];elapsed=saved["training_seconds"];restore_trace_prefix(out,start)
        trace=[json.loads(s) for s in (out/"trace.jsonl").read_text().splitlines()]
        for row in trace:identity.update(json.dumps(identity_record(row),sort_keys=True).encode())
        if identity.hexdigest()!=saved["realized_training_identity_sha256"]:raise RuntimeError("Resume trace identity differs")
    else:
        if (out/"initial.json").exists():raise RuntimeError("Explicit resume required")
        atomic_json(out/"initial.json",dict(frozen_component_sha256=expected,adapter_sha256=state_digest(m.coarse_model.adapter),
            trainable_parameters=p["trainable_parameters"],training_AA_sequences=len(m.coarse_model.feature_bank)))
    writer=CompactTrainingTrace(p["updates"],already_written=start);clock=time.perf_counter()
    for step in range(start,p["updates"]):
        row=update(m,opt,sample_training_batch(groups,rng),seed,step,p)
        trace.append(row);identity.update(json.dumps(identity_record(row),sort_keys=True).encode())
        if (step+1)%p["progress_every"]==0:
            writer(out/"trace.json",trace)
            progress=dict(seed=seed,arm=arm,phase="training",updates=step+1,total=p["updates"],FM_loss=row["FM_loss"],
                correction_squared_velocity=row["correction_squared_velocity"],training_seconds=elapsed+time.perf_counter()-clock,session_updates=step+1-start)
            atomic_json(out/"progress.json",progress);print("COARSE_RELATIVE_GEOMETRY_TRAIN",json.dumps(progress),flush=True)
        if (step+1)%p["checkpoint_every"]==0:
            frozen(m,codec,expected);writer(out/"trace.json",trace)
            save_checkpoint(out/"last.pt",m,opt,rng,seed=seed,arm=arm,updates=step+1,
                training_seconds=elapsed+time.perf_counter()-clock,protocol_sha256=sha(ROOT/"protocol.json"),realized_training_identity_sha256=identity.hexdigest())
    elapsed+=time.perf_counter()-clock
    atomic_json(out/"progress.json",dict(seed=seed,arm=arm,phase="evaluation",updates=p["updates"],total=p["updates"]))
    dev=torch.load(LABELS/"development_labels.pt",map_location="cpu",weights_only=True);bank=m.coarse_model.feature_bank
    try:
        attach_development_features(p,m,dev)
        with torch.no_grad():
            m.eval();bridges(p,m,groups,dev,seed,out)
            evaluate(p,m,codec,{"train":groups["train"],"validation":dev["validation"]},out/"ensembles",seed)
    finally:m.coarse_model.feature_bank=bank;m.fine_model.feature_bank=bank
    max_roundoff=0.
    for name in dev["validation"]:
        with np.load(out/f"ensembles/{name}/oracle_coarse.npz") as z:after=z["backbone"].copy()
        with np.load(FULL/f"seed{seed}/step0/ensembles/{name}/oracle_coarse.npz") as z:before=z["backbone"].copy()
        np.testing.assert_allclose(after,before,atol=2e-5,rtol=2e-5)
        max_roundoff=max(max_roundoff,float(np.max(np.abs(after-before))))
    frozen(m,codec,expected);verify(p)
    atomic_json(out/"complete.json",dict(status="complete",arm=arm,seed=seed,updates=p["updates"],training_seconds=elapsed,
        runtime_host=platform.node(),frozen_base_fine_codec=True,maximum_frozen_fine_roundoff_A=max_roundoff,
        realized_training_identity_sha256=identity.hexdigest(),evaluation=str(out/"ensembles/evaluation.json"),model_default_replaced=False))
    atomic_json(out/"progress.json",dict(seed=seed,arm=arm,phase="complete",updates=p["updates"],total=p["updates"]))


def report(p):
    rows=[];summaries=[];bridge_rows=[];done=[]
    for seed in p["seeds"]:
        identities=[]
        for arm in ARMS:
            folder=ROOT/arm/f"seed{seed}";c=load(folder/"complete.json")
            if not c["frozen_base_fine_codec"] or c["updates"]!=8192:raise RuntimeError("Incomplete or unfrozen correction")
            trace=[json.loads(s) for s in (folder/"trace.jsonl").read_text().splitlines()];identity=hashlib.sha256()
            if len(trace)!=8192 or [x["update"] for x in trace]!=list(range(1,8193)):raise RuntimeError("Bad correction trace")
            for x in trace:identity.update(json.dumps(identity_record(x),sort_keys=True).encode())
            if identity.hexdigest()!=c["realized_training_identity_sha256"]:raise RuntimeError("Correction identity mismatch")
            identities.append(identity.hexdigest());done.append(c)
            evaluation=load(c["evaluation"])
            for x in evaluation["rows"]:rows.append(dict(method=arm,seed=seed,**x))
            for x in load(folder/"native_bridge.json")["rows"]:bridge_rows.append(dict(method=arm,seed=seed,**x))
            for split,outlet in [("MD_train","full"),("development","full"),("development","oracle_coarse"),("development","scrambled_AA")]:
                group=[x for x in evaluation["rows"] if x["split"]==split and x["outlet"]==outlet]
                summaries.append(dict(method=arm,seed=seed,split=split,outlet=outlet,**summarize_rows(group)))
        if identities[0]!=identities[1]:raise RuntimeError("Arms did not match cases/noise/time/LR")
    previous=load(PREVIOUS/"report.json")
    for row in previous["rows"]:
        if row["method"] in ["parent","author_steps25","author_steps5"]:rows.append(row)
    mechanism=load(MECHANISM/"report.json")
    for row in mechanism["rows"]:
        if row["arm"]=="parent":
            for v in row["native_bridge"]:bridge_rows.append(dict(method="parent",seed=row["seed"],split=row["split"],protein=row["protein"],**v))
    verify(p)
    atomic_json(ROOT/"report.json",dict(status="complete",rows=rows,summaries=summaries,native_bridge=bridge_rows,
        completed_models=done,paired_training_draws=True,scope=p["scope"],model_default_replaced=False,independent_test_claim=False))
    atomic_json(ROOT/"assessment.json",dict(status="complete_awaiting_review",scope=p["scope"],
        next="Review native-bridge and raw full accuracy/coverage/geometry together for both seeds. No automatic extension, selection or promotion."))
    plot(rows,bridge_rows)
    print("COARSE_RELATIVE_GEOMETRY_REPORT_COMPLETE",flush=True)


def plot(rows,bridge):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods=["author_steps25","parent","cartesian","radial"]
    fig,axes=plt.subplots(1,4,figsize=(16,4),constrained_layout=True)
    for ax,key,title,scale in zip(axes,["bridge","mean_sample_nearest_reference_ca_rmsd_A","mean_reference_nearest_sample_ca_rmsd_A","cn_rate"],
        ["Native bridge t=.9(A)","Generated to reference(A)","Reference coverage error(A)","C-N abnormalities(%)"],[1.,1.,1.,100.]):
        for i,method in enumerate(methods):
            values=[]
            for seed in [2026,2027]:
                if key=="bridge":
                    group=[x for x in bridge if x["method"]==method and x["seed"]==seed and x["split"]=="development" and x["time"]==.9]
                    values.append(float(np.mean([x["oracle_bridge_endpoint_center_RMSD_A"] for x in group])) if group else None)
                else:
                    group=[x for x in rows if x["method"]==method and x["seed"]==seed and x["split"]=="development" and x["outlet"]=="full"]
                    if not group or not all(x["metrics_available"] for x in group):values.append(None)
                    else:values.append(scale*float(np.mean([x["geometry"][key] if key=="cn_rate" else x[key] for x in group])))
            if all(v is not None for v in values):
                ax.bar(i,np.mean(values),color="#327ba8" if method=="radial" else "#aaaaaa",alpha=.7)
                ax.scatter([i-.08,i+.08],values,c="black",s=18)
            elif key=="bridge" and method=="author_steps25":
                ax.text(i,.02,"N/A",ha="center",transform=ax.get_xaxis_transform(),fontsize=8)
        ax.set_xticks(range(4),["Author25","Our parent","Cartesian","Radial"],rotation=30);ax.set_title(title);ax.grid(axis="y",alpha=.2)
    fig.suptitle("Frozen-parent coarse correction ·matched adapters ·2 reused development proteins /2 seeds\nNative bridges use reference centers; full generation uses AA only. No independent-test claim.")
    dest=Path("paper/figures/coarse_relative_geometry_v1");dest.mkdir(parents=True,exist_ok=True)
    for ext in ["pdf","png","svg"]:fig.savefig(dest/("Coarse_Relative_Geometry."+ext),dpi=170)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["prepare","train","report"],required=True)
    parser.add_argument("--seed",type=int,choices=[2026,2027]);parser.add_argument("--arm",choices=ARMS);parser.add_argument("--resume",action="store_true")
    args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=="prepare":prepare()
    else:
        protocol=load(ROOT/"protocol.json");verify(protocol)
        if args.stage=="train":train(protocol,args.seed,args.arm,args.resume)
        else:report(protocol)
