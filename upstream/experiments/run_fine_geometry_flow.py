"""Controlled fine-flow continuation with a frozen parent coarse sampler."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch

from compact_training_trace import CompactTrainingTrace,restore_trace_prefix
from compare_two_level_ae_author import metrics
from diagnostic_io import atomic_json
from fine_geometry_flow import FixedCoarseFineFlow,sample_training_batch,draw_seed,fine_flow_loss,fine_rollout_geometry_loss
from hierarchical_ensemble_flow import component_seed
from hierarchical_latent_contract import pool_ca
from hierarchical_pipeline_io import sha,save_checkpoint,restore_checkpoint
from joint_sequence_predictor import encode_aa
from latent_source_diagnostic import diagnostic_seed
from run_sequence_condition_probe import make_model as make_parent_model
from run_sequence_multiscale_ensemble import load_codec
from run_multisequence_ensemble_flow import evaluate
from sequence_feature_ensemble_flow import SequenceFeatureEnsembleFlow
from sequence_multiscale_ensemble import sample_levels

RUN=Path("artifacts/fine_geometry_flow_v1")
PARENT=Path("artifacts/generator_curriculum_v1")
DIAGNOSTIC=Path("artifacts/latent_source_diagnostic_v1")
ARMS=["fine_flow_only","fine_flow_geometry"]


def verify(p):
    for name,digest in p["source_sha256"].items():
        if sha(name)!=digest:raise RuntimeError("Changed fine training input: "+name)


def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/"protocol.json").exists():raise RuntimeError("Fine protocol already frozen")
    parent=json.loads((PARENT/"protocol.json").read_text());verify(parent)
    audit=json.loads((DIAGNOSTIC/"continuous_geometry_gradient_audit.json").read_text())
    if audit["status"]!="passed" or len(audit["rows"])!=8:raise RuntimeError("Missing exact gradient checks")
    integration=json.loads((RUN/"integration_check.json").read_text())
    if integration["status"]!="passed" or not integration["coarse_sampling_bitwise_unchanged_after_update"]:
        raise RuntimeError("Missing real-checkpoint integration check")
    # Check that the exact differentiable helper used below matches that audit.
    for path,digest in audit["source_sha256"].items():
        if sha(path)!=digest:raise RuntimeError("Changed audited geometry interface: "+path)
    shutil.copyfile(PARENT/"labels.pt",RUN/"labels.pt")
    p=copy.deepcopy(parent)
    p.update(arms=ARMS,updates=2048,batch_size=4,PDB_events=6144,MD_events=2048,
        lr=1e-4,min_lr=1e-5,geometry_weights={ARMS[0]:0.,ARMS[1]:.01},
        checkpoints={str(seed):str(PARENT/"interleaved"/f"seed{seed}"/"last.pt") for seed in [2026,2027]},
        parent_protocol=str(PARENT/"protocol.json"),parent_arm="interleaved",
        training="Fine-only continuation from each fixed 12288-step interleaved checkpoint; fresh AdamW in both arms. Per update 3 uniform PDB training examples and 1 uniformly selected MD-training protein/frame. Same batches, fine Gaussian/time draws and learning rate. Coarse sampler is a separate immutable copy of the parent.",
        objective="Mean normalized fine flow velocity MSE over four teacher examples + lambda * geometry of ONE full 16-step Euler fine rollout. Rollout case is batch slot update_index modulo 4 (therefore PDB:MD 3:1), with an independent paired Gaussian seed. Geometry is mean C-N hinge-squared outside [1.1,1.7] A + mean boundary C-N hinge-squared + 0.1 CA-clash hinge-squared below 3 A. Lambda is fixed at 0 or 0.01 without a scan. Geometry proxies are not force-field energies.",
        matching="Both arms run identical teacher and complete 16-step rollout/decoder forward-backward branches, including geometry at zero weight in the control. Full trajectory gradients, without detached prefixes; non-reentrant activation checkpointing recomputes velocity forwards during backward. Same initial fine weights, batches, teacher/rollout Gaussian draws, optimizer updates and inference NFE. Actual time is measured; frozen ESM3 cost is shared.",
        architecture="Keep a frozen parent model for coarse sampling and an initially identical model for trainable fine inference. Fine reads the sampled coarse through the same broadcast/prefix interfaces. Duplication is needed because updating a shared Transformer would change coarse. This increases stored generator weights; neither arm increases inference NFE.",
        fine_training_condition="Native coarse from the SAME training structure; its cached native fine label is kept in that same coarse-derived local frame. This isolates fine training and does not eliminate training/rollout exposure bias.",
        geometry_gradient="Differentiate all 16 Euler steps and the frozen fine_tokenizer.decode_vectors. No detached rollout prefix, VQ snap or STE. Do not use the no_grad inference convenience decoder. Checkpointed and direct full-rollout gradients are checked on a small model; real-checkpoint integration verifies fine updates and frozen coarse/codec.",
        preflight_rationale="Before freezing, one training-only batch showed near-zero penalties on teacher-corrupted endpoint predictions despite bad rollout geometry. The retained endpoint_integration_check records this. Full-rollout preflight showed a geometry/flow parameter-gradient norm ratio about 0.049 at lambda 0.01; no development data or model-quality scan was used.",
        compute_interpretation="The zero-weight arm intentionally retains the geometry graph to isolate the objective under equal computation in this ablation. It is not a minimum-cost implementation of pure flow and cannot establish training efficiency against an optimized flow-only baseline. Report measured training overhead and unchanged inference NFEs separately.",
        PDB_probe_ids=parent["pretrain_probe_ids"],PDB_probe_samples=4,
        primary="Same two repeatedly used development proteins, 64 retained raw AA-only samples each. Compare both trained arms and their unchanged parent, with accuracy, geometry, spread/coverage and elapsed time. Oracle-coarse and training-set probes are diagnostics only.",
        stop_rule="Fixed final 2048-update endpoint only; no best checkpoint, loss-weight scan, sample rejection, reference-based acceptance, test/R3 access or automatic model replacement.",
        scope="Exploratory fine-only pilot; two reused development sequences, two seeds. Reserved test and R3 coordinates stay closed; ESM3 pretraining exposure is unknown.")
    paths=[Path(__file__),Path("upstream/experiments/fine_geometry_flow.py"),
        Path("upstream/tests/test_fine_geometry_flow.py"),Path("upstream/scripts/fine_geometry_flow.sbatch"),
        Path("upstream/experiments/report_fine_geometry_flow.py"),
        Path("upstream/experiments/audit_continuous_fine_geometry.py"),
        Path("upstream/experiments/geometry_objective.py"),
        PARENT/"protocol.json",PARENT/"report.json",DIAGNOSTIC/"protocol.json",
        DIAGNOSTIC/"report.json",DIAGNOSTIC/"continuous_geometry_gradient_audit.json",
        RUN/"integration_check.json",RUN/"endpoint_integration_check.json",RUN/"labels.pt",
        Path("upstream/experiments/fine_geometry_flow.rst")]
    for seed in p["seeds"]:
        paths += [Path(p["checkpoints"][str(seed)]),PARENT/"interleaved"/f"seed{seed}"/"evaluation.json",
                  DIAGNOSTIC/f"seed{seed}"/"complete.json"]
    p["source_sha256"].update({str(path):sha(path) for path in paths})
    atomic_json(RUN/"protocol.json",p)
    print("FINE_GEOMETRY_PROTOCOL_READY",p["updates"],p["geometry_weights"],flush=True)


def state_digest(model):
    value=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        value.update(name.encode());value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def make_model(p,seed):
    parent=make_parent_model(p,"ESM3_context")
    saved=torch.load(p["checkpoints"][str(seed)],map_location="cpu",weights_only=False)
    if saved["updates"]!=12288 or saved["protocol_sha256"]!=sha(PARENT/"protocol.json"):
        raise RuntimeError("Wrong parent generator checkpoint")
    parent.load_state_dict(saved["model"])
    fine=SequenceFeatureEnsembleFlow(**p["normalization"],**p["model"],
        coarse_dim=3,feature_mode="ESM3_context",feature_dim=p["feature_dimension"])
    fine.load_state_dict(saved["model"])
    # Frozen AA tensors can share storage; neither branch changes this bank.
    fine.feature_bank=parent.feature_bank
    model=FixedCoarseFineFlow(parent,fine)
    if state_digest(parent)!=state_digest(fine):raise RuntimeError("Fine initialization differs from parent")
    return model


@torch.no_grad()
def pdb_probe(p,model,codec,groups,seed,out):
    cases={c["id"]:c for c in groups["PDB_train"]};rows=[]
    for name in p["PDB_probe_ids"]:
        c=cases[name];aa=encode_aa(c["sequence"])
        for draw in range(p["PDB_probe_samples"]):
            key=component_seed(diagnostic_seed(seed,name,draw),"fine")
            fine=model.sample_fine(aa,c["coarse"],key,p["fine_nfe"])
            bb=codec.decode(c["coarse"],fine)
            rows.append(dict(case_id=name,protein=c["protein"],draw=draw,metrics=metrics(bb.numpy(),c["target"].numpy())))
    atomic_json(out/"PDB_oracle_probe.json",dict(status="complete",rows=rows,
        scope="Eight fixed PDB training proteins; true coarse; same noise as the completed latent-source diagnostic. Not AA-only generation or independent validation."))


def train(p,arm,seed):
    groups=torch.load(RUN/"labels.pt",map_location="cpu",weights_only=True)
    codec=load_codec(p,"centers3");model=make_model(p,seed)
    initial_coarse=state_digest(model.coarse_model);initial_codec=state_digest(codec)
    initial_buffers={name:value.clone() for name,value in model.fine_model.named_buffers()}
    params=model.trainable_parameters()
    opt=torch.optim.AdamW(params,lr=p["lr"],weight_decay=p["weight_decay"])
    rng=np.random.default_rng(seed+37247)
    out=RUN/arm/f"seed{seed}";out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists():raise RuntimeError("Fine arm already complete")
    start=0;digest=sha(RUN/"protocol.json")
    first=next(c for c in groups["PDB_train"] if c["id"]==p["PDB_probe_ids"][0])
    aa=encode_aa(first["sequence"]);key=diagnostic_seed(seed,first["id"],0)
    model.eval();before=sample_levels(model,codec,aa,key)
    parent_c=model.coarse_model.sample_coarse(aa,component_seed(key,"coarse"),16)
    parent_c=codec.project_coarse(parent_c,len(aa))
    parent_f=model.coarse_model.sample_fine(aa,parent_c,component_seed(key,"fine"),16)
    torch.testing.assert_close(before[0],parent_c,atol=0,rtol=0)
    torch.testing.assert_close(before[1],parent_f,atol=0,rtol=0)
    if (out/"last.pt").exists():
        saved=restore_checkpoint(out/"last.pt",model,opt,rng,torch.device("cpu"))
        if saved["protocol_sha256"]!=digest:raise RuntimeError("Resume source mismatch")
        start=saved["updates"]
    else:
        atomic_json(out/"initial.json",dict(seed=seed,coarse_sha256=initial_coarse,
            fine_sha256=state_digest(model.fine_model),codec_sha256=initial_codec,
            trainable_parameters=sum(x.numel() for x in params),epoch0_sampler_bitwise_replayed=True))
    restore_trace_prefix(out,start)
    rows=[json.loads(line) for line in (out/"trace.jsonl").read_text().splitlines()]
    writer=CompactTrainingTrace(p["updates"],start);started=time.perf_counter()
    for step in range(start,p["updates"]):
        model.train();batch=sample_training_batch(groups,rng)
        rate=p["min_lr"]+.5*(p["lr"]-p["min_lr"])*(1+math.cos(math.pi*step/max(1,p["updates"]-1)))
        for group in opt.param_groups:group["lr"]=rate
        opt.zero_grad(set_to_none=True);flow_total=0.;terms=[]
        for slot,case in enumerate(batch):
            gen=torch.Generator().manual_seed(draw_seed(seed,step,slot))
            value,term=fine_flow_loss(model,encode_aa(case["sequence"]),
                case["coarse"],case["fine_continuous"],gen)
            if not torch.isfinite(value):raise RuntimeError("Nonfinite fine objective")
            (value/4).backward();flow_total+=float(value.detach())/4;terms.append(term)
        rollout_case=batch[step%4]
        geometry,rollout_terms,_,_=fine_rollout_geometry_loss(model,codec,
            encode_aa(rollout_case["sequence"]),rollout_case["coarse"],draw_seed(seed,step,4),p["fine_nfe"])
        if not torch.isfinite(geometry):raise RuntimeError("Nonfinite rollout geometry")
        weighted=p["geometry_weights"][arm]*geometry
        # The zero-weight control retains the same full backward graph.
        weighted.backward()
        total=flow_total+float(weighted.detach())
        grad=torch.nn.utils.clip_grad_norm_(params,p["gradient_clip"])
        if not torch.isfinite(grad):raise RuntimeError("Nonfinite fine gradient")
        opt.step()
        row=dict(update=step+1,loss=total,lr=rate,batch=[c["id"] for c in batch],
            draws=[t["draw_sha256"] for t in terms],gradient_norm=float(grad),
            flow=flow_total,geometry=float(geometry.detach()),weighted_geometry=float(weighted.detach()),
            rollout_case=rollout_case["id"],rollout_draw=rollout_terms["draw_sha256"])
        rows.append(row)
        if (step+1)%64==0 or step+1==p["updates"]:
            writer(out/"trace.json",rows)
            progress=dict(arm=arm,seed=seed,updates=step+1,total=p["updates"],
                seconds=time.perf_counter()-started,loss=total,flow=row["flow"],weighted_geometry=row["weighted_geometry"])
            atomic_json(out/"progress.json",progress);print("FINE_GEOMETRY_TRAIN",json.dumps(progress),flush=True)
        if (step+1)%512==0 or step+1==p["updates"]:
            save_checkpoint(out/"last.pt",model,opt,rng,updates=step+1,model_config=model.config,
                protocol_sha256=digest,arm=arm,seed=seed)
    model.eval()
    if state_digest(model.coarse_model)!=initial_coarse:raise RuntimeError("Coarse model changed")
    if state_digest(codec)!=initial_codec:raise RuntimeError("Structure codec changed")
    for name,value in model.fine_model.named_buffers():torch.testing.assert_close(value,initial_buffers[name],atol=0,rtol=0)
    torch.testing.assert_close(before[0],sample_levels(model,codec,aa,key)[0],atol=0,rtol=0)
    evaluate(p,model,codec,groups,out,seed)
    pdb_probe(p,model,codec,groups,seed,out)
    # Every final sample must preserve the parent coarse centers, including tails.
    maximum=0.
    for row in json.loads((out/"evaluation.json").read_text())["rows"]:
        if row["outlet"]!="full":continue
        name=row["protein"]
        path=PARENT/"interleaved"/f"seed{seed}"/name/"full.npz"
        with np.load(path) as z:old=z["backbone"].copy()
        with np.load(out/name/"full.npz") as z:new=z["backbone"].copy()
        if old.shape!=new.shape:raise RuntimeError("Before/after sample count differs")
        for x,y in zip(old,new):
            a=pool_ca(torch.from_numpy(x));b=pool_ca(torch.from_numpy(y))
            maximum=max(maximum,float((a-b).abs().max()))
            torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
    verify(p)
    atomic_json(out/"complete.json",dict(status="complete",arm=arm,seed=seed,
        updates=p["updates"],seconds=time.perf_counter()-started,
        trainable_parameters=sum(v.numel() for v in params),coarse_weights_unchanged=True,
        runtime_host=platform.node(),torch_threads=torch.get_num_threads(),
        coarse_sha256=initial_coarse,codec_weights_unchanged=True,normalization_buffers_unchanged=True,
        max_sample_coarse_center_difference_A=maximum,test_coordinates_opened=False,R3_opened=False))
    print("FINE_GEOMETRY_ARM_COMPLETE",arm,seed,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["prepare","train"],required=True)
    parser.add_argument("--arm",choices=ARMS);parser.add_argument("--seed",type=int,default=2026);args=parser.parse_args()
    torch.set_num_threads(2)
    if args.stage=="prepare":prepare()
    else:
        p=json.loads((RUN/"protocol.json").read_text());verify(p)
        if args.seed not in p["seeds"] or args.arm not in p["arms"]:raise RuntimeError("Undeclared fine arm")
        train(p,args.arm,args.seed)
