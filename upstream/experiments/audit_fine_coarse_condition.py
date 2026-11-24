"""Read-only sensitivity audit on fixed same-protein native coarse pairs.

Native pairs are first/last cached train or reused-development frames. A swapped
teacher condition can be out of the joint training distribution; this is an
intervention probe, not proof of conditional sufficiency or model quality.
"""
import argparse,copy,hashlib,json
from pathlib import Path

import numpy as np
import torch

from compare_two_level_ae_author import metrics
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from joint_sequence_predictor import encode_aa
from run_sequence_multiscale_ensemble import load_codec
from sequence_feature_ensemble_flow import SequenceFeatureEnsembleFlow
from structural_metrics import aligned_rmsd

RUN=Path("artifacts/fine_coarse_condition_audit_v1")
PARENT=Path("artifacts/fine_target_scaling_v1")
BASE=Path("artifacts/generator_curriculum_v1")


def verify(p):
    for name,digest in p["source_sha256"].items():
        if sha(name)!=digest:raise RuntimeError("Changed conditioning-audit input: "+name)


def key(seed,name,phase,index):
    return int(hashlib.sha256(f"fine-coarse-response-v1:{seed}:{name}:{phase}:{index}".encode()).hexdigest()[:8],16)


def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/"protocol.json").exists():raise RuntimeError("Conditioning audit already frozen")
    p=copy.deepcopy(json.loads((PARENT/"protocol.json").read_text()));verify(p)
    result=json.loads((PARENT/"assessment.json").read_text())
    if result["status"]!="complete_training_fit_improves_development_tradeoff_no_overall_gain":raise RuntimeError("Missing completed scaling assessment")
    labels=torch.load(BASE/"labels.pt",map_location="cpu",weights_only=True)
    pairs=[]
    for split in ["train","validation"]:
        for name,cases in sorted(labels[split].items()):
            a,b=cases[0],cases[-1]
            if a["sequence"]!=b["sequence"] or a["split"]!=split or b["split"]!=split:
                raise RuntimeError("Mismatched native pair")
            pairs.append(dict(protein=name,split=split,a=a,b=b))
    if len(pairs)!=7:raise RuntimeError("Expected five train and two reused-development pairs")
    torch.save(pairs,RUN/"pairs.pt")
    p.update(versions=["parent","32_long"],noise_samples=4,
        selection="First and last cached native training frames from each of five parent MD-training proteins; first and last cached reused-development frames from both development proteins. No selection by separation or outcome.",
        audit="At fixed AA and paired fine Gaussian noise, change native coarse from frame A to B and measure normalized fine response, native fine-label change and local geometry-context change. Also compare teacher velocities under matched versus swapped native coarse at times 0.1/0.5/0.9 with identical noise.",
        scope="Read-only native-coarse sensitivity audit, not AA-only generation, training, independent validation or ensemble fidelity. Five proteins were seen by the parent, but not used in the PDB-only 32-target adaptation. Both development proteins are repeatedly reused.",
        caveat="Swapped (coarse, noisy fine) pairs have valid marginal components but can be outside their joint training distribution. Small response alone cannot establish that coarse is ignored, especially if native fine differences are small. No sensitivity threshold is selected as a performance gate.")
    sources=[Path(__file__),Path("upstream/scripts/fine_coarse_condition_audit.sbatch"),
        PARENT/"protocol.json",PARENT/"report.json",PARENT/"assessment.json",RUN/"pairs.pt",BASE/"labels.pt"]
    for seed in p["seeds"]:sources.append(PARENT/"matched_presentations"/f"seed{seed}"/"last.pt")
    p["source_sha256"].update({str(v):sha(v) for v in sources});atomic_json(RUN/"protocol.json",p)
    print("FINE_COARSE_AUDIT_READY",len(pairs),flush=True)


def model_for(p,pairs,seed,version):
    model=SequenceFeatureEnsembleFlow(**p["normalization"],**p["model"],coarse_dim=3,
        feature_mode="ESM3_context",feature_dim=p["feature_dimension"])
    path=Path(p["checkpoints"][str(seed)]) if version=="parent" else PARENT/"matched_presentations"/f"seed{seed}"/"last.pt"
    saved=torch.load(path,map_location="cpu",weights_only=False)
    if version=="32_long" and saved["protocol_sha256"]!=sha(PARENT/"protocol.json"):raise RuntimeError("Wrong adapted checkpoint")
    model.load_state_dict(saved["model"])
    features=json.loads(Path(p["feature_index"]).read_text());bank={}
    for row in pairs:
        sequence=row["a"]["sequence"];entry=features[sequence]
        if sha(entry["path"])!=entry["sha256"]:raise RuntimeError("Changed AA feature")
        with np.load(entry["path"],allow_pickle=False) as z:bank[sequence]=torch.from_numpy(z["h"].copy()).float()
    model.attach_feature_bank(bank);return model.eval().requires_grad_(False)


def rms(x):return float(x.square().mean().sqrt())


@torch.no_grad()
def audit(p,seed):
    pairs=torch.load(RUN/"pairs.pt",map_location="cpu",weights_only=True)
    codec=load_codec(p,"centers3");records=[]
    for version in p["versions"]:
        model=model_for(p,pairs,seed,version)
        original={k:v.clone() for k,v in model.state_dict().items()}
        for pair in pairs:
            a,b=pair["a"],pair["b"];name=pair["protein"];aa=encode_aa(a["sequence"])
            targets=[a,b];conditions=[a["coarse"],b["coarse"]]
            normalized_labels=[model.normalize_fine(v["fine_continuous"]) for v in targets]
            samples=[];quality=[]
            for index in range(p["noise_samples"]):
                values=[]
                for target,c in zip(targets,conditions):
                    f=model.sample_fine(aa,c,key(seed,name,"sample",index),16)
                    values.append(model.normalize_fine(f));bb=codec.decode(c,f)
                    quality.append(metrics(bb.numpy(),target["target"].numpy()))
                samples.append(torch.stack(values))
            samples=torch.stack(samples);mean=samples.mean(0)
            native_delta=normalized_labels[1]-normalized_labels[0];predicted_delta=mean[1]-mean[0]
            target_change=rms(native_delta);response=rms(predicted_delta)
            cosine=None
            if target_change>1e-8 and response>1e-8:cosine=float((native_delta*predicted_delta).sum()/(native_delta.norm()*predicted_delta.norm()))
            teacher=[]
            for t in [.1,.5,.9]:
                matched=[];swapped=[];responses=[]
                for side,target in enumerate(targets):
                    f=normalized_labels[side]
                    for index in range(p["noise_samples"]):
                        rng=torch.Generator().manual_seed(key(seed,name,f"teacher{side}",index))
                        noise=torch.randn(f.shape,generator=rng);state=(1-t)*noise+t*f;truth=f-noise
                        va=model.fine_velocity(aa,state,t,model.normalize_coarse(conditions[side]))
                        vb=model.fine_velocity(aa,state,t,model.normalize_coarse(conditions[1-side]))
                        matched.append(float((va-truth).square().mean()));swapped.append(float((vb-truth).square().mean()));responses.append(rms(va-vb))
                teacher.append(dict(time=t,matched_velocity_mse=float(np.mean(matched)),swapped_velocity_mse=float(np.mean(swapped)),
                    swapped_minus_matched_mse=float(np.mean(swapped)-np.mean(matched)),velocity_condition_response_rms=float(np.mean(responses))))
            geometry=[codec.context(c,len(aa))["geometry"] for c in conditions]
            records.append(dict(seed=seed,version=version,protein=name,split=pair["split"],case_ids=[a["id"],b["id"]],
                coarse_aligned_center_rmsd_A=aligned_rmsd(conditions[0].numpy(),conditions[1].numpy()),
                native_geometry_context_change_rms=rms(geometry[1]-geometry[0]),
                native_fine_change_rms=target_change,predicted_mean_fine_change_rms=response,
                paired_sample_fine_change_rms=float(np.mean([rms(v[1]-v[0]) for v in samples])),
                within_condition_noise_rms=rms(samples-mean),
                response_to_native_change_ratio=response/target_change if target_change>1e-8 else None,
                response_native_direction_cosine=cosine,teacher=teacher,
                oracle_coarse_geometry={k:float(np.mean([q[k] for q in quality])) for k in ['ca_rmsd','cn_rate','boundary_cn_rate']}))
            print("FINE_COARSE_RESPONSE",seed,version,name,target_change,response,flush=True)
        for k,v in model.state_dict().items():torch.testing.assert_close(v,original[k],atol=0,rtol=0)
    verify(p);atomic_json(RUN/f"seed{seed}.json",dict(status="complete",records=records,
        model_weights_unchanged=True,training_performed=False,test_coordinates_opened=False,R3_opened=False))


def report(p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records=[]
    for seed in p["seeds"]:
        d=json.loads((RUN/f"seed{seed}.json").read_text())
        if not d["model_weights_unchanged"]:raise RuntimeError("Audit mutated model")
        records+=d["records"]
    verify(p);atomic_json(RUN/"report.json",dict(status="complete",records=records,scope=p["scope"],caveat=p["caveat"],
        model_default_replaced=False,training_performed=False,test_coordinates_opened=False,R3_opened=False))
    names=list(dict.fromkeys(r["protein"] for r in records));fig,axes=plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True)
    for version,color in [("parent","#777777"),("32_long","#24679e")]:
        rows=[[r for r in records if r["version"]==version and r["protein"]==name] for name in names]
        for ax,key in [(axes[0],"native_fine_change_rms"),(axes[1],"predicted_mean_fine_change_rms")]:
            ax.plot(range(len(names)),[np.mean([r[key] for r in group]) for group in rows],marker="o",color=color,label=version)
        axes[2].plot(range(len(names)),[np.mean([r["teacher"][0]["swapped_minus_matched_mse"] for r in group]) for group in rows],marker="o",color=color)
    for ax,title in zip(axes,["Native fine-label change (normalized)","Predicted fine response (normalized)","Swapped minus matched velocity MSE, t=0.1"]):
        ax.set_title(title,fontsize=10);ax.set_xticks(range(len(names)),names,rotation=45,ha="right",fontsize=8);ax.grid(axis="y",alpha=.2)
    axes[0].legend(frameon=False);fig.suptitle("Fine response to coarse: fixed native frame pairs, sensitivity only")
    folder=Path("paper/figures/fine_coarse_condition_audit_v1");folder.mkdir(parents=True,exist_ok=True)
    for ext in ["pdf","png","svg"]:fig.savefig(folder/("Fine_Coarse_Response."+ext),dpi=180)
    plt.close(fig);print("FINE_COARSE_AUDIT_COMPLETE",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["prepare","audit","report"],required=True)
    parser.add_argument("--seed",type=int,default=2026);args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=="prepare":prepare()
    else:
        p=json.loads((RUN/"protocol.json").read_text());verify(p)
        if args.stage=="report":report(p)
        else:
            if args.seed not in p["seeds"]:raise RuntimeError("Undeclared seed")
            audit(p,args.seed)
