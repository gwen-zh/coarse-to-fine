"""Read-only attribution probes after the bounded coarse-coverage studies."""
import argparse
from collections import Counter, deque
import copy
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from coarse_coverage_mechanism import (bridge_endpoint, distance_features, distance_variance,
    event_loss, flatten_gradient, gradient_projection, mean_event_gradient, observable_moments,
    parameters, project, response_gain, same_forward_kernel, trajectory)
from coarse_generation_diagnostic import coarse_metrics, pool_backbones, weighted_rmsd_matrix
from diagnostic_io import atomic_json
from hierarchical_ensemble_flow import component_seed
from hierarchical_pipeline_io import sha
from joint_sequence_predictor import encode_aa
from sequence_feature_ensemble_flow import SequenceFeatureEnsembleFlow


ROOT = Path("artifacts/coarse_coverage_mechanism_v1")
FULL = Path("artifacts/coarse_full_cohort_v1")
REPLAY = Path("artifacts/coarse_coverage_repair_v1")
LABELS = Path("artifacts/multiframe_early_fine_v1")
ARMS = ["parent", "static_FM32768", "static_FM131072", "PDB_replay32768"]


def load(path): return json.loads(Path(path).read_text())


def digest_state(state):
    value = hashlib.sha256()
    for name, tensor in state.items():
        value.update(name.encode()); value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def sample_key(seed, name, index):
    return int(hashlib.sha256(f"multi-flow-v1:{seed}:{name}:{index}".encode()).hexdigest()[:8], 16)


def diagnostic_key(seed, name, kind, index):
    return int(hashlib.sha256(f"coarse-mechanism-v1:{seed}:{name}:{kind}:{index}".encode()).hexdigest()[:8], 16)


def registry(groups):
    cases = groups["PDB_train"] + [c for frames in groups["train"].values() for c in frames]
    by_id = {c["id"]: c for c in cases}
    if len(by_id) != len(cases): raise RuntimeError("Training IDs are not unique")
    return by_id


def selected_cases(p, train, dev):
    rows = []
    for split, by_name in [("MD_train", train["train"]), ("development", dev["validation"])]:
        for name, cases in sorted(by_name.items()):
            refs = [cases[int(i)] for i in np.linspace(0, len(cases) - 1, min(128, len(cases)), dtype=int)]
            rows.append(dict(protein=name, split=split, sequence=cases[0]["sequence"],
                references=torch.stack([c["coarse"] for c in refs])))
    by_id = registry(train)
    for name in p["PDB_probe_ids"]:
        c = by_id[name]
        rows.append(dict(protein=c["protein"], split="PDB_train_single_structure", sequence=c["sequence"], references=c["coarse"][None]))
    return rows


def model(p, seed, arm):
    m = SequenceFeatureEnsembleFlow(**p["normalization"], **p["model"], coarse_dim=3,
        feature_mode="ESM3_context", feature_dim=p["feature_dimension"])
    spec = p["checkpoints"][str(seed)][arm]
    saved = torch.load(spec["path"], map_location="cpu", weights_only=False)
    state = saved["model"]
    if spec["prefix"]: state = {k[len(spec["prefix"]):]: v for k, v in state.items() if k.startswith(spec["prefix"])}
    m.load_state_dict(state, strict=True); del saved
    if digest_state(m.state_dict()) != spec["coarse_state_sha256"]: raise RuntimeError("Wrong coarse state")
    for name in ["fine_head", "fine_input", "coarse_to_fine"]: getattr(m, name).requires_grad_(False)
    return m.eval()


def feature_bank(p):
    bank = {}
    for sequence, entry in p["features"].items():
        if sha(entry["path"]) != entry["sha256"]: raise RuntimeError("Changed AA-only features")
        with np.load(entry["path"]) as z: value = torch.from_numpy(z["h"].copy()).float()
        if value.shape != (len(sequence), p["feature_dimension"]) or not torch.isfinite(value).all():
            raise RuntimeError("Invalid cached AA-only feature mapping")
        bank[sequence] = value
    return bank


def verify(p):
    for name, digest in p["source_sha256"].items():
        if sha(name) != digest: raise RuntimeError("Changed mechanism input: " + name)


def cache_folder(seed, arm):
    if arm == "PDB_replay32768": return REPLAY / f"seed{seed}/repair/ensembles"
    step = {"parent": 0, "static_FM32768": 32768, "static_FM131072": 131072}[arm]
    return FULL / f"seed{seed}/step{step}/ensembles"


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "protocol.json").exists(): raise RuntimeError("Fresh diagnostic namespace required")
    old = load(REPLAY / "protocol.json")
    if load(REPLAY / "assessment.json")["status"] != "complete_reviewed_replay_preserves_parent_coverage_author_baseline_remains_better":
        raise RuntimeError("Review the repair/author comparison first")
    train = torch.load(LABELS / "train_labels.pt", map_location="cpu", weights_only=True)
    dev = torch.load(LABELS / "development_labels.pt", map_location="cpu", weights_only=True)
    by_id = registry(train); batches = {}; events = {}; checkpoints = {}; paths = []
    for seed in [2026, 2027]:
        recent = deque(maxlen=16)
        with (FULL / f"seed{seed}/trace.jsonl").open() as f:
            for line in f:
                row = json.loads(line)
                if row["update"] > 32768: break
                recent.append(row)
        if len(recent) != 16 or recent[-1]["update"] != 32768: raise RuntimeError("Incomplete fixed control trace")
        batches[str(seed)] = [v["update"] for v in recent]; rows = []
        for row in recent:
            for slot, entry in enumerate(row["batch"]):
                case = by_id[entry["case_id"]]
                if case["split"] != "train" or entry["domain"] != ("PDB" if slot < 3 else "MD"):
                    raise RuntimeError("Unexpected diagnostic training event")
                key = int(hashlib.sha256(f"full-cohort-coarse-v1:{seed}:{row['update']-1}:{slot}".encode()).hexdigest()[:8], 16)
                rows.append(dict(**entry, seed=key, draw_sha256=row["base_draws"][slot], update=row["update"], slot=slot))
        events[str(seed)] = rows; checkpoints[str(seed)] = {}
        for arm, path, prefix in [("parent", old["checkpoints"][str(seed)], ""),
            ("static_FM32768", str(FULL / f"seed{seed}/checkpoint_32768.pt"), "coarse_model."),
            ("static_FM131072", str(FULL / f"seed{seed}/last.pt"), "coarse_model."),
            ("PDB_replay32768", str(REPLAY / f"seed{seed}/repair/last.pt"), "coarse_model.")]:
            saved = torch.load(path, map_location="cpu", weights_only=False); state = saved["model"]
            if prefix: state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            checkpoints[str(seed)][arm] = dict(path=path, prefix=prefix, coarse_state_sha256=digest_state(state)); del saved
            paths.append(Path(path))
    p = {key: copy.deepcopy(old[key]) for key in ["normalization", "model", "feature_dimension", "ca_step_cap_A"]}
    p.update(seeds=[2026, 2027], arms=ARMS, checkpoints=checkpoints, PDB_probe_ids=old["PDB_probe_ids"][:4],
        training_performed=False, optimizer_updates=0, steps=16, samples=16, gradient_samples=4,
        torch_version=torch.__version__, numpy_version=np.__version__, CPU_threads=2,
        perturbation_draws=4, epsilons=[.01, .02], bridge_times=[.1, .5, .9], bridge_frames=8, bridge_draws=2,
        training_events=events, fixed_control_batch_updates=batches,
        hypotheses=["A more accurate numerical solver did not restore spread. Locate where the fixed learned flow contracts different initial noises.",
                    "Separate distance-observable mean bias from ensemble variance; restored noise sensitivity alone need not fix inaccurate folds.",
                    "Measure whether native-PDB, native-MD and parent-rehearsal training directions locally decrease or increase generated distance variance.",
                    "Compare native interpolation endpoint estimates at early/late time and existing shuffled-AA response as conditioning diagnostics."],
        observations="Frozen original parent, static-FM32768, static-FM131072 and PDB-replay32768, both seeds. Five seen MD proteins, four predeclared static TRAIN targets, two reused development proteins. Original same-seed Euler16 endpoints must replay saved outputs. Author output block centers are pooled only as an observational reference, not its internal coarse tokens.",
        spread="Mean population variance of long-range inter-center distances(index gap>=4) in A^2, plus proper-aligned residue-count-weighted center RMSD. Pure translation/rotation does not count as structural diversity. Intermediate noisy states are not decoded protein structures.",
        local_gradients="Full differentiable16-step coarse Euler rollout on four fixed Gaussian draws. Autograd gradient of generated distance-feature variance projected onto actual fixed-training-example native-PDB, native-MD or parent-replay FM gradients. Report cosine and -dot(g_spread,g_loss)/norm(g_loss). These are infinitesimal plain-SGD directions, not real AdamW step predictions or global causal proofs. No parameter update, optimizer, development optimization or weight scan.",
        gradient_events="Use exactly the last16 recorded minibatches through update32768 of the static-FM control:48 PDB and16 MD events per seed. Replay actual case IDs, Gaussian/time keys and draw hashes for all objectives/checkpoints. The observed MD protein frequencies are reported; this finite probe is not the full training gradient.",
        bridges="Diagnostic oracle: build x_t from a fixed native reference and noise, then x1_hat=x_t+(1-t)*v. This uses native coordinates and is not AA-only deployment. Full trajectory samples receive only AA features and noise.",
        scope="Read-only exploratory mechanism probes on already used TRAIN/development data. No new test/R3 coordinates, architecture/weight changes, optimization, model selection or claim that any single diagnostic proves the cause.",
        stop_rule="One fixed pass per seed over all four checkpoints and all cases. Report every time, perturbation size and direction. No training or automatic promotion; freeze a separate protocol before any later intervention.")
    cases = selected_cases(p, train, dev)
    needed = {c["sequence"] for c in cases} | {by_id[e["case_id"]]["sequence"] for es in events.values() for e in es}
    index = load(old["all_feature_index"]); p["features"] = {s: index[s] for s in sorted(needed)}
    bank = feature_bank(p); checks = []
    for seed in p["seeds"]:
        m = model(p, seed, "parent"); m.feature_bank = bank; teacher = copy.deepcopy(m).requires_grad_(False)
        expected = load(FULL / f"seed{seed}/initial.json")["coarse_sha256"]
        if digest_state(m.state_dict()) != expected: raise RuntimeError("Direct coarse loader differs from original parent")
        for e in p["training_events"][str(seed)]:
            loss, draw = event_loss(m, teacher, by_id[e["case_id"]], e["seed"], "native")
            if draw != e["draw_sha256"] or not torch.isfinite(loss): raise RuntimeError("Actual training draw replay failed")
        c = cases[0]; aa = encode_aa(c["sequence"]); key = component_seed(sample_key(seed, c["protein"], 0), "coarse")
        noise = torch.randn(((len(aa)+3)//4, 3), generator=torch.Generator().manual_seed(key))
        with torch.no_grad():
            torch.testing.assert_close(trajectory(m, aa, noise)[-1], project(m.sample_coarse(aa, key, 16), len(aa)), atol=0, rtol=0)
        checks.append(dict(seed=seed, direct_parent_state_exact=True, training_draws_replayed=64, path_sampler_exact=True, optimizer_updates=0))
    atomic_json(ROOT / "integration_check.json", dict(status="passed", checks=checks))
    p["dataset"] = dict(PDB_structures=len(train["PDB_train"]), PDB_unique_sequences=len({c["sequence"] for c in train["PDB_train"]}),
        MD_proteins=len(train["train"]), MD_frames={k:len(v) for k,v in train["train"].items()}, event_mix="3 static PDB events:1 MD-frame event",
        note="Native MD frames are not averaged per AA sequence. Static-PDB sequences each supply one available conformation, which is not an equilibrium-ensemble label.")
    paths += [REPLAY / "protocol.json", REPLAY / "review.json", REPLAY / "assessment.json", FULL / "protocol.json", FULL / "report.json",
        LABELS / "train_labels.pt", LABELS / "development_labels.pt", Path(old["all_feature_index"]), ROOT / "integration_check.json",
        Path("upstream/tests/test_coarse_coverage_mechanism.py"), Path("upstream/scripts/coarse_coverage_mechanism.sbatch")]
    # Freeze the actually loaded local source dependency closure, not unrelated
    # historical weights (e.g. the unused1.4B author generator).
    workspace = Path.cwd().resolve()
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file:
            q = Path(file).resolve()
            if q.suffix == ".py" and q.is_relative_to(workspace / "upstream"): paths.append(q)
    paths.append(Path(__file__))
    for entry in p["features"].values(): paths.append(Path(entry["path"]))
    for seed in p["seeds"]:
        paths.append(FULL / f"seed{seed}/trace.jsonl")
        for arm in ARMS:
            for name in dev["validation"]:
                for outlet in ["full", "scrambled_AA"]: paths.append(cache_folder(seed, arm) / name / (outlet + ".npz"))
        for name in dev["validation"]:
            for n in [25,5]: paths.append(REPLAY / f"seed{seed}/author/{name}_steps{n}_backbone.npz")
    p["source_sha256"] = {str(q): sha(q) for q in dict.fromkeys(paths)}
    atomic_json(ROOT / "protocol.json", p)
    print("COARSE_MECHANISM_READY", len(p["source_sha256"]), json.dumps(checks), flush=True)


@torch.no_grad()
def trajectory_probe(p, m, case, seed, arm, out):
    aa = encode_aa(case["sequence"]); n = len(aa); name = case["protein"]
    initial = [torch.randn(((n+3)//4, 3), generator=torch.Generator().manual_seed(
        component_seed(sample_key(seed, name, i), "coarse"))) for i in range(p["samples"])]
    paths = torch.stack([trajectory(m, aa, z, p["steps"]) for z in initial])
    if not torch.isfinite(paths).all(): raise RuntimeError("Nonfinite trajectory probe")
    snapshots = []
    for step in range(p["steps"]+1):
        c = paths[:, step]; matrix = weighted_rmsd_matrix(c.numpy(), c.numpy(), n)
        snapshots.append(dict(time=step/p["steps"], distance_variance_A2=float(distance_variance(c)),
            proper_aligned_center_spread_A=float(matrix[np.triu_indices(len(c),1)].mean())))
    endpoint = paths[:, -1]; reference = case["references"]
    stats = coarse_metrics(endpoint.numpy(), reference.numpy(), n, p["ca_step_cap_A"])
    stats.update(observable_moments(endpoint, reference))
    conditioning = None; replay_difference = None
    if case["split"] == "development":
        folder = cache_folder(seed, arm) / name
        with np.load(folder / "full.npz") as z: cached = z["coarse"].copy()[:p["samples"]]
        np.testing.assert_allclose(endpoint.numpy(), cached, atol=2e-4, rtol=2e-5)
        replay_difference = float(np.max(np.abs(endpoint.numpy()-cached)))
        with np.load(folder / "scrambled_AA.npz") as z: shuffled = torch.from_numpy(z["coarse"].copy()[:p["samples"]])
        contrast = distance_features(endpoint) - distance_features(shuffled)
        conditioning = dict(paired_shuffled_AA_distance_contrast_RMS_A=float(contrast.square().mean().sqrt()),
            unshuffled_noise_distance_SD_A=float(distance_variance(endpoint).sqrt()),
            limits="Shuffled AA is OOD. Response is sensitivity, not correct biological conditioning; a ratio alone would be confounded by noise collapse.")
    gains = []
    for i in range(p["perturbation_draws"]):
        direction = torch.randn(initial[i].shape, generator=torch.Generator().manual_seed(diagnostic_key(seed,name,"noise_direction",i)))
        for eps in p["epsilons"]:
            gains.append(dict(draw=i, epsilon=eps, endpoint_distance_response_A_per_unit_noise=response_gain(m,aa,initial[i],direction,eps,p["steps"])))
    bridges = []
    chosen = np.linspace(0,len(reference)-1,min(len(reference),p["bridge_frames"]),dtype=int)
    for t in p["bridge_times"]:
        errors=[]
        for j in chosen:
            target=reference[int(j)]
            for draw in range(p["bridge_draws"]):
                noise=torch.randn(target.shape,generator=torch.Generator().manual_seed(diagnostic_key(seed,name,f"bridge_{j}",draw)))
                pred=bridge_endpoint(m,aa,target,noise,t)
                errors.append(float(weighted_rmsd_matrix(pred.numpy()[None],target.numpy()[None],n)[0,0]))
        bridges.append(dict(time=t, oracle_bridge_endpoint_center_RMSD_A=float(np.mean(errors)),native_frames=len(chosen),noise_draws_per_frame=p["bridge_draws"]))
    np.savez_compressed(out/f"{name}_trajectories.npz", centers=paths.numpy(), reference_centers=reference.numpy(), times=np.arange(p["steps"]+1)/p["steps"])
    return dict(protein=name,split=case["split"],snapshots=snapshots,endpoint=stats,noise_response=gains,
        native_bridge=bridges,AA_conditioning_sensitivity=conditioning,maximum_cached_endpoint_roundoff_A=replay_difference)


def run(p,seed):
    train=torch.load(LABELS/"train_labels.pt",map_location="cpu",weights_only=True)
    dev=torch.load(LABELS/"development_labels.pt",map_location="cpu",weights_only=True)
    cases=selected_cases(p,train,dev); by_id=registry(train);bank=feature_bank(p)
    teacher=model(p,seed,"parent").eval().requires_grad_(False);teacher.feature_bank=bank
    teacher_hash=digest_state(teacher.state_dict());out=ROOT/f"seed{seed}";out.mkdir(exist_ok=True)
    if (out/"complete.json").exists():raise RuntimeError("Diagnostic already complete")
    pdb_events=[e for e in p["training_events"][str(seed)] if e["domain"]=="PDB"]
    md_events=[e for e in p["training_events"][str(seed)] if e["domain"]=="MD"]
    results=[];started=time.perf_counter()
    for arm in ARMS:
        m=model(p,seed,arm).eval();m.feature_bank=bank;original=digest_state(m.state_dict())
        folder=out/arm;folder.mkdir(exist_ok=True)
        rows=[]
        for c in cases:
            rows.append(trajectory_probe(p,m,c,seed,arm,folder))
            atomic_json(out/"progress.json",dict(seed=seed,arm=arm,phase="trajectories",case=c["protein"],completed_cases=len(rows),cases_per_arm=len(cases),seconds=time.perf_counter()-started))
        domains={};domain_meta={}
        for name,events,mode in [("PDB_native",pdb_events,"native"),("MD_native",md_events,"native"),("PDB_parent_replay",pdb_events,"parent_replay")]:
            domains[name],domain_meta[name]=mean_event_gradient(m,teacher,events,by_id,mode)
            domain_meta[name]["protein_event_counts"]=dict(Counter(e["protein"] for e in events))
        domains["mixture_static_FM"]=(3*domains["PDB_native"]+domains["MD_native"])/4
        domains["mixture_PDB_replay"]=(3*domains["PDB_parent_replay"]+domains["MD_native"])/4
        for i,c in enumerate(cases):
            aa=encode_aa(c["sequence"]); n=len(aa); endpoints=[]
            with same_forward_kernel():
                for draw in range(p["gradient_samples"]):
                    key=component_seed(sample_key(seed,c["protein"],draw),"coarse")
                    noise=torch.randn(((n+3)//4,3),generator=torch.Generator().manual_seed(key))
                    endpoints.append(trajectory(m,aa,noise,p["steps"])[-1])
                spread=distance_variance(torch.stack(endpoints));g=flatten_gradient(spread,parameters(m))
            rows[i]["local_training_directions"]=dict(spread_four_draws_A2=float(spread.detach()),
                gradients={name:gradient_projection(g,value) for name,value in domains.items()})
            atomic_json(out/"progress.json",dict(seed=seed,arm=arm,phase="gradient_projections",case=c["protein"],completed_cases=i+1,cases_per_arm=len(cases),seconds=time.perf_counter()-started))
        if digest_state(m.state_dict())!=original or digest_state(teacher.state_dict())!=teacher_hash:
            raise RuntimeError("Read-only probe changed weights")
        if any(v.grad is not None for v in m.parameters()) or any(v.grad is not None for v in teacher.parameters()):
            raise RuntimeError("Diagnostic left parameter gradient buffers")
        result=dict(arm=arm,rows=rows,training_domain_gradients=domain_meta,weights_unchanged=True,optimizer_updates=0)
        results.append(result);atomic_json(folder/"complete.json",result)
        print("COARSE_MECHANISM_ARM_COMPLETE",seed,arm,flush=True)
    verify(p)
    atomic_json(out/"complete.json",dict(status="complete",seed=seed,arms=results,seconds=time.perf_counter()-started,
        host=platform.node(),weights_unchanged=True,optimizer_updates=0,scope=p["scope"]))
    atomic_json(out/"progress.json",dict(seed=seed,phase="complete",arms=4,optimizer_updates=0))


def report(p):
    complete=[load(ROOT/f"seed{s}/complete.json") for s in p["seeds"]]
    rows=[]
    for d in complete:
        if not d["weights_unchanged"] or d["optimizer_updates"]!=0:raise RuntimeError("Non-read-only result")
        for arm in d["arms"]:
            for row in arm["rows"]:rows.append(dict(seed=d["seed"],arm=arm["arm"],**row))
    dev=torch.load(LABELS/"development_labels.pt",map_location="cpu",weights_only=True);author=[]
    for seed in p["seeds"]:
        for name,cases in dev["validation"].items():
            refs=[cases[int(i)] for i in np.linspace(0,len(cases)-1,min(128,len(cases)),dtype=int)]
            reference=torch.stack([c["coarse"] for c in refs])
            for steps in [25,5]:
                with np.load(REPLAY/f"seed{seed}/author/{name}_steps{steps}_backbone.npz") as z:c=torch.from_numpy(pool_backbones(z["backbone"]))
                author.append(dict(seed=seed,protein=name,author_steps=steps,**observable_moments(c,reference),
                    note="Block centers pooled from final author decoded coordinates, not its coarse tokens."))
    verify(p)
    atomic_json(ROOT/"report.json",dict(status="complete",rows=rows,author_output_observables=author,
        completed_runs=[{k:v for k,v in d.items() if k!="arms"} for d in complete],scope=p["scope"],training_performed=False,
        optimizer_updates=0,model_default_replaced=False,test_coordinates_opened=False,R3_opened=False))
    atomic_json(ROOT/"assessment.json",dict(status="complete_awaiting_review",scope=p["scope"],
        next="Review local gradient signs, noise responses, bridge errors and distance-mean bias together. Do not infer a unique cause from any one probe or start training automatically."))
    plot(rows)
    print("COARSE_MECHANISM_REPORT_COMPLETE",flush=True)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names=sorted({r["protein"] for r in rows if r["split"]=="development"})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    colors={"parent":"#555555","static_FM32768":"#d68a33","static_FM131072":"#b94b4b","PDB_replay32768":"#2e7caf"}
    for column,name in enumerate(names):
        ax=axes[0,column]
        for arm in ARMS:
            selected=[r for r in rows if r["protein"]==name and r["arm"]==arm]
            y=np.mean([[v["distance_variance_A2"] for v in r["snapshots"]] for r in selected],0)
            ax.plot(np.arange(17)/16,y,label=arm,color=colors[arm])
        native=selected[0]["endpoint"]["reference_distance_variance_A2"]
        ax.axhline(native,color="black",ls="--",label="Native reference")
        ax.set_title(name+": distance-observable spread");ax.set_xlabel("Flow time(noisy states)");ax.set_ylabel("Distance variance(A²)");ax.grid(alpha=.2)
        if column==0:ax.legend(fontsize=8)
        ax=axes[1,column]
        for arm in ARMS:
            selected=[r for r in rows if r["protein"]==name and r["arm"]==arm]
            y=np.mean([[v["oracle_bridge_endpoint_center_RMSD_A"] for v in r["native_bridge"]] for r in selected],0)
            ax.plot([.1,.5,.9],y,marker="o",label=arm,color=colors[arm])
        ax.set_title(name+": native-interpolation diagnostic");ax.set_xlabel("Bridge time(native structure used)");ax.set_ylabel("Endpoint center RMSD(A)");ax.grid(alpha=.2)
    fig.suptitle("Frozen coarse-flow mechanism probes ·2 reused development cases ·2 seeds\nIntermediate states and native bridges are diagnostics, not generated protein structures")
    dest=Path("paper/figures/coarse_coverage_mechanism_v1");dest.mkdir(parents=True,exist_ok=True)
    for ext in ["pdf","png","svg"]:fig.savefig(dest/("Coarse_Coverage_Mechanism."+ext),dpi=170)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["prepare","run","report"],required=True)
    parser.add_argument("--seed",type=int,choices=[2026,2027]);args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=="prepare":prepare()
    else:
        protocol=load(ROOT/"protocol.json");verify(protocol)
        if args.stage=="run":run(protocol,args.seed)
        else:report(protocol)
