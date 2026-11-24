"""Matched staged PDB pretraining versus interleaved generator training.

Same frozen contextual sequence features, structure representation, exact
training-event multiset, noise/time draws, initialization and LR per update.
The intervention includes temporal ordering and minibatch domain composition.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from compact_training_trace import CompactTrainingTrace, restore_trace_prefix
from compare_two_level_ae_author import metrics
from diagnostic_io import atomic_json
from generator_curriculum import training_schedule, event_pools, resolve_event, event_seed, learning_rate
from hierarchical_pipeline_io import sha, save_checkpoint, restore_checkpoint
from joint_sequence_predictor import encode_aa
from run_sequence_condition_probe import make_model
from run_sequence_multiscale_ensemble import load_codec
from run_multisequence_ensemble_flow import evaluate
from sequence_multiscale_ensemble import paired_flow_loss, sample_levels

RUN = Path("artifacts/generator_curriculum_v1")
PARENT = Path("artifacts/sequence_condition_probe_v1")
ARMS = ["interleaved", "PDB_then_MD"]


def verify(p):
    for name, digest in p["source_sha256"].items():
        if sha(name) != digest:
            raise RuntimeError("Changed curriculum input: " + name)


def protocol():
    p = json.loads((RUN / "protocol.json").read_text())
    verify(p)
    return p


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / "protocol.json").exists():
        raise RuntimeError("Curriculum protocol already frozen")
    parent = json.loads((PARENT / "protocol.json").read_text())
    verify(parent)
    result = json.loads((PARENT / "report.json").read_text())
    if not result["all_examples_and_noise_draws_matched"]:
        raise RuntimeError("Prior conditioning control did not verify")
    pretrain, adapt = 8192, 4096
    schedules = {arm: training_schedule(arm, pretrain, adapt) for arm in ARMS}
    events = {arm: [event for row in rows for event in row] for arm, rows in schedules.items()}
    if Counter(events[ARMS[0]]) != Counter(events[ARMS[1]]):
        raise RuntimeError("Curricula do not use the same events")
    counts = Counter(domain for domain, _ in events[ARMS[0]])
    if counts != Counter(PDB=36864, MD=12288):
        raise RuntimeError("Unexpected training budget")
    atomic_json(RUN / "training_plan.json", dict(
        updates=pretrain+adapt, PDB_events=counts["PDB"], MD_events=counts["MD"],
        exact_same_events=True, ordering_different=True,
        order_sha256={arm: hashlib.sha256(json.dumps(rows).encode()).hexdigest()
                      for arm, rows in schedules.items()},
        schedules={"PDB_then_MD":"8192 PDB-only updates (4 PDB each), then 4096 updates with 3 MD + 1 PDB",
                   "interleaved":"12288 updates with 3 PDB + 1 MD"}))
    shutil.copyfile(PARENT / "labels.pt", RUN / "labels.pt")
    groups = torch.load(RUN / "labels.pt", map_location="cpu", weights_only=True)
    probe = sorted(groups["PDB_train"], key=lambda c: hashlib.sha256(
        ("curriculum-PDB-probe-v1:"+c["id"]).encode()).hexdigest())[:8]
    p = copy.deepcopy(parent)
    p.update(arms=ARMS, feature_mode="ESM3_context", parent_trial=str(PARENT / "protocol.json"),
        pretrain_updates=pretrain, adapt_updates=adapt, updates=pretrain+adapt,
        pretrain_lr=3e-4, pretrain_min_lr=3e-5, adapt_lr=1e-4, adapt_min_lr=1e-5,
        pretrain_probe_ids=[c["id"] for c in probe], pretrain_probe_samples=4,
        training="Fresh models. Per-seed identical PDB and balanced MD event pools; each unique event gets a fixed noise/time seed independent of its optimizer position. Both curricula see all the same 36864 PDB and 12288 MD examples with replacement.",
        matching="Same ESM3 features, model initialization, train-only normalization, frozen structural maps, 12288 optimizer updates, all event identities/noise draws, and LR at each optimizer step. Total example shapes and flow-call counts match; data order and within-batch domain composition differ. Measured runtime is reported.",
        learning_rate_schedule="Both arms: cosine 3e-4->3e-5 for 8192 steps, then cosine 1e-4->1e-5 for 4096. Adam state is carried across the boundary in both arms; neither optimizer is reset.",
        feature_cost="Both curricula reuse the same frozen ESM3 features. Total training-example shapes and per-example generator flow calls are matched; measured runtime can still differ because batching/order and system behavior differ.",
        diagnostics="The fixed 8192-step checkpoint is saved in both arms. After training, an eight-PDB-training-protein/four-sample probe measures training-set learning at that endpoint. It is not validation or checkpoint selection. Final new-sequence evaluation is unchanged.",
        scope="Two repeatedly used new-sequence development proteins, not independent confirmation. ESM3 pretraining exposure is unknown. No reserved test or R3 coordinate use.",
        stop_rule="Only the fixed 8192 and 12288 endpoints; no checkpoint selection, budget extension, sampling sweep, geometry rejection or automatic model replacement. Report both seeds and both curricula regardless of outcome.")
    paths = [Path(__file__), Path("upstream/experiments/generator_curriculum.py"),
        Path("upstream/tests/test_generator_curriculum.py"),
        Path("upstream/scripts/generator_curriculum.sbatch"),
        Path("upstream/experiments/compact_training_trace.py"), PARENT / "protocol.json",
        PARENT / "report.json", PARENT / "assessment.json", RUN / "training_plan.json", RUN / "labels.pt"]
    p["source_sha256"].update({str(path): sha(path) for path in paths})
    atomic_json(RUN / "protocol.json", p)
    print("GENERATOR_CURRICULUM_READY", counts, p["updates"], flush=True)


def initial_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train(p, arm, seed):
    groups = torch.load(RUN / "labels.pt", map_location="cpu", weights_only=True)
    codec = load_codec(p, "centers3")
    torch.manual_seed(seed)
    model = make_model(p, "ESM3_context")
    opt = torch.optim.AdamW(model.parameters(), lr=p["pretrain_lr"], weight_decay=p["weight_decay"])
    rng = np.random.default_rng(seed)
    pools = event_pools(groups, seed, p["pretrain_updates"], p["adapt_updates"])
    schedule = training_schedule(arm, p["pretrain_updates"], p["adapt_updates"])
    out = RUN / arm / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        raise RuntimeError("Curriculum arm already complete")
    start = 0
    digest = sha(RUN / "protocol.json")
    if (out / "last.pt").exists():
        saved = restore_checkpoint(out / "last.pt", model, opt, rng, torch.device("cpu"))
        if saved["protocol_sha256"] != digest:
            raise RuntimeError("Resume protocol mismatch")
        start = saved["updates"]
    else:
        atomic_json(out / "initial.json", dict(seed=seed, state_sha256=initial_digest(model),
            trainable_parameters=sum(v.numel() for v in model.parameters())))
    restore_trace_prefix(out, start)
    rows = [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines()]
    write_trace = CompactTrainingTrace(p["updates"], start)
    started = time.perf_counter()
    for step in range(start, p["updates"]):
        model.train()
        events = schedule[step]
        cases = [resolve_event(groups, pools, event) for event in events]
        rate = learning_rate(p, step)
        for group in opt.param_groups:
            group["lr"] = rate
        opt.zero_grad(set_to_none=True)
        value, draws = 0., []
        for event, case in zip(events, cases):
            generator = torch.Generator().manual_seed(event_seed(seed, *event))
            loss, terms = paired_flow_loss(model, codec, encode_aa(case["sequence"]),
                case["target"], case["coarse"], case["fine_continuous"], generator, 0.)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite curriculum loss")
            (loss/4).backward()
            value += float(loss.detach())/4
            draws.append(terms["draw_sha256"])
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), p["gradient_clip"])
        if not torch.isfinite(grad):
            raise RuntimeError("Nonfinite curriculum gradient")
        opt.step()
        rows.append(dict(update=step+1, loss=value, lr=rate,
            events=[f"{domain}:{index}" for domain,index in events],
            batch=[c["id"] for c in cases], draws=draws))
        if (step+1)%128 == 0 or step+1 == p["updates"]:
            write_trace(out / "trace.json", rows)
            progress = dict(arm=arm, seed=seed, updates=step+1, total=p["updates"],
                phase="pretrain_budget" if step < p["pretrain_updates"] else "adapt_budget",
                seconds=time.perf_counter()-started, loss=value)
            atomic_json(out / "progress.json", progress)
            print("CURRICULUM_TRAIN", json.dumps(progress), flush=True)
        if (step+1)%512 == 0 or step+1 == p["updates"]:
            save_checkpoint(out / "last.pt", model, opt, rng, updates=step+1,
                model_config=model.config, protocol_sha256=digest, arm=arm, seed=seed)
        if step+1 == p["pretrain_updates"]:
            shutil.copyfile(out / "last.pt", out / "pretrain_endpoint.pt")
    evaluate(p, model, codec, groups, out, seed)
    verify(p)
    atomic_json(out / "complete.json", dict(status="complete", arm=arm, seed=seed,
        updates=p["updates"], seconds=time.perf_counter()-started,
        parameters=sum(v.numel() for v in model.parameters()), frozen_representation=True))


@torch.no_grad()
def pretrain_probe(p, arm, seed):
    out = RUN / arm / f"seed{seed}"
    if (out / "pretrain_probe.json").exists():
        return
    model = make_model(p, "ESM3_context").eval()
    saved = torch.load(out / "pretrain_endpoint.pt", map_location="cpu", weights_only=False)
    if saved["updates"] != p["pretrain_updates"] or saved["protocol_sha256"] != sha(RUN / "protocol.json"):
        raise RuntimeError("Incorrect fixed pretraining endpoint")
    model.load_state_dict(saved["model"])
    groups = torch.load(RUN / "labels.pt", map_location="cpu", weights_only=True)
    cases = {case["id"]:case for case in groups["PDB_train"]}
    codec = load_codec(p, "centers3")
    records = []
    for name in p["pretrain_probe_ids"]:
        case = cases[name]
        for index in range(p["pretrain_probe_samples"]):
            key = int(hashlib.sha256(f"curriculum-probe-v1:{seed}:{name}:{index}".encode()).hexdigest()[:8],16)
            c,f = sample_levels(model, codec, encode_aa(case["sequence"]), key, p["coarse_nfe"], p["fine_nfe"])
            bb = codec.decode(c,f)
            records.append(dict(id=name, sample=index, metrics=metrics(bb.numpy(),case["target"].numpy())))
    atomic_json(out / "pretrain_probe.json", dict(status="complete", records=records,
        mean_metrics={key:float(np.mean([r["metrics"][key] for r in records])) for key in records[0]["metrics"]},
        scope="Fixed training-set probe at 8192 updates; not validation or model selection."))


def report(p):
    rows = []
    for seed in p["seeds"]:
        maps, initial, rates = [], [], []
        for arm in p["arms"]:
            out = RUN / arm / f"seed{seed}"
            complete = json.loads((out / "complete.json").read_text())
            trace = json.loads((out / "trace.json").read_text())
            events = {event:(case,draw) for row in trace for event,case,draw in zip(row["events"],row["batch"],row["draws"])}
            if len(events) != 4*p["updates"]:
                raise RuntimeError("A training event was omitted or duplicated")
            maps.append(events)
            initial.append(json.loads((out / "initial.json").read_text())["state_sha256"])
            rates.append([row["lr"] for row in trace])
            pretrain_probe(p,arm,seed)
            rows.append(dict(arm=arm,seed=seed,complete=complete,
                evaluation=json.loads((out / "evaluation.json").read_text()),
                pretrain_probe=json.loads((out / "pretrain_probe.json").read_text())))
        if maps[0] != maps[1] or initial[0] != initial[1] or rates[0] != rates[1]:
            raise RuntimeError("Curriculum matching failed")
    atomic_json(RUN / "report.json", dict(status="complete",rows=rows,
        exact_event_multisets_and_noise_verified=True,initial_states_and_LR_verified=True,
        test_coordinates_opened=False,R3_opened=False,scope=p["scope"],automatic_adoption=False))
    print("GENERATOR_CURRICULUM_COMPLETE",len(rows),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["prepare","train","report"],required=True)
    parser.add_argument("--arm",choices=ARMS)
    parser.add_argument("--seed",type=int,default=2026)
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare()
    else:
        p=protocol()
        if args.stage == "train":
            if args.arm not in p["arms"] or args.seed not in p["seeds"]:
                raise RuntimeError("Undeclared curriculum arm")
            train(p,args.arm,args.seed)
        else:
            report(p)
