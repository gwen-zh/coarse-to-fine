"""Same structure targets and flow budget; capacity-matched sequence conditions."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from joint_sequence_predictor import AA_ALPHABET, encode_aa
import run_multisequence_ensemble_flow as engine
from sequence_feature_ensemble_flow import SequenceFeatureEnsembleFlow

RUN = Path("artifacts/sequence_condition_probe_v1")
PARENT = Path("artifacts/multisequence_ensemble_flow_v1")
OLD_FEATURES = Path("artifacts/development/coordinate_flow_v1")


def prepare_requests():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / "feature_requests.json").exists():
        raise RuntimeError("Feature requests already frozen")
    p = engine.protocol()
    labels = torch.load(PARENT / "labels.pt", map_location="cpu", weights_only=True)
    feature_protocol = json.loads((OLD_FEATURES / "protocol.json").read_text())
    corpus = Path(feature_protocol["corpus_manifest"])
    if sha(corpus) != feature_protocol["corpus_manifest_sha256"]:
        raise RuntimeError("Existing sequence-feature corpus changed")
    manifest = {r["id"]: r for r in json.loads(corpus.read_text())["targets"]}
    digests = {}
    for chunk in [0, 1]:
        entry = json.loads((OLD_FEATURES / f"sequence_chunk{chunk}.json").read_text())
        if entry["status"] != "complete":
            raise RuntimeError("Existing pure-sequence cache is incomplete")
        digests.update({r["id"]: r["sha256"] for r in entry["targets"]})
    pdb = []
    for case in labels["PDB_train"]:
        name = case["protein"]
        if manifest[name]["sequence"] != case["sequence"] or manifest[name]["subset"] != "train":
            raise RuntimeError("PDB feature identity or split mismatch")
        path = OLD_FEATURES / "sequence_features" / (name + ".npz")
        if sha(path) != digests[name]:
            raise RuntimeError("Existing feature cache changed: " + name)
        pdb.append(dict(id=name, sequence=case["sequence"], length=len(case["sequence"]),
                        path=str(path), sha256=digests[name]))
    requests = []
    for split in ["train", "validation"]:
        for name, cases in labels[split].items():
            requests.append(dict(id=name, sequence=cases[0]["sequence"], variant="native"))
            if split == "validation":
                aa = encode_aa(cases[0]["sequence"])
                shuffled = aa[torch.randperm(len(aa), generator=torch.Generator().manual_seed(719))]
                requests.append(dict(id=name + "_shuffled", sequence="".join(AA_ALPHABET[i] for i in shuffled.tolist()), variant="shuffled_sensitivity_control"))
    anchor = sorted(pdb, key=lambda r: r["id"])[0]
    code = [Path(__file__), Path("upstream/experiments/cache_multisequence_sequence_features.py"),
        Path("upstream/scripts/sequence_condition_features.sbatch"),
        Path("upstream/experiments/sequence_feature_ensemble_flow.py"),
        Path("upstream/tests/test_sequence_feature_ensemble_flow.py"),
        PARENT / "protocol.json", PARENT / "labels.pt", corpus,
        OLD_FEATURES / "protocol.json", OLD_FEATURES / "sequence_chunk0.json", OLD_FEATURES / "sequence_chunk1.json"]
    vendor = Path("upstream/Hierarchical_VQ_VAE/esm")
    code.extend(vendor / name for name in ["models/esm3.py", "pretrained.py",
        "tokenization/sequence_tokenizer.py", "utils/encoding.py", "utils/constants/esm3.py"])
    atomic_json(RUN / "feature_requests.json", dict(requests=requests, anchor=anchor,
        PDB_features=pdb, expected_feature_dimension=1536,
        compatibility_gate=dict(relative_RMSE_max=.01, mean_residue_cosine_min=.999),
        scope="GPU extraction receives only AA strings and one old feature anchor. No coordinates, structure tokens or native coarse are supplied to ESM3. Two reserved test sequences are excluded.",
        source_sha256={str(path): sha(path) for path in code}))
    print("SEQUENCE_FEATURE_REQUESTS_READY", len(requests), len(pdb), flush=True)


def requests():
    p = json.loads((RUN / "feature_requests.json").read_text())
    for name, digest in p["source_sha256"].items():
        if sha(name) != digest:
            raise RuntimeError("Changed feature request input: " + name)
    return p


def prepare():
    r = requests()
    if (RUN / "protocol.json").exists():
        raise RuntimeError("Condition-probe protocol already frozen")
    done = json.loads((RUN / "feature_extraction.json").read_text())
    if done["status"] != "complete" or not done["anchor_compatibility_passed"]:
        raise RuntimeError("New sequence features failed compatibility")
    parent = engine.protocol()
    index = {}
    feature_paths = []
    for row in r["PDB_features"] + done["features"]:
        path = Path(row["path"])
        if sha(path) != row["sha256"]:
            raise RuntimeError("Changed sequence feature: " + str(path))
        with np.load(path, allow_pickle=False) as z:
            h = z["h"]
            if h.shape != (len(row["sequence"]), 1536) or not np.isfinite(h).all():
                raise RuntimeError("Feature shape/finite check failed")
        index[row["sequence"]] = dict(path=str(path), sha256=row["sha256"], id=row["id"])
        feature_paths.append(path)
    atomic_json(RUN / "feature_index.json", index)
    shutil.copyfile(PARENT / "labels.pt", RUN / "labels.pt")
    p = copy.deepcopy(parent)
    p.update(arms={"AA_lookup": True, "ESM3_context": True},
        feature_dimension=1536, feature_index=str(RUN / "feature_index.json"),
        parent_trial=str(PARENT / "protocol.json"),
        training="Both arms use the identical MD_plus_PDB four-slot recipe and 4096 fresh-start updates. Only conditioning differs: frozen random per-AA lookup versus frozen contextual ESM3 AA-only features.",
        matching="Identical trainable parameter count, initialization of every trainable weight, Transformer width/depth, structure labels, train-only normalization, minibatch examples, per-slot random noise/time draws and solver NFEs. The 1536->128 sequence projection is trained in BOTH arms.",
        feature_cost="Frozen 1.4B ESM3 sequence extraction is additional offline compute; small-generator parameter/NFE matching does not equalize total pretraining or end-to-end inference cost. Report extraction metadata separately.",
        scope="Two new development proteins relative to our training corpus; their absence from ESM3 pretraining is unknown. This is not a test of sequences unseen by the foundation model.",
        stop_rule="Final fixed endpoints, no epoch selection, no automatic extension or test/R3 opening. Keep our structural encoder/decoder and fine book frozen. Continuous generation does not quantize fine.")
    sources = [Path(__file__), Path("upstream/experiments/sequence_feature_ensemble_flow.py"),
        Path("upstream/tests/test_sequence_feature_ensemble_flow.py"),
        Path("upstream/scripts/sequence_condition_probe.sbatch"),
        RUN / "feature_requests.json", RUN / "feature_extraction.json", RUN / "feature_index.json",
        RUN / "labels.pt", *feature_paths]
    p["source_sha256"].update({str(path): sha(path) for path in sources})
    atomic_json(RUN / "protocol.json", p)
    print("SEQUENCE_CONDITION_PROTOCOL_READY", len(index), p["updates"], flush=True)


def make_model(p, mode):
    model = SequenceFeatureEnsembleFlow(**p["normalization"], **p["model"],
        coarse_dim=3, feature_mode=mode, feature_dim=p["feature_dimension"])
    if mode == "ESM3_context":
        bank = {}
        for sequence, row in json.loads(Path(p["feature_index"]).read_text()).items():
            with np.load(row["path"], allow_pickle=False) as z:
                bank[sequence] = torch.from_numpy(z["h"].copy()).float()
        model.attach_feature_bank(bank)
    return model


def configure(mode):
    engine.RUN = RUN
    engine.make_model = lambda p: make_model(p, mode)


def report(p):
    engine.report(p)
    result = json.loads((RUN / "report.json").read_text())
    counts = {r["complete"]["parameters"] for r in result["rows"]}
    if len(counts) != 1:
        raise RuntimeError("Conditioning arms differ in trainable parameter count")
    for seed in p["seeds"]:
        a, b = [json.loads((RUN / arm / f"seed{seed}/trace.json").read_text()) for arm in p["arms"]]
        if [r["batch"] for r in a] != [r["batch"] for r in b] or [r["draws"] for r in a] != [r["draws"] for r in b]:
            raise RuntimeError("Conditioning comparison changed examples or noise draws")
    result.update(all_examples_and_noise_draws_matched=True, trainable_parameters_equal=True,
        scope=p["scope"], feature_extraction=json.loads((RUN / "feature_extraction.json").read_text()))
    atomic_json(RUN / "report.json", result)
    print("SEQUENCE_CONDITION_REPORT_COMPLETE", list(counts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["requests", "prepare", "train", "report"], required=True)
    parser.add_argument("--arm", choices=["AA_lookup", "ESM3_context"])
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "requests":
        prepare_requests()
    elif args.stage == "prepare":
        prepare()
    else:
        configure(args.arm)
        p = engine.protocol()
        if args.stage == "train":
            if args.arm not in p["arms"] or args.seed not in p["seeds"]:
                raise RuntimeError("Undeclared condition probe arm")
            engine.train(p, args.arm, args.seed)
        else:
            report(p)
