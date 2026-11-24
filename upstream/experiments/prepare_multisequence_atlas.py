"""Bounded native-MD corpus selection, independent of generated structure quality.

Four new train proteins and two development proteins are downloaded. Two
prospective test proteins are fixed by sequence metadata but their coordinates
are not downloaded/opened. The old single-protein pilot is separately listed.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import re
import time
import zipfile

from Bio.Align import PairwiseAligner
import requests

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha

RUN = Path("artifacts/multisequence_atlas_v1")
COUNTS = {"train": 4, "val": 2, "test": 2}
CAP = 256 * 1024 * 1024
ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")


def sequence_identity(aligner, first, second):
    if min(len(first), len(second)) / max(len(first), len(second)) < .4:
        return 0.
    return aligner.align(first, second)[0].counts().identities / max(len(first), len(second))


def select():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / "selection.json").exists():
        raise RuntimeError("Selection already frozen; do not reselect based on model results")
    prior_paths = [Path("artifacts/pdb_training_v2/manifest.json"),
                   Path("artifacts/development/manifest.json"),
                   Path("artifacts/atlas_md_pilot_v1/selection.json")]
    previous = json.loads(prior_paths[0].read_text())["targets"] + json.loads(prior_paths[1].read_text())["all_groups"]
    old_md = json.loads(prior_paths[2].read_text())["selected"]
    previous.append(dict(id=old_md["name"], sequence=old_md["sequence"]))
    seen = set()
    previous = [r for r in previous if not (r["sequence"] in seen or seen.add(r["sequence"]))]
    aligner = PairwiseAligner(mode="global", match_score=2, mismatch_score=-1,
                             open_gap_score=-3, extend_gap_score=-.5)
    selected, screen, sources = [], [], list(prior_paths)
    for split in ["train", "val", "test"]:
        url = f"https://raw.githubusercontent.com/bjing2016/alphaflow/master/splits/atlas_{split}.csv"
        destination = RUN / f"atlas_{split}.csv"
        if not destination.exists():
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            destination.write_bytes(response.content)
        sources.append(destination)
        rows = sorted(csv.DictReader(io.StringIO(destination.read_text())), key=lambda r: r["name"])
        chosen = 0
        for row in rows:
            sequence = row["seqres"]
            if not 64 <= len(sequence) <= 128 or set(sequence) - ALPHABET:
                continue
            related = previous + [dict(id=r["name"], sequence=r["sequence"]) for r in selected]
            best = dict(id=None, identity=0.)
            for reference in related:
                identity = sequence_identity(aligner, sequence, reference["sequence"])
                if identity > best["identity"]:
                    best = dict(id=reference["id"], identity=identity)
                if identity >= .4:
                    break
            entry = dict(name=row["name"], official_split=split, length=len(sequence),
                         closest_screened_sequence=best)
            if best["identity"] >= .4:
                screen.append(dict(entry, decision="exclude_global_identity_at_least_40_percent"))
                continue
            archive_url = f"https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{row['name']}/{row['name']}_protein.zip"
            response = requests.head(archive_url, timeout=30, allow_redirects=True)
            size = int(response.headers.get("Content-Length", -1))
            entry.update(url=archive_url, expected_bytes=size, status=response.status_code)
            if response.status_code != 200 or not 0 < size <= CAP:
                screen.append(dict(entry, decision="exclude_unavailable_or_size_cap"))
                continue
            selected.append(dict(entry, sequence=sequence))
            screen.append(dict(entry, decision="selected_by_sequence_and_availability_only"))
            chosen += 1
            print("ATLAS_SELECTED", split, row["name"], len(sequence), size, flush=True)
            if chosen == COUNTS[split]:
                break
        if chosen != COUNTS[split]:
            atomic_json(RUN / "incomplete_selection.json", dict(selected=selected, screen=screen, missing_split=split))
            raise RuntimeError("Insufficient eligible proteins in " + split)
    sources.extend([Path(__file__), Path("upstream/scripts/multisequence_atlas.sbatch"),
                    Path("upstream/tests/test_multisequence_atlas.py")])
    selection = dict(version=1, selected=selected, screen=screen,
        previous_pilot=dict(name=old_md["name"], role="additional existing training protein; previous development/test usage remains recorded"),
        selection_rule="First lexicographic official entries of each split, 64-128 standard residues, all screened global identities below 40%, known archive size <=256MiB. No geometry, model score, or ensemble spread selection.",
        sequence_screen="Global alignment identities / max sequence length; match2 mismatch-1 gap-open-3 gap-extend-.5. Screening includes local PDB representation train/development and prior benchmarks, the old MD sequence, and all newly selected proteins. This does not establish CATH/fold separation or rule out local-domain homology.",
        download_roles="Download train/val archives only. Extract topology and R1/R2 only; R3 remains compressed. Test sequences are metadata only, no test coordinates downloaded.",
        planned_training="New training proteins: R1+R2. Development: wholly different val proteins, R1+R2. Existing 1b2s_E remains an explicitly reused diagnostic. Future test is reserved for a separately frozen final model, not this selection run.",
        source_sha256={str(p): sha(p) for p in sources})
    atomic_json(RUN / "selection.json", selection)
    print("MULTISEQUENCE_ATLAS_SELECTION_FROZEN", len(selected), flush=True)


def selection():
    p = json.loads((RUN / "selection.json").read_text())
    for name, digest in p["source_sha256"].items():
        if sha(name) != digest:
            raise RuntimeError("Changed data-selection source: " + name)
    return p


def download(p, index):
    rows = [r for r in p["selected"] if r["official_split"] != "test"]
    if not 0 <= index < len(rows):
        raise ValueError("Invalid download index")
    row = rows[index]
    out = RUN / row["name"]
    out.mkdir(exist_ok=True)
    if (out / "complete.json").exists():
        raise RuntimeError("Download already complete")
    archive = out / (row["name"] + "_protein.zip")
    if not archive.exists():
        partial = archive.with_suffix(".part")
        received = 0
        started = time.perf_counter()
        with requests.get(row["url"], stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for block in response.iter_content(1024 * 1024):
                    if block:
                        received += len(block)
                        if received > CAP:
                            raise RuntimeError("Archive exceeded the declared cap")
                        stream.write(block)
        if received != row["expected_bytes"]:
            raise RuntimeError("Archive length changed; stop instead of silently replacing a protein")
        partial.replace(archive)
        print("ATLAS_ARCHIVE_DOWNLOADED", row["name"], received, time.perf_counter() - started, flush=True)
    if archive.stat().st_size != row["expected_bytes"]:
        raise RuntimeError("Existing archive length differs from the frozen selection")
    extracted, sealed = [], []
    with zipfile.ZipFile(archive) as bundle:
        if sum(z.file_size for z in bundle.infolist()) > 4 * CAP:
            raise RuntimeError("Unexpected expanded archive size")
        for member in bundle.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError("Unsafe archive member")
            if member.is_dir():
                continue
            if re.search(r"(?:^|_)R3(?:_|\.|$)", path.name, re.I):
                sealed.append(member.filename)
                continue
            if path.suffix.lower() not in {".pdb", ".xtc", ".txt", ".mdp"} and not path.name.upper().startswith("README"):
                continue
            destination = out / "raw" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                with bundle.open(member) as src, destination.open("wb") as dst:
                    while block := src.read(1024 * 1024):
                        dst.write(block)
            extracted.append(dict(path=str(destination), sha256=sha(destination), bytes=destination.stat().st_size))
    topologies = [r for r in extracted if r["path"].endswith(".pdb")]
    trajectories = [r for r in extracted if r["path"].endswith(".xtc")]
    if len(topologies) != 1 or len(trajectories) != 2:
        raise RuntimeError("Expected one topology and exactly R1/R2 trajectories")
    atomic_json(out / "complete.json", dict(status="complete", name=row["name"],
        official_split=row["official_split"], sequence=row["sequence"], url=row["url"],
        archive_sha256=sha(archive), extracted=extracted, R3_members_not_opened=sealed,
        topology=topologies[0]["path"], trajectories=[r["path"] for r in trajectories],
        selection_sha256=sha(RUN / "selection.json"), test_coordinates_downloaded=False))


def report(p):
    rows = [json.loads((RUN / r["name"] / "complete.json").read_text())
            for r in p["selected"] if r["official_split"] != "test"]
    atomic_json(RUN / "report.json", dict(status="download_complete", proteins=rows,
        metadata_only_test_proteins=[r["name"] for r in p["selected"] if r["official_split"] == "test"],
        training_started=False, scope="New native-MD inputs and sequence split prepared; no generated-quality or unseen-protein results yet."))
    print("MULTISEQUENCE_ATLAS_DOWNLOAD_COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["select", "download", "report"], required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.stage == "select":
        select()
    elif args.stage == "download":
        download(selection(), args.index)
    else:
        report(selection())
