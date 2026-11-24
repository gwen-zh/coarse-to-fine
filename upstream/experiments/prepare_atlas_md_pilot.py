"""Select and download a bounded official ATLAS MD pilot before model evaluation.

Selection depends on sequence, public training split, and archive size only.
No existing trajectory is silently relabelled as native MD.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import time
import zipfile
import requests
from Bio.Align import PairwiseAligner
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha

RUN = Path('artifacts/atlas_md_pilot_v1')
SPLIT_URL = 'https://raw.githubusercontent.com/bjing2016/alphaflow/master/splits/atlas_train.csv'
DOWNLOAD_CODE = 'https://raw.githubusercontent.com/IDEA-XL/BioKinema/main/scripts/data_prep/download_atlas.sh'
MD_PARAMETERS_URL = 'https://www.dsimb.inserm.fr/ATLAS/api/MD_parameters'
CAP_BYTES = 256 * 1024 * 1024


def fetch_metadata(url, path):
    if path.exists():
        return path.read_bytes()
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    path.write_bytes(response.content)
    return response.content


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    split_path = RUN / 'atlas_train.csv'
    rows = sorted(csv.DictReader(io.StringIO(fetch_metadata(SPLIT_URL, split_path).decode())), key=lambda r: r['name'])
    code_path = RUN / 'published_download_atlas.sh.txt'
    fetch_metadata(DOWNLOAD_CODE, code_path)
    prior_paths = [Path('artifacts/pdb_training_v2/manifest.json'), Path('artifacts/development/manifest.json')]
    selection_path = RUN / 'selection.json'
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        for path, digest in selection['source_sha256'].items():
            if sha(path) != digest:
                raise RuntimeError('Changed selection source: ' + path)
    else:
        previous = json.loads(prior_paths[0].read_text())['targets'] + json.loads(prior_paths[1].read_text())['all_groups']
        aligner = PairwiseAligner(mode='global', match_score=2, mismatch_score=-1, open_gap_score=-3, extend_gap_score=-.5)
        screen = []
        chosen = None
        for row in rows:
            sequence = row['seqres']
            if not 64 <= len(sequence) <= 128 or set(sequence) - set('ACDEFGHIKLMNPQRSTVWY'):
                continue
            closest = dict(identity=0., id=None)
            for old in previous:
                other = old['sequence']
                if min(len(sequence), len(other)) / max(len(sequence), len(other)) < .8:
                    continue
                identity = int(aligner.align(sequence, other)[0].counts().identities) / max(len(sequence), len(other))
                if identity > closest['identity']:
                    closest = dict(identity=identity, id=old['id'])
            entry = dict(name=row['name'], length=len(sequence), closest_prior=closest)
            if closest['identity'] >= .8:
                screen.append(dict(entry, decision='exclude_existing_sequence_overlap'))
                continue
            url = f'https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{row["name"]}/{row["name"]}_protein.zip'
            response = requests.head(url, timeout=30, allow_redirects=True)
            size = int(response.headers.get('Content-Length', -1))
            entry.update(status=response.status_code, bytes=size, url=url)
            if response.status_code != 200 or size < 1 or size > CAP_BYTES:
                screen.append(dict(entry, decision='exclude_unavailable_or_over_size_cap'))
                continue
            screen.append(dict(entry, decision='selected'))
            chosen = dict(entry, sequence=sequence)
            break
        if chosen is None:
            raise RuntimeError('No eligible bounded public training archive')
        selection = dict(
            version=1, selected=chosen, selection_screen=screen,
            selection_rule='First lexicographic AlphaFlow ATLAS TRAIN entry with 64-128 standard residues, less than 80% global aligned identities/max(sequence lengths) to all local PDB training/development/holdout manifest entries, and known 0<size<=256MiB official archive. No geometry/model-output selection.',
            alignment=dict(match=2, mismatch=-1, gap_open=-3, gap_extend=-.5, identity_denominator='max_sequence_lengths'),
            source_urls=dict(train_split=SPLIT_URL, published_download_code=DOWNLOAD_CODE, official_md_parameters=MD_PARAMETERS_URL),
            source_sha256={str(p): sha(p) for p in [split_path, code_path, *prior_paths, Path(__file__)]},
            planned_split={'R1': 'train', 'R2': 'development', 'R3': 'sealed_test'},
            scope='One-sequence MD capacity/trajectory-transfer pilot only. Not independent-protein generalization or a demonstration of equilibrium populations. Existing benchmark trajectories are not training data.'
        )
        atomic_json(selection_path, selection)
    chosen = selection['selected']
    archive = RUN / (chosen['name'] + '_protein.zip')
    if not archive.exists():
        tmp = archive.with_suffix('.part')
        start = time.perf_counter()
        received = 0
        with requests.get(chosen['url'], stream=True, timeout=(20, 45)) as response:
            response.raise_for_status()
            with tmp.open('wb') as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > CAP_BYTES:
                        raise RuntimeError('Public archive exceeded the declared download size cap')
                    stream.write(chunk)
        if received != chosen['bytes']:
            raise RuntimeError('Archive size changed; review source before extracting')
        tmp.replace(archive)
        print('DOWNLOAD_COMPLETE', received, round(time.perf_counter() - start, 2), flush=True)
    # Extract data members only; never execute scripts or deserialize arbitrary objects.
    destination = RUN / 'raw'
    destination.mkdir(exist_ok=True)
    extracted = []
    with zipfile.ZipFile(archive) as bundle:
        if sum(z.file_size for z in bundle.infolist()) > 4 * CAP_BYTES:
            raise RuntimeError('Unexpected uncompressed archive size')
        for info in bundle.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or '..' in relative.parts:
                raise RuntimeError('Unsafe public archive member')
            if info.is_dir() or not (relative.suffix.lower() in {'.pdb', '.xtc', '.mdp', '.tsv', '.txt'} or relative.name.upper().startswith('README')):
                continue
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with bundle.open(info) as src, path.open('wb') as dst:
                    while chunk := src.read(1024 * 1024):
                        dst.write(chunk)
            extracted.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
    provenance = dict(status='downloaded', name=chosen['name'], official_url=chosen['url'], archive_sha256=sha(archive), archive_bytes=archive.stat().st_size, selection_sha256=sha(selection_path), extracted=extracted, training_started=False)
    atomic_json(RUN / 'download_manifest.json', provenance)
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == '__main__':
    main()
